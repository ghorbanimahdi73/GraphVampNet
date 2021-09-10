import torch
import numpy as np
import deeptime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import mdshare
from torch.utils.data import DataLoader


if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')



ala_coords_file = mdshare.fetch(
    "alanine-dipeptide-3x250ns-heavy-atom-positions.npz", working_directory="data"
)
with np.load(ala_coords_file) as fh:
    data = [fh[f"arr_{i}"].astype(np.float32) for i in range(3)]

dihedral_file = mdshare.fetch(
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz", working_directory="data"
)
with np.load(dihedral_file) as fh:
    dihedral = [fh[f"arr_{i}"] for i in range(3)]


# reshape the data to be in share list of [N,num_atoms,3]
data_reshaped = []
for i in range(len(data)):
    temp = data[i].reshape(data[0].shape[0], 3, 10).swapaxes(1,2)
    data_reshaped.append(temp)
#-----------------------------------------------------


def get_nbrs(all_coords, num_neighbors=5):
	'''
	inputs: a trajectory or list of trajectories [N,num_atoms,dim]

	Returns:
		if all_coords is a list:
			list of trajectories of ditances and indices 
		else:
			trajectory of distances and indices

	[N, num_atoms, num_neighbors]
	'''
	k_nbr=num_neighbors+1
	if type(all_coords) == list:
		all_dists = []
		all_inds = []
		for i in range(len(all_coords)):
			dists = []
			inds = []
			tmp_coords = all_coords[i]
			for j in range(len(tmp_coords)):
				tree = BallTree(tmp_coords[j], leaf_size=3)
				dist, ind = tree.query(tmp_coords[j], k=k_nbr)
				dists.append(dist[:,1:])
				inds.append(ind[:,1:])

			dists = np.array(dists)
			inds = np.array(inds)
			all_dists.append(dists)
			all_inds.append(inds)
	else:
		all_inds = []
		all_dists = []
		for i in range(len(all_coords)):
			tree = BallTree(all_coords[i], leaf_size=3)
			dist , ind = tree.query(all_coords[i], k=k_nbr)
			dists.append(dist[:,1:])
			inds.append(ind[:,1:])
			all_dists = np.array(dists)
			all_inds = np.array(inds)

	return all_dists, all_inds


#dists, inds = get_nbrs(data_reshaped)
#  list of trajectories [T,N,M]
#np.savez('dists.npz', dists)
#np.savez('inds.npz', inds)

dists, inds = np.load('dists.npz')['arr_0'], np.load('inds.npz')['arr_0']
mydists = torch.from_numpy(dists).to(device)
myinds = torch.from_numpy(inds).to(device)
data = []
for i in range(len(dists)):
	data.append(torch.cat((mydists[i],myinds[i]), axis=-1))

data_np = []
for i in range(len(dists)):
	data_np.append(np.concatenate((dists[i],inds[i]), axis=-1))

# data is a list of trajectories [T,N,M+M]

class ConvLayer(nn.Module):
	'''
	h_a: atom embedding
	h_b: bond embedding

	returns: 
	atom_embedding after convolution
	[B,N,atom_fea_len]
	'''
	def __init__(self, h_a, h_b):
		super(ConvLayer, self).__init__()
		self.h_a = h_a
		self.h_b = h_b
		self.fc_full = nn.Linear(2*self.h_a+self.h_b, 2*self.h_a)
		self.sigmoid = nn.Sigmoid()
		self.activation_hidden =nn.ReLU()
		self.bn_hidden = nn.BatchNorm1d(2*self.h_a)
		self.bn_output = nn.BatchNorm1d(self.h_a)
		self.activation_output = nn.ReLU()

	def forward(self, atom_emb, nbr_emb, nbr_adj_list):
		N, M = nbr_adj_list.shape[1:]
		B = atom_emb.shape[0]

		atom_nbr_emb = atom_emb[torch.arange(B).unsqueeze(-1), nbr_adj_list.to(torch.long).view(B,-1)].view(B,N,M,self.h_a).to(device)
		# [B,N,M,h_a]
		total_nbr_emb = torch.cat([atom_emb.unsqueeze(2).expand(B,N,M,self.h_a), atom_nbr_emb, nbr_emb],dim=-1).to(torch.float32)
		# [B,N,M,2*h_a+h_b]

		total_gated_emb = self.fc_full(total_nbr_emb)
		total_gated_emb = self.bn_hidden(total_gated_emb.view(-1,self.h_a*2)).view(B,N,M,self.h_a*2)
		nbr_filter, nbr_core = total_gated_emb.chunk(2, dim=3)
		nbr_filter = self.sigmoid(nbr_filter)
		nbr_core = self.activation_hidden(nbr_core)
		nbr_sumed = torch.sum(nbr_filter*nbr_core, dim=2)
		nbr_sumed = self.bn_output(nbr_sumed.view(-1, self.h_a)).view(B,N,self.h_a)
		out = self.activation_output(atom_emb+nbr_sumed)
		return out

class GaussianDistance(object):
	'''
	expand the distance by gaussian basis
	'''
	def __init__(self, dmin, dmax, step, var=None):
		'''
		parameters:
		-------------------
		dmin: float, minimum interatomic distance
		dmax: float, maximum interatomic distance
		step: float, step size for the gaussian filter
		'''
		assert dmin < dmax
		assert dmax - dmin > step
		self.filter = torch.arange(dmin, dmax+step, step)
		self.num_features = len(self.filter)
		if var is None:
			var = step
		self.var = var

	def expand(self, distance):
		'''
		apply gaussian distance filter to a numpy array distance
		parameters:
		-----------------
		N: number of atoms
		M: number of neighbors
		B: batch-size
		distance: shape [B, N, M]

		returns:
		expanded distance with shape [B, N, M, bond_fea_len]
		'''
		return torch.exp(-(torch.unsqueeze(distance,-1).to(device)-self.filter.to(device))**2/self.var**2)


class GraphVampNet(nn.Module):

	def __init__(self, num_atoms=10, num_neighbors=5, tau=1,
				n_classes=6, n_conv=3, dmin=0., dmax=3., step=0.2,
				learning_rate=0.001, batch_size=10000, n_epochs=100,
				h_a=16):

		super(GraphVampNet, self).__init__()
		self.num_atoms = num_atoms
		self.num_neighbors = num_neighbors
		self.tau = tau
		self.n_classes= n_classes
		self.n_conv = n_conv
		self.dmin = dmin
		self.dmax = dmax
		self.step = step
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.h_a = h_a
		self.gauss = GaussianDistance(dmin, dmax, step)
		self.h_b = self.gauss.num_features
		self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
		self.atom_emb.weight.data.normal_()
		self.convs = nn.ModuleList([ConvLayer(self.h_a, h_b=16) for _ in range(self.n_conv)])
		self.conv_activation = nn.ReLU()
		self.num_neighbors = num_neighbors
		self.fc_classes = nn.Linear(self.h_a, n_classes)
		

	def pooling(self, atom_emb):

		summed = torch.sum(atom_emb, dim=1)
		return summed / self.num_atoms

	def forward(self,data):

		n = data.shape[-1]
		nbr_adj_dist = data[:,:,:n//2]
		nbr_adj_list = data[:,:,n//2:]
		N = nbr_adj_list.shape[1]
		B = nbr_adj_list.shape[0]

		nbr_emb = self.gauss.expand(nbr_adj_dist)
		# this is the edge embedding

		atom_emb_idx = torch.arange(N).repeat(B,1).to(device)
		atom_emb = self.atom_emb(atom_emb_idx)
		for idx in range(self.n_conv):
			atom_emb = self.convs[idx](atom_emb, nbr_emb, nbr_adj_list)

		atom_emb = self.conv_activation(atom_emb)
		# [B, N, h_a]
		prot_emb = self.pooling(atom_emb)
		# [B, h_a]
		class_logits = self.fc_classes(prot_emb)
		# [B, n_classes]
		class_probs = F.softmax(class_logits, dim=-1)
		return class_probs

from deeptime.util.data import TrajectoryDataset
dataset = TrajectoryDataset.from_trajectories(lagtime=1, data=data)

n_val = int(len(dataset)*0.3)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])

loader_train = DataLoader(train_data, batch_size=10000, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

lobe = GraphVampNet()
from copy import deepcopy
lobe_timelagged = deepcopy(lobe).to(device=device)
lobe = lobe.to(device)


from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP

vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=0.0005, device=device)

model = vampnet.fit(loader_train, n_epochs=100, validation_loader=loader_val, progress=tqdm).fetch_model()

plt.set_cmap('jet')

plt.loglog(*vampnet.train_scores.T, label='training')
plt.loglog(*vampnet.validation_scores.T, label='validation')
plt.xlabel('step')
plt.ylabel('score')
plt.legend()

plt.savefig('scores.png')

state_probabilities = model.transform(data[0])

f, axes = plt.subplots(3,2, figsize=(12,16))
for i, ax in enumerate(axes.flatten()):
	ax.scatter(*dihedral[0][::5].T, c=state_probabilities[...,i][::5])
	ax.set_title(f'state {i+1}')
f.savefig('state_prob.png')


fig, ax = plt.subplots(1,1, figsize=(8,10))
assignments = state_probabilities.argmax(1)
plt.scatter(*dihedral[0].T, c=assignments, s=5, alpha=0.1)
plt.title('Transformed state assignments')
plt.savefig('assignments.png')



lagtimes = np.arange(1,201, dtype=np.int32)
timescales = []
for lag in tqdm(lagtimes):
	ts = VAMP(lagtime=lag, observable_transform=model).fit(data).fetch_model().timescales(k=5)
	timescales.append(ts)

f, ax = plt.subplots(1, 1)
ax.semilogy(lagtimes, timescales)
ax.set_xlabel('lagtime')
ax.set_ylabel('timescale / step')
ax.fill_between(lagtimes, ax.get_ylim()[0]*np.ones(len(lagtimes)), lagtimes, alpha=0.5, color='grey');
f.savefig('ITS.png')