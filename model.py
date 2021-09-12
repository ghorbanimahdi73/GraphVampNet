import torch
import numpy as np
import deeptime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import mdshare
from torch.utils.data import DataLoader
import json
from sklearn.neighbors import BallTree
from args import buildParser
from layers import GaussianDistance, ConvLayer, NeighborAttention


args = buildParser().parse_args()


if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

print('device is use is :', device)

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


class GraphVampNet(nn.Module):

	def __init__(self, num_atoms=args.num_atoms, num_neighbors=args.num_neighbors, tau=args.tau,
				n_classes=args.num_classes, n_conv=args.n_conv, dmin=args.dmin, dmax=args.dmax, step=args.step,
				learning_rate=args.lr, batch_size=args.batch_size, n_epochs=args.epochs,
				h_a=args.h_a, atom_embedding_init='normal', use_pre_trained=False,
				pre_trained_weights_file=None):

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
		#self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
		#self.atom_emb.weight.data.normal_()
		self.convs = nn.ModuleList([ConvLayer(self.h_a, self.h_b) for _ in range(self.n_conv)])
		self.conv_activation = nn.ReLU()
		self.num_neighbors = num_neighbors
		self.fc_classes = nn.Linear(self.h_a, n_classes)
		self.init = atom_embedding_init
		self.use_pre_trained = use_pre_trained
		
		if use_pre_trained:
			self.pre_trained_emb(pre_trained_weights_file)
		else:
			self.atom_emb = nn.Embedding(num_neighbors=self.num_atoms, embedding_dim=self.h_a)
			self.init_emb()

	def pre_trained_emb(self, file):
		'''
		loads the pre-trained node embedings from a file
		For now we are not freezing the pre-trained embeddings since
		we are going to update it in the graph convolution
		'''
		with open(self.pre_trained_weights_file) as f:
			loaded_emb = json.load(f)

		embed_list = [torch.tensor(value, dtype=torch.float32) for value in loaded_emb.values()]
		self.atom_embeddings = torch.stack(embed_list, dim=0)
		self.h_init = self.atom_embeddings.shape[-1] # dimension atom embedding init
		self.atom_emb = nn.Embedding.from_pretrained(self, atom_embeddings, freeze=False)
		self.embedding = nn.Linear(self.h_init, self.h_a)


	def init_emb(self):
		'''
		Initialize random embedding for the atoms
		'''
		#--------------initialization for the embedding--------------
		if init == 'normal':
			self.atom_emb.weight.data.normal_()

		elif init == 'xavier_normal'
			self.atom_emb.weight.data._xavier_normal()

		elif init == 'uniform':
			self.atom_emb.weight.data._uniform()
		#------------------------------------------------------------

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
		# atom_emb [B,N,h_a]
		# nbr_emb [B,N,M,h_b]
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