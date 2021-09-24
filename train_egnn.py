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
from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP
from copy import deepcopy
import os
import pickle
import warnings
from deeptime.util.data import TrajectoryDataset
from egnn import EGNN, E_GCL
import time
import argparse
from utils_vamp import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=5000, help='batch-size for training')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
parser.add_argument('--hidden', type=int, default=16, help='number of hidden neurons')
parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms')
parser.add_argument('--num-classes', type=int, default=6, help='number of coarse-grained classes')
parser.add_argument('--save-folder', type=str, default='egnn_1', help='Where to save the trained model')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--atom_init', type=str, default=None, help='inital embedding for atoms file')
parser.add_argument('--h_a', type=int, default=16, help='Atom hidden embedding dimension')
parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors for each atom in the graph')
parser.add_argument('--n_conv', type=int, default=4, help='Number of convolution layers')
parser.add_argument('--save_checkpoints', default=True,  action='store_true', help='If True, stores checkpoints')
parser.add_argument('--tau', default=1, type=int, help='lag time for the model')
parser.add_argument('--val-frac', default=0.3, type=float, help='fraction of dataset for validation')
parser.add_argument('--trained-model', default=None, type=str, help='path to the trained model for loading')
parser.add_argument('--train', default=False, action='store_true', help='Whether to train the model or not')
parser.add_argument('--use_backbone_atoms', default=False, action='store_true', help='Whether to use all the back bone atoms for training')
parser.add_argument('--dont-pool-backbone', default=False, action='store_true', help='Whether not to pool backbone atoms')
parser.add_argument('--h_g', type=int, default=8, help='Number of embedding dimension after backbone pooling')




if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

# ignore deprecation warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

args =  parser.parse_args()

with open(args.save_folder+'/args.txt','w') as f:
	f.write(str(args))

if not os.path.exists(args.save_folder):
	print('making the folder for saving checkpoints')
	os.makedirs(args.save_folder)

meta_file = os.path.join(args.save_folder, 'metadata.pkl')
pickle.dump({'args': args}, open(meta_file, 'wb'))



#---------------load Ala-dipeptide traj ---------------
ala_coords_file = mdshare.fetch(
    "alanine-dipeptide-3x250ns-heavy-atom-positions.npz", working_directory="data")

with np.load(ala_coords_file) as fh:
    data = [fh[f"arr_{i}"].astype(np.float32) for i in range(3)]
#
dihedral_file = mdshare.fetch(
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz", working_directory="data")

with np.load(dihedral_file) as fh:
    dihedral = [fh[f"arr_{i}"] for i in range(3)]

# -------------------------------------------------------

def get_edges(ind_matrix):
	rows, cols = [], []
	for i in range(10):
		for j in range(5):
			rows.append(i)
			cols.append(ind_matrix[i,j])
	edges = [rows, cols]
	return edges

def get_edges_batch(batch_inds, batch_size):
	t1 = time.time()
	N = batch_inds.shape[1] # number of atoms
	rows, cols = [], []
	for i in range(batch_size):
		edges = get_edges(batch_inds[i])
		edge_attr = torch.ones(len(edges[0])*batch_size, 1).to(device).to(torch.int32)
		edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
		rows.append(edges[0]+N*i)
		cols.append(edges[1]+N*i)
	edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]
	t2 = time.time()
	print('get-batch: ', t2-t1)
	return edges, edge_attr


myinds = np.load('inds.npz')['arr_0']

# make the dense adjacency matrix ################
myadj = np.zeros((3, 250000, 10, 10))
for i in range(len(myinds)):
	for j in range(myinds.shape[1]):
		for k in range(10):
			myadj[i,j,k,myinds[i,j,k]] = 1

myadj = torch.tensor(myadj).to(device)
#############################################

data_reshaped = []
for i in range(len(data)):
	temp = data[i].reshape(data[0].shape[0], 3, 10).swapaxes(1,2)
	data_reshaped.append(temp)

data_reshaped = np.array(data_reshaped)
mypos = torch.from_numpy(data_reshaped).to(device)

data = []
for i in range(len(myadj)):
	data.append(torch.cat((mypos[i], myadj[i]), axis=-1))

dataset = TrajectoryDataset.from_trajectories(lagtime=1, data=data)
n_val = int(len(dataset)*0.3)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])

loader_train = DataLoader(train_data, batch_size=10000, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)


class GraphVampNet(nn.Module):
	def __init__(self, num_atoms=10, num_neighbors=5, tau=1, n_classes=6, n_conv=4,
				learning_rate=0.0005, batch_size=10000, n_epochs=30, h_a=16):
		super(GraphVampNet, self).__init__()
		self.num_atoms = num_atoms
		self.num_neighbors = num_neighbors
		self.tau = tau
		self.n_classes = n_classes
		self.n_conv = n_conv
		self.learning_rate =learning_rate
		self.batch_size = batch_size
		self.h_a = h_a

		self.conv_activation = nn.ReLU()
		self.fc_classes = nn.Linear(self.h_a, n_classes)
		self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
		self.atom_emb.weight.data.normal_()
		self.egnn = EGNN(in_node_nf=self.h_a, hidden_nf=16, out_node_nf=self.h_a, in_edge_nf=1, device='cuda')

	def pooling(self, atom_emb):
		summed = torch.sum(atom_emb, dim=1)
		return summed / self.num_atoms

	def forward(self, data):
		# data is [batch-size, num-atoms, num_atoms+M]
		B = data.shape[0]
		N = data.shape[1]

		atom_emb_idx = torch.arange(N).repeat(B,1).to(device)
		atom_emb = self.atom_emb(atom_emb_idx).view(B*self.num_atoms, self.h_a).to(device)
		x = data[:,:,:3].view(B*self.num_atoms, 3).to(torch.float32)

		edge_inds = data[:, :, 3:].to(torch.int32).to(device) # adjacency matrix

		#t1 = time.time()
		#edges, edge_attr = get_edges_batch(edge_inds, B)
		#t2 = time.time()
		#print('time_1: ', t2-t1)
		offset, row, col = (edge_inds>0).nonzero().t().to(device)
		
		row += offset * N
		col += offset * N 
		edge_index = [row, col]

		edge_attr = torch.ones(len(row)).unsqueeze(-1).to(device)

		atom_emb, x = self.egnn(atom_emb, x, edge_index, edge_attr)

		atom_emb = atom_emb.view(B,N,self.h_a)
		prot_emb = self.pooling(atom_emb)

		class_logits = self.fc_classes(prot_emb)
		class_probs = F.softmax(class_logits, dim=-1)
		return class_probs

lobe = GraphVampNet()
lobe_timelagged = deepcopy(lobe).to(device)

lobe = lobe.to(device)
vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=0.0005, device=device, optimizer='Adam', score_method='VAMP2')


def train(train_loader, n_epochs, validation_loader=None):
	for epoch in tqdm(range(n_epochs)):

		t=0
		for batch_0, batch_t in train_loader:
			t = t+1
			print(t)
			vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))

		if validation_loader is not None:
			with torch.no_grad():
				scores = []
				for val_batch in validation_loader:
					scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))

				mean_score = torch.mean(torch.stack(scores))
				vampnet._validation_scores.append((vampnet._step, mean_score.item()))

	return vampnet.fetch_model()

def count_parameters(model):
	'''
	count the number of parameters in the model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('number of parameters', count_parameters(lobe))


model = train(train_loader=loader_train, n_epochs=30, validation_loader=loader_val)


plt.loglog(*vampnet.train_scores.T, label='training')
plt.loglog(*vampnet.validation_scores.T, label='validation')
plt.xlabel('step')
plt.ylabel('score')
plt.legend()
plt.savefig('egnn_results'+'/scores.png')







data_np = []
for i in range(len(data)):
	data_np.append(data[i].cpu().numpy())


probs = []
for i in range(len(data_np)):
	state_prob = model.transform(data_np[i])
	probs.append(state_prob)


# for the analysis part create an iterator for the whole dataset to feed in batches
whole_dataset = TrajectoryDataset.from_trajectories(lagtime=1, data=data_np)
whole_dataloder = DataLoader(whole_dataset, batch_size=10000, shuffle=False)



# for plotting the implied timescales
lagtimes = np.arange(1,101,2, dtype=np.int32)
timescales = []
for lag in tqdm(lagtimes):
	vamp = VAMP(lagtime=lag, observable_transform=model)
	whole_dataset = TrajectoryDataset.from_trajectories(lagtime=lag, data=data_np)
	whole_dataloder = DataLoader(whole_dataset, batch_size=10000, shuffle=False)
	for batch_0, batch_t in whole_dataloder:
		vamp.partial_fit((batch_0.numpy(), batch_t.numpy()))

	covariances = vamp._covariance_estimator.fetch_model()
	ts = vamp.fit_from_covariances(covariances).fetch_model().timescales(k=5)
	timescales.append(ts)


f, ax = plt.subplots(1, 1)
ax.semilogy(lagtimes, timescales)
ax.set_xlabel('lagtime')
ax.set_ylabel('timescale / step')
ax.fill_between(lagtimes, ax.get_ylim()[0]*np.ones(len(lagtimes)), lagtimes, alpha=0.5, color='grey');
f.savefig('egnn_results'+'/ITS.png')


ax_tau = 200
lags = np.arange(1, max_tau, 2)

its = get_its(probs, lags)
plot_its(its, lags, ylog=False, save_folder='egnn_results')

steps = 8
tau_msm = 100
predicted, estimated = get_ck_test(probs, steps, tau_msm)
n_classes = 6 
plot_ck_test(predicted, estimated, n_classes, steps, tau_msm, 'egnn_results')