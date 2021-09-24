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
from utils_vamp import estimate_koopman_op, get_ck_test, plot_ck_test
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
parser.add_argument('--attention', default=False, action='store_true', help='Whether to use attention in E(n) equivariance NN')
parser.add_argument('--normalize', default=False, action='store_true', help='Whether to normalize the coordinates')


if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

# ignore deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

args =  parser.parse_args()



if not os.path.exists(args.save_folder):
	print('making the folder for saving checkpoints')
	os.makedirs(args.save_folder)

with open(args.save_folder+'/args.txt','w') as f:
	f.write(str(args))

meta_file = os.path.join(args.save_folder, 'metadata.pkl')
pickle.dump({'args': args}, open(meta_file, 'wb'))

myinds = []
myinds1 = np.load('inds_BBA_7.npz')['arr_0'] # the input data for indices of edges
myinds2 = np.load('inds_BBA_7.npz')['arr_1']


# [num_traj, T, N, M]
mypos = []
mypos1 = np.load('pos_BBA.npz')['arr_0'] # the input data for the positions of atoms
mypos2 = np.load('pos_BBA.npz')['arr_1']


# make the dense adjacency matrix ################
myadj1 = np.zeros((myinds1.shape[0], myinds1.shape[1], myinds1.shape[1]))
for j in range(myinds1.shape[0]):
	for k in range(myinds1.shape[1]):
		myadj1[j,k,myinds1[j,k]] = 1

myadj2 = np.zeros((myinds2.shape[0], myinds2.shape[1], myinds2.shape[1]))
for j in range(myinds2.shape[1]):
	for k in range(myinds2.shape[2]):
		myadj2[j,k,myinds2[j,k]] = 1

#############################################
myadj = list((myadj1, myadj2))
mypos = list((mypos1, mypos2))

num_trajs = 2
data = []
for i in range(num_trajs):
	data.append(torch.cat((torch.tensor(mypos[i], dtype=torch.float32), torch.tensor(myadj[i], dtype=torch.int32)), axis=-1))

dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)
n_val = int(len(dataset)*args.val_frac)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])

loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)


class GraphVampNet(nn.Module):
	def __init__(self, num_atoms=args.num_atoms, num_neighbors=args.num_neighbors, tau=args.tau, n_classes=args.num_classes,
				 n_conv=args.n_conv, learning_rate=args.lr, batch_size=args.batch_size, n_epochs=args.epochs, h_a=args.h_a,
				 h_g=args.h_g, atom_embedding_init='normal', use_pre_trained=False, pre_trained_weights_file=None,
				 dont_pool_backbone=args.dont_pool_backbone, attention=args.attention, normalize=False):

		super(GraphVampNet, self).__init__()
		self.num_atoms = num_atoms
		self.num_neighbors = num_neighbors
		self.tau = tau
		self.n_classes = n_classes
		self.n_conv = n_conv
		self.learning_rate =learning_rate
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.h_a = h_a
		self.h_g = h_g
		self.atom_embedding_init = atom_embedding_init
		self.use_pre_trained = use_pre_trained
		self.attention = attention
		self.normalize = normalize
		self.conv_activation = nn.ReLU()
		self.fc_classes = nn.Linear(self.h_a, n_classes)
		self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
		self.egnn = EGNN(in_node_nf=self.h_a, hidden_nf=self.h_a, out_node_nf=self.h_a, in_edge_nf=1, device=device, act_fn=nn.SiLU(),
			             n_layers=args.n_conv, residual=True, attention=args.attention, normalize=args.normalize, tanh=False)

		self.init = atom_embedding_init

		if args.use_backbone_atoms:
			self.amino_emb = nn.Linear(self.h_a, self.h_g)

		if use_pre_trained:
			self.pre_trained_emb(pre_trained_weights_file)
		else:
			self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
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
		if self.init == 'normal':
			self.atom_emb.weight.data.normal_()

		elif self.init == 'xavier_normal':
			self.atom_emb.weight.data._xavier_normal()

		elif self.init == 'uniform':
			self.atom_emb.weight.data._uniform()
		#------------------------------------------------------------

	def pooling(self, atom_emb):
		'''
		global pooling for graph features
		'''
		summed = torch.sum(atom_emb, dim=1)
		return summed / self.num_atoms



	def pool_amino(self, atom_emb):
		'''
		pooling the features of atoms in each amino acid to get a feature vector for each residue
		parameters:
		--------------------------
		atom_emb: embedding of atoms [B,N,h_a]
		residue_atom_idx: mapping between every atom and every residue in the protein
				size: [N] example [0,0,0,1,1,1,2,2,2] for N=6 and NA=3

		Returns: 
		--------------------------
		pooled features of amino acids in the graph
		[B, Na, h_a]
		'''

		B = atom_emb.shape[0]
		N = atom_emb.shape[1]
		h_a = atom_emb.shape[2]

		residue_atom_idx = torch.arange(N).repeat(1,3)
		residue_atom_idx = residue_atom_idx.view(3,N).T.reshape(-1,1).squeeze(-1)
		Na = torch.max(residue_atom_idx)+1 # number of residues
		pooled = scatter_mean(atom_emb, residue_atom_idx, out=atom_emb.new_zeros(B,Na,h_a), dim=1)
		return pooled


	def forward(self, data):
		# data is [batch-size, num-atoms, num_atoms+M]
		B = data.shape[0]
		N = data.shape[1]

		atom_emb_idx = torch.arange(N).repeat(B,1).to(device)
		atom_emb = self.atom_emb(atom_emb_idx).view(B*self.num_atoms, self.h_a).to(device)
		x = data[:,:,:3].view(B*self.num_atoms, 3).to(torch.float32)

		edge_inds = data[:, :, 3:].to(torch.int32).to(device) # dense adjacency matrix

		#t1 = time.time()
		#edges, edge_attr = get_edges_batch(edge_inds, B)
		#t2 = time.time()
		#print('time_1: ', t2-t1)
		offset, row, col = (edge_inds>0).nonzero().t().to(device)
		
		row += offset * N
		col += offset * N 
		edge_index = [row, col]

		edge_attr = torch.ones(len(row)).unsqueeze(-1).to(device)

		atom_emb, x = self.egnn(atom_emb, x, edge_index, edge_attr=edge_attr)

		emb = atom_emb.view(B,N,self.h_a)

		#emb = self.conv_activation(atom_emb)

		if args.use_backbone_atoms:
			emb = self.pool_amino(emb)
			emb = self.amino_emb(emb)

		prot_emb = self.pooling(emb)

		class_logits = self.fc_classes(prot_emb)
		class_probs = F.softmax(class_logits, dim=-1)
		return class_probs


lobe = GraphVampNet()
lobe_timelagged = deepcopy(lobe).to(device)

lobe = lobe.to(device)
vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device, optimizer='Adam', score_method='VAMP2')


def train(train_loader, n_epochs=args.epochs, validation_loader=None):
	for epoch in tqdm(range(n_epochs)):
		for batch_0, batch_t in train_loader:
			vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))

		if validation_loader is not None:
			with torch.no_grad():
				scores = []
				for val_batch in validation_loader:
					scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))

				mean_score = torch.mean(torch.stack(scores))
				vampnet._validation_scores.append((vampnet._step, mean_score.item()))

		if args.save_checkpoints:
			torch.save({
				'epoch': epoch,
				'state_dict': lobe.state_dict()
				}, args.save_folder+'/logs_'+str(epoch)+'.pt')

	return vampnet.fetch_model()

def count_parameters(model):
	'''
	count the number of parameters in the model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('number of parameters', count_parameters(lobe))

plt.set_cmap('jet')

#model = train(train_loader=loader_train, n_epochs=args.epochs, validation_loader=loader_val)


if not args.train and os.path.isfile(args.trained_model):
	print('Loading model')
	checkpoint = torch.load(args.trained_model)
	lobe.load_state_dict(checkpoint['state_dict'])
	lobe_timelagged = deepcopy(lobe).to(device=device)
	lobe = lobe.to(device)
	lobe.eval()
	lobe_timelagged.eval()
	vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device)
	model = vampnet.fetch_model()

elif args.train:
	model = train(train_loader=loader_train, n_epochs=args.epochs, validation_loader=loader_val)

	# save the training and validation scores
	with open(args.save_folder+'/train_scores.npy', 'wb') as f:
		np.save(f, vampnet.train_scores)

	with open(args.save_folder+'/validation_scores.npy', 'wb') as f:
		np.save(f, vampnet.validation_scores)



plt.loglog(*vampnet.train_scores.T, label='training')
plt.loglog(*vampnet.validation_scores.T, label='validation')
plt.xlabel('step')
plt.ylabel('score')
plt.legend()
plt.savefig(args.save_folder+'/scores.png')


data_np = []
for i in range(len(data)):
	data_np.append(data[i].cpu().numpy())


# for the analysis part create an iterator for the whole dataset to feed in batches
whole_dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data_np)
whole_dataloder = DataLoader(whole_dataset, batch_size=5000, shuffle=False)



# for plotting the implied timescales
lagtimes = np.arange(1, 201,5, dtype=np.int32)
timescales = []
for lag in tqdm(lagtimes):
	vamp = VAMP(lagtime=lag, observable_transform=model)
	whole_dataset = TrajectoryDataset.from_trajectories(lagtime=lag, data=data_np)
	whole_dataloder = DataLoader(whole_dataset, batch_size=5000, shuffle=False)
	for batch_0, batch_t in whole_dataloder:
		vamp.partial_fit((batch_0.numpy(), batch_t.numpy()))
#
	covariances = vamp._covariance_estimator.fetch_model()
	ts = vamp.fit_from_covariances(covariances).fetch_model().timescales(k=5)
	timescales.append(ts)


f, ax = plt.subplots(1, 1)
ax.semilogy(lagtimes, timescales)
ax.set_xlabel('lagtime')
ax.set_ylabel('timescale / step')
ax.fill_between(lagtimes, ax.get_ylim()[0]*np.ones(len(lagtimes)), lagtimes, alpha=0.5, color='grey');
f.savefig(args.save_folder+'/ITS.png')


probs = []
for i in range(len(data_np)):
	state_prob = model.transform(data_np[i])
	probs.append(state_prob)


max_tau = 200
lags = np.arange(1, max_tau, 2)

its = get_its(probs, lags)
plot_its(its, lags, ylog=False, save_folder=args.save_folder)



steps = 8
tau_msm = 100
predicted, estimated = get_ck_test(probs, steps, tau_msm)
n_classes = args.num_classes 
plot_ck_test(predicted, estimated, n_classes, steps, tau_msm, args.save_folder)
