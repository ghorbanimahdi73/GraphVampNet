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
from torch_scatter import scatter_mean


args = buildParser().parse_args()


if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')


class GraphVampNet(nn.Module):

	def __init__(self, num_atoms=args.num_atoms, num_neighbors=args.num_neighbors, tau=args.tau,
				n_classes=args.num_classes, n_conv=args.n_conv, dmin=args.dmin, dmax=args.dmax, step=args.step,
				learning_rate=args.lr, batch_size=args.batch_size, n_epochs=args.epochs,
				h_a=args.h_a, h_g=args.h_g, atom_embedding_init='normal', use_pre_trained=False,
				pre_trained_weights_file=None, conv_type=args.conv_type, num_heads=args.num_heads, pool_backbone=args.pool_backbone):

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
		self.h_g = h_g
		self.gauss = GaussianDistance(dmin, dmax, step)
		self.h_b = self.gauss.num_features
		self.num_heads = num_heads
		#self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
		#self.atom_emb.weight.data.normal_()
		if args.conv_type == 'ConvLayer':
			self.convs = nn.ModuleList([ConvLayer(self.h_a, self.h_b) for _ in range(self.n_conv)])
		elif args.conv_type == 'NeighborAttention':
			self.convs = nn.ModuleList([NeighborAttention(self.h_a, self.h_b, self.num_heads) for _ in range(self.n_conv)])
		self.conv_type = conv_type
		self.conv_activation = nn.ReLU()
		self.num_neighbors = num_neighbors
		self.fc_classes = nn.Linear(self.h_a, n_classes)
		self.init = atom_embedding_init
		self.use_pre_trained = use_pre_trained

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
		if args.conv_type == 'ConvLayer':
			for idx in range(self.n_conv):
				atom_emb = self.convs[idx](atom_emb, nbr_emb, nbr_adj_list)
		
		elif args.conv_type == 'NeighborAttention':
			for idx in range(self.n_conv):
				atom_emb = self.convs[idx](atom_emb, nbr_emb, nbr_adj_list)


		emb = self.conv_activation(atom_emb)
		# [B, N, h_a]

		if args.use_backbone_atoms:
			emb = self.pool_amino(emb)
			emb = self.amino_emb(emb)

		prot_emb = self.pooling(emb)
		# [B, h_a]
		class_logits = self.fc_classes(prot_emb)
		# [B, n_classes]
		class_probs = F.softmax(class_logits, dim=-1)
		return class_probs