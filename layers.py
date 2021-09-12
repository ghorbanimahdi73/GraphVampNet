from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')


def gather_nodes(nodes, neighbor_idx):
	'''
	given node-features [B,N,h_a] and neighbor_dix [B,N,M] this find the neighbors of each
	node and concatentates their features together:
	returns:
	[B,N,M,h_a]
	'''
	neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1)) # [B,NM]
	neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1,-1,nodes.size(2)) # [B,NM,h_a]

	neighbor_features = torch.gather(nodes, 1, neighbors_flat)
	neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3]+[-1])
	return neighbor_features

def cat_neighbor_nodes(h_nodes, h_neighbors, E_idx):
	'''
	given node embeddings [h_nodes] and neighbor embeddings [h_neighbors] and indexes this concatentates the features
	params:
		h_nodes: [B,N,h_nodes]
		h_neighbors: [B,N,M,h_edges]
		E_idx: [B,N,M] 
	returns:
		concatenated features [B,N,M,h_nodes+h_edges]
	'''
	h_nodes = gather_nodes(h_nodes, E_idx)
	h_nn = torch.cat([h_neighbors, h_nodes], -1)
	return h_nn


class Normalize(nn.Module):
	def __init__(self, features, epsilon=1e-6):
		super(Normalize, self).__init__()
		self.gain = nn.Parameter(torch.ones(features))
		self.bias = nn.Parameter(torch.zeros(features))
		self.epsilon = epsilon

	def forward(self, x, dim=1):
		mu = x.mean(dim, keepdim=True)
		sigma = torch.sqrt(x.var(dim, keepdim=True)+self.epsilon)
		gain = self.gain
		bias = self.bias
		# Reshape
		if dim != -1:
			shape = [-1] *len(mu.size())
			shape[dim] = self.gain.size()[0]
			gain = gain.view(shape)
			bias = bias.view(shape)
		return gain*(x-mu)/(sigma+self.epsilon) + bias


class NeighborAttention(nn.Module):
	def __init__(self, num_hidden, num_in, num_heads=4):
		'''
		num_hidden: number of features for atoms
		num_in: number of features for edges
		num_heads: number of heads in multi-head attention
		'''
		super(NeighborAttention, self).__init__()
		self.num_heads = num_heads
		self.num_hidden = num_hidden

		# self-attention layers: {queries, keys, values, output}
		self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
		self.W_K = nn.Linear(num_in, num_hidden, bias=False)
		self.W_V = nn.Linear(num_in, num_hidden, bias=False)
		self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
		return

	def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
		''' numercially stable masked softmax '''
		negative_inf = np.finfo(np.float32).min
		attend_logits = torch.where(mask_attend>0, attend_logits, torch.tensor(negative_inf).to(device))
		attend = F.softmax(attend_logits, dim)
		attend = mask_attend * attend
		return attend

	def forward(self, h_V, h_E, mask_attend=None):
		''' self attention graph structure O(NK)
		args:
			h_V: node features [batch_size, num_nodes, n_hidden]
			h_E: neighbor featues [batch_size, num_nodes, num_neighbors, N_in]
			mask_attend: mask for attention [batch_size, num_nodes, num_neighbors] 
		returns:
			h_V_t: node update after neighborattention
		'''
		# dimensions
		n_batch, n_nodes, n_neighbors = h_E.shape[:3]
		n_heads = self.num_heads
		d = int(self.num_hidden/n_heads)

		Q = self.W_Q(h_V.to(torch.float32)).view([n_batch, n_nodes, 1, n_heads, 1, d]) # [B,N,1,n_heads,1,d]
		K = self.W_K(h_E.to(torch.float32)).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
		V = self.W_V(h_E.to(torch.float32)).view([n_batch, n_nodes, n_neighbors, n_heads, d])

		# attention with scaled inner product
		attend_logits = torch.matmul(Q,K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
		#[B,N,n_heads,n_nbrs]
		attend_logits = attend_logits / np.sqrt(d)
		if mask_attend is not None:
			# masked softmax
			mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1).to(device)
			attend = self._masked_softmax(attend_logits, mask)
		else:
			attend = F.softmax(attend_logits, -1).to(device)
		#[B,N,n_heads,n_nbrs]

		h_V_update = torch.matmul(attend.unsqueeze(-2).to(torch.float32), V.transpose(2,3))
		h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
		h_V_update = self.W_O(h_V_update)
		return h_V_update


class ConvLayer(nn.Module):
	'''
	h_a: atom embedding [B,N,h_a]
	h_b: bond embedding [B,N,M,h_b]

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