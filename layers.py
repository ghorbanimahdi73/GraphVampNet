# Author: Mahdi Ghorbani

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

def LinearLayer(d_in, d_out, bias=True, activation=None, dropout=0, weight_init='xavier'):
	'''
	Linear layer function

	Parameters:
	---------------------
	d_in: int, input dimension
	d_out: int, output dimension
	bias: bool, (default=True) Whether or not to add bias
	activation: torch.nn.Module() (default=None)
		activation function of the layer
	dropout: float (default=0)
		the dropout to be added 
	weight_init: str,, float, or nn.init function (default='xavier')
		specifies the initialization of the layer weights. If float or int is passed a constant initialization
		is used.

	Returns:
	----------------------
	seq: list of torch.nn.Module instances
		the full linear layer including activations and optional dropout
	'''
	seq = [nn.Linear(d_in, d_out, bias=bias)]
	if activation is not None:
		if isinstance(activation, nn.Module):
			seq += [activation]
		else:
			raise TypeError('Activation {} is not a valid torch.nn.Module'.format(str(activation)))

	if dropout is not None:
		seq += [nn.Dropout(dropout)]

	with torch.no_grad():
		if weight_init == 'xavier':
			torch.nn.init.xavier_uniform_(seq[0].weight)
		if weight_init == 'identity':
			torch.nn.init.eye_(seq[0].weight)
		if weight_init not in ['xavier', 'identity', None]:
			if isinstance(weight_init, int) or isinstance(weight_init, float):
				torch.nn.init.constant_(seq[0].weight, weight_init)

	return seq



class NeighborNormLayer(nn.Module):
	''' Normalization layer that divides the output of a preceding layer by the number of neighbor features.
	'''
	def __init__(self):
		super(NeighborNormLayer, self).__init__()

	def forward(self, input_features, n_neighbors):
		''' Computes normalized output

		Parameters:
		----------------
		input_features: torch.tensor
			input tensor of features [n_frames, n_atoms, n_feats]
		n_neighbors: int, number of neighbors

		Returns:
		----------------
		normalized_features: torch.tensor, normalized input features
		'''
		return input_features / n_neighbors


def gather_nodes(nodes, neighbor_idx):
	'''
	given node-features [batch-size, num-atoms, num-features] and neighbor_dix [batch-size, num-atoms, num-neighbors] 
	this find the neighbors of each node and concatentates their features together:

	Returns:
	[batch-szie, num-atoms, num-neighbors, num-features]
	'''
	neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1)) 
	# [batch-size, num_nodes*num_neighbors]
	neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1,-1,nodes.size(2)) 
	# [batch-size, num_nodes*num_neighbors, num_features]

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

class NeighborMultiHeadAttention(nn.Module):
	''' MultiHeadAttention class for neighbors of every node in the graph representation
	
	multihead attention is defined on the edges between node in the neighborhood of every node

	quries: current node embedding 
	keys: relational information r(i,j)=(h_j, e_ij) j from N(i,k)
	values: relational information same as keys

	
	parameters:
	------------------
	num_hidden: int, number of features of atoms
	num_int: int, number of gaussians for edge
	num_heads: int, number of heads for multi-head attention

	Returns:
	------------------
	updated node embedding after multihead attention of neighbors
	'''
	def __init__(self, num_hidden, num_in, num_heads=4):
		super(NeighborMultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.num_hidden = num_hidden

		# self-attention layers: {queries, keys, values, output}
		self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
		self.W_K = nn.Linear(num_in, num_hidden, bias=False) # make the edges the same size as node embedding size
		self.W_V = nn.Linear(num_in, num_hidden, bias=False) # make the edges the same size as node embedding size
		self.W_O = nn.Linear(num_hidden, num_hidden, bias=False) # linear layer for the output
		return

	def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
		''' numercially stable masked softmax '''
		# attend_logits: [batch-size, num_nodes, num_heads, num_neighbors]
		# mask_attend: [batch-size, num_nodes, num_heads, num_neighbors]
		negative_inf = np.finfo(np.float32).min
		attend_logits = torch.where(mask_attend>-1, attend_logits, torch.tensor(negative_inf).to(device))
		attend = F.softmax(attend_logits, dim)
		attend = mask_attend * attend
		return attend

	def forward(self, h_V, h_E, mask_attend=None):
		''' self attention graph structure O(NK)

		parameters:
			h_V: node features [batch_size, num_nodes, n_features]
			h_E: neighbor featues [batch_size, num_nodes, num_neighbors, n_gaussians]
			mask_attend: mask for attention [batch_size, num_nodes, num_neighbors] 
		Returns:
			h_V_t: node update after neighborattention [batch-size, num_nodes, n_features]
		'''
		# dimensions
		n_batch, n_nodes, n_neighbors = h_E.shape[:3]
		n_heads = self.num_heads
		d = int(self.num_hidden/n_heads)

		Q = self.W_Q(h_V.to(torch.float32)).view([n_batch, n_nodes, 1, n_heads, 1, d]) # [B,N,1,n_heads,1,d]
		# [batch-size, num-nodes, 1, num_heads, 1, d]
		K = self.W_K(h_E.to(torch.float32)).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
		# [batch-size, num-nodes, num_neighbors, num_heads, d, 1]
		V = self.W_V(h_E.to(torch.float32)).view([n_batch, n_nodes, n_neighbors, n_heads, d])
		# [batch-size, num-nodes, num_neighbors, num_heads, d]

		# attention with scaled inner product between keys and queries
		attend_logits = torch.matmul(Q,K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
		# [batch-size, num-nodes, n_heads, num_nbrs]
		attend_logits = attend_logits / np.sqrt(d) # normalize
		if mask_attend is not None:
			# masked softmax
			mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1).to(device)
			# [batch-size, num_nodes, num_heads, num_neighbors]
			attend = self._masked_softmax(attend_logits, mask)
		else:
			attend = F.softmax(attend_logits, -1).to(device)
		# [batch-size, num-nodes, num-heads, num-neighbors]

		h_V_update = torch.matmul(attend.unsqueeze(-2).to(torch.float32), V.transpose(2,3))
		# multiply attention scores with the values
		h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
		h_V_update = self.W_O(h_V_update)
		return h_V_update

class GraphConvLayer(nn.Module):
	r''' Graph convolutional layer as introduced by Tian et al in 2019 for materrials and reformultated for protines.
	

	formulation:
		v_{i}^{k+1} = v_{i}{k} + \sim_{j} w_{i,j}^k \circ g(z_{i,j}^k W_{c}^k + b_{c}^k)
	where:
		w_{i,j}^k = sigmoid(z_{i,j}^k W_{g}^k + b_{g}^k) and z_{i,j}^k = concat([v_{i}^k, v_{j}^k, u_{i,j}^k])

	v_{i}^k is the node embedding of atom i at layer k, z_{i,j}^k is the concatentation of neighboring atoms and their bond features.
	g() denote a non-linear activation function, w_{i,j}^k is an edge-gating mechanism to incorporate different interaction strength
	among neighbors of a node.

	
	Parameters:
	---------------------
	atom_emb_dim : int,  atom embedding dimension 
	bond_emb_dim:  int, bond embedding dimension (number of gaussians)


	References:
	----------------------
		{Sanyal2020.04.06.028266,
		author = {Sanyal, Soumya and Anishchenko, Ivan and Dagar, Anirudh and Baker, David and Talukdar, Partha},
		title = {ProteinGCN: Protein model quality assessment using Graph Convolutional Networks},
		year = {2020},
		doi = {10.1101/2020.04.06.028266},
		publisher = {Cold Spring Harbor Laboratory},
		URL = {https://www.biorxiv.org/content/early/2020/04/07/2020.04.06.028266},
		journal = {bioRxiv}
	'''
	def __init__(self, atom_emb_dim, bond_emb_dim):
		super(GraphConvLayer, self).__init__()
		self.h_a = atom_emb_dim
		self.h_b = bond_emb_dim
		self.fc_full = nn.Linear(2*self.h_a+self.h_b, 2*self.h_a)
		self.sigmoid = nn.Sigmoid()
		self.activation_hidden =nn.ReLU()
		self.bn_hidden = nn.BatchNorm1d(2*self.h_a)
		self.bn_output = nn.BatchNorm1d(self.h_a)
		self.activation_output = nn.ReLU()

	def forward(self, atom_emb, nbr_emb, nbr_adj_list):
		''' Compute the GraphConvLayer

		Parameters:
		------------------
		atom_emb: torch.tensor, atom embeddings [batch-size, num_atoms, num_features]
		nbr_emb: torch.tensor, bond embedding of atom neighbors [batch-size, num_atom, num_neighbor, n_gaussians]
		nbr_adj_list: torch.tensor, indices for neighbors of a node [batch-size, num_atoms, num_neighbors]
		
		Returns:
		-------------------
		out: atom embeddings after applying GraphConvLayer [batch-size, num_atom, num_features]
		'''
		N, M = nbr_adj_list.shape[1:] 
		# N is number of atoms and M is number of neighbors
		B = atom_emb.shape[0] # batch-size

		# gather the feature of neighbors into size [batch-size, num_atoms, num_neighbors, num_features]
		atom_nbr_emb = atom_emb[torch.arange(B).unsqueeze(-1), nbr_adj_list.to(torch.long).view(B,-1)].view(B,N,M,self.h_a).to(device)
		# concatenate the embedding of neighboring atoms with the bond embedding connecting the two 
		# shape [batch-size, num-atoms, num_neighbors, 2*num_features + num_gaussians]
		total_nbr_emb = torch.cat([atom_emb.unsqueeze(2).expand(B,N,M,self.h_a), atom_nbr_emb, nbr_emb],dim=-1).to(torch.float32)

		# apply a linear layer
		total_gated_emb = self.fc_full(total_nbr_emb)
		total_gated_emb = self.bn_hidden(total_gated_emb.view(-1,self.h_a*2)).view(B,N,M,self.h_a*2)
		nbr_filter, nbr_core = total_gated_emb.chunk(2, dim=3)
		# Sigmoid function to for edge-gating mechanism
		nbr_filter = self.sigmoid(nbr_filter)
		nbr_core = self.activation_hidden(nbr_core)
		# element-wise multiplication and aggregation of neighbors
		nbr_sumed = torch.sum(nbr_filter*nbr_core, dim=2)
		# apply batch-norm to output
		nbr_sumed = self.bn_output(nbr_sumed.view(-1, self.h_a)).view(B,N,self.h_a)
		# apply non-linear activation to output
		out = self.activation_output(atom_emb+nbr_sumed)
		return out

class GaussianDistance(object):
	def __init__(self, dmin, dmax, step, var=None):

		''' Expands ditsnces by gaussian basis functions
		parameters:
		-------------------
		dmin: float, minimum distance between atoms to be considered for gaussian basis
		dmax: float, maximum distance between atoms to be considered for gaussian basis
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


class ContinuousFilterConv(nn.Module):
	'''
	Continuous filter convolution layer for SchNet as described by Schütt et al.(2018)
	A continuous-filter convolutional layer uses continuous radial basis functions for discrete data. (Schütt et al. (2018))
	Continuous-filter convolution block consists of a filter generating network as follows:

	Filter generator:
		1. get distances betwee nodes.
		2. atom-wise/Linear layer with non-linear activation
		3. atom-wise/Linear layer with non-linear activation

	The filter generator output is then multiplied element-wise with the continuous convolution filter as part of the interaction block.

	Parameters:
	-----------------
	n_gaussians: int
		number of gaussian used in hte radial basis function. needed to determine input feature size of first dense layer
	n_filters: int
		number of filters that will be created. Also determines the output size. Needs to be the same size as the features of residual 
		connection in the interaction block.
	activation: nn.Module
		Activation function for the filter generating network.

	Notes:
	------------------
	Following current implementation in SchNetPack, the last linear layer of the filter generator does not contain an activation
	function. This allows the filter generator to contain negative values.

	References:

	'''
	def __init__(self, n_gaussians, n_filters, activation=nn.Tanh(), normalization_layer=None):

		super(ContinuousFilterConv, self).__init__()
		filter_layers =  LinearLayer(n_gaussians, n_filters, bias=True, activation=activation)
		filter_layers += LinearLayer(n_filters, n_filters, bias=True) # no activation here
		self.filter_generator = nn.Sequential(*filter_layers)

		if normalization_layer is not None:
			self.normalization_layer = normalization_layer
		else:
			self.normalization_layer = None

	def forward(self, features, rbf_expansion, neighbor_list):
		''' Compute convolutional block
		Parameters:
		-------------
		features: torch.Tensor
			Feature vector of size [n_frames, n_atoms, n_features]
		rbf_expansion: torch.Tensor
			Gaussian expansion of bead distances of size, [n_frames, n_atoms, n_neighbors, n_gaussians]
		neighbor_list: torch.Tensor
			indices of all neighbors of each bead size [n_frames, n_atoms, n_neighbors]

		Returns:
		-------------
		aggregated features: torch.Tensor
			Residual features of shape [n_frames, n_atoms, n_features]
		'''

		# generate convolutional filter of size [n_frames, n_atoms, n_neighbors, n_features]

		conv_filter = self.filter_generator(rbf_expansion.to(torch.float32))

		# Feature tensor needs to also be transformed from [n_frames, n_atoms, n_features]
		# to [n_frames, n_atoms, n_neighbors, n_features]
		n_batch, n_atoms, n_neighbors = neighbor_list.size()

		# size [n_frames, n_atoms*n_neighbors, 1]
		neighbor_list = neighbor_list.reshape(-1, n_atoms*n_neighbors, 1)

		# size [n_frames, natoms*n_neighbors, n_features]
		neighbor_list = neighbor_list.expand(-1, -1, features.size(2))

		# Gather the features into the respective places in the neighbor list
		neighbor_features = torch.gather(features, 1, neighbor_list.to(torch.int64))
		# Reshape back to [n_frames, n_atoms, n_neighbors, n_features] for element-wise multiplication

		neighbor_features = neighbor_features.reshape(n_batch, n_atoms, n_neighbors, -1)

		# element-wise multiplication of the features with the convolutional filter
		conv_features = neighbor_features * conv_filter

		# aggregate/pool the features from [n_frames, n_atoms, n_neighbors, n_features] to [n_frames, n_atoms, n_features]
		aggregated_features = torch.sum(conv_features, dim=2)


		if self.normalization_layer is not None:
			if isinstance(self.normalization_layer, NeighborNormLayer):
				return self.normalization_layer(aggregated_features, n_neighbors)
			else:
				return self.normalization_layer(aggregated_features)
		else:
			return aggregated_features


class InteractionBlock(nn.Module):
	'''
	SchNet interaction block as described by Schütt et al. (2018).

	An interaction block consists of :
		1. Atom-wise/Linear layer without activation function
		2. Continuous filter convolution, which is a filter-generator multiplied element-wise with the output of previous layer
		3. Atom-wise/Linear layer with the activation
		4. Atom-wise/Linear layer without activation

	The output of an interaction block will then be used to form an additive residual connection with the original input features, 
	[x'1, ..., x'n]

	Parameters:
	---------------
	n_inputs: int, number of input features, determines input size for the initial linear layer
	n_gaussians: int, number of gaussians that has been used in the radial basis function. needed to determine the input size of the continuous
			filter convolution.
	n_filters: int, number of filters that will be created in the continuous filter convolution. The same feature size will be used for the output
			linear layers of the interaction block.
	activation: nn.Module activation function for the atom-wise layers. 
	normalization_layer: nn.Module (default=None)
			normalization layer to be applied to the output of the ContinuousFilterConvolution

	The residul connection will be added later in model module between the interaction blocks

	References:
	----------------
	    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779
	'''

	def __init__(self, n_inputs, n_gaussians, n_filters, activation=nn.Tanh(), normalization_layer=None):
		super(InteractionBlock, self).__init__()

		self.initial_dense = nn.Sequential(*LinearLayer(n_inputs, n_filters, bias=False, activation=None))

		self.cfconv = ContinuousFilterConv(n_gaussians=n_gaussians,
										   n_filters=n_filters,
										   activation=activation,
										   normalization_layer=normalization_layer)

		output_layers = LinearLayer(n_filters, n_filters, bias=True, activation=activation)
		output_layers += LinearLayer(n_filters, n_filters, bias=True)
		self.output_dense = nn.Sequential(*output_layers)

	def forward(self, features, rbf_expansion, neighbor_list):
		''' Compute interaction block

		Parameters:
		-----------------
		features: torch.Tensor
			Input features from an embedding or ineteraction layer. [n_frames, n_atom, n_features]
		rbf_expansion: torch.Tensor,
			Radial basis function expansion of distances [n_frames, n_atoms, n_neighbors, n_gaussians]
		neighbor_list: torch.Tensor
			Indices of all neighbors of each atom [n_frames, n_atoms, n_neighbors]

		Returns:
		-----------------
		output_features: torch.Tensor
			Output of an interaction block. This output can be used to form a residual connection with the output of 
			a prior embedding/interaction layer. [n_frames, n_atoms, n_filters]
		'''
		init_feature_output = self.initial_dense(features)
		conv_output = self.cfconv(init_feature_output.to(torch.float32), rbf_expansion.to(torch.float32), neighbor_list).to(torch.float32)
		output_features = self.output_dense(conv_output).to(torch.float32)
		return output_features