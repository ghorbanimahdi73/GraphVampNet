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

np.savez('dists.npz', dists)
np.savez('inds.npz', inds)