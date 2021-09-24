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
import mdtraj as md
import argparse
import sys

from deeptime.decomposition._koopman import KoopmanChapmanKolmogorovValidator


parser = argparse.ArgumentParser()
parser.add_argument('--num-neighbors', type=int, default=7, help='number of neighbors')
parser.add_argument('--traj-folder', type=str, default=None, help='the path to the trajecotyr folder')
parser.add_argument('--stride', type=int, default=5, help='stride for trajectory')


args = parser.parse_args()
########## for loading the BBA trajectory ####################################


traj_1 = ['data/DESRES-Trajectory_1FME-0-c-alpha/1FME-0-c-alpha/1FME-0-c-alpha-'+str(i).zfill(3)+'.dcd' for i in range(12)]
traj_2 = ['data/DESRES-Trajectory_1FME-1-c-alpha/1FME-1-c-alpha/1FME-1-c-alpha-'+str(i).zfill(3)+'.dcd' for i in range(6)]
crys = md.load_pdb('1fme_ca.pdb')
top = crys.topology

t1 = md.load_dcd(traj_1[0],top=top, stride=args.stride)
coor_t1 = t1.xyz
for i in range(1,len(traj_1)):
	t1 = md.load_dcd(traj_1[i], top=top, stride=args.stride)
	t1.superpose(crys)
	coor_t1 = np.concatenate((coor_t1, t1.xyz), axis=0)

t2= md.load_dcd(traj_2[0], top=top, stride=args.stride)
coor_t2 = t2.xyz
for i in range(1,len(traj_2)):
	t2 = md.load_dcd(traj_2[i], top=top, stride=args.stride)
	t2.superpose(crys)
	coor_t2 = np.concatenate((coor_t2, t2.xyz), axis=0)

print(coor_t1.shape)
print(coor_t2.shape)

data = list([coor_t1, coor_t2])
np.savez('pos_BBA.npz', data[0],data[1])

quit()

if torch.cuda.is_available():
	device = torch.device('cpu')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

#ala_coords_file = mdshare.fetch(
#    "alanine-dipeptide-3x250ns-heavy-atom-positions.npz", working_directory="data"
#)
#with np.load(ala_coords_file) as fh:
#    data = [fh[f"arr_{i}"].astype(np.float32) for i in range(3)]
#
#dihedral_file = mdshare.fetch(
#    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz", working_directory="data"
#)
#
#with np.load(dihedral_file) as fh:
#    dihedral = [fh[f"arr_{i}"] for i in range(3)]



# reshape the data to be in share list of [N,num_atoms,3]
#data_reshaped = []
#for i in range(len(data)):
#    temp = data[i].reshape(data[0].shape[0], 3, 10).swapaxes(1,2)
#    data_reshaped.append(temp)
#-----------------------------------------------------

def get_nbrs(all_coords, num_neighbors=args.num_neighbors):
	'''
	inputs: a trajectory or list of trajectories with shape [T, num_atoms, dim]
		T: number of steps
		dim: number of dimensions (3 coordinates) 

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
			for j in tqdm(range(len(tmp_coords))):
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

dists, inds = get_nbrs(data, args.num_neighbors)
np.savez('dists_BBA_'+str(args.num_neighbors)+'.npz', dists[0], dists[1])
np.savez('inds_BBA_'+str(args.num_neighbors)+'.npz', inds[0], inds[1])




def chunks(data, chunk_size=5000):
	'''
	splitting the trajectory into chunks for passing into analysis part
	data: list of trajectories
	chunk_size: the size of each chunk
	'''
	if type(data) == list:
			
		for data_tmp in data:
			for j in range(0, len(data_tmp),chunk_size):
				print(data_tmp[j:j+chunk_size,...].shape)
				yield data_tmp[j:j+chunk_size,...]

	else:

		for j in range(0, len(data), chunk_size):
			yield data[j:j+chunk_size,...]


def chapman_kolmogrov_validator(model, mlags, test_model: CovarianceKoopmanModel=None, n_observables=None,
								observables='phi', statistics='psi'):
	''' returns a chapman-kolmogrov validator based on this estimator and a test model

	parameters:
	-----------------
	model: VAMP model
	mlags: int or int-array
		multiple of lagtimes of the test_model to test against
	test_model: CovarianceKoopmanModel, optional, default=None,
		The model that is tested, if not provided uses this estimator's encapsulated model.
	n_observables: int, optional, default=None,
		limit the number of default observables to this number. only used if 'observables' are Nonr or 'statistics' are None.
	observables: (input_dimension, n_observables) ndarray
		coefficents that express one or multiple observables in the basis of the input features
	statistics: (input_dim, n_statistics) ndarray
		coefficents that express one or more statistics in the basis of the input features

	Returns:
	------------------
	validator: KoopmanChapmanKolmogrovValidator
		the validator
	''' 
	test_model = model.fetch_model()
	assert test_model is not None, 'We need a test model via argument or an estimator which was already fit to data'

	lagtime = model.lagtime
	if n_observables is not None:
		if n_observables > test_model.dim:
			import warnings
			warnings.warn('selected singular functgions as observables but dimension is lower thanthe requested number of observables')
			n_observables = test_model.dim

	else:
		n_observables = test_model.dim

	if isinstance(observables, str) and observables == 'phi':
		observables = test_model.singular_vectors_right[:, :n_observables]
		observables_mean_free = True
	else:
		observables_mean_free = False

	if isinstance(statistics, str) and statistics == 'psi':
		statistics = test_model.singular_vectors_left[:, :n_observables]
		statistics_mean_free = True
	else:
		statistics_mean_free = False

	return VAMPKoopmanCKValidator(test_model, model, lagtime, mlags, observables, statistics,
										observables_mean_free, statistics_mean_free)

def _vamp_estimate_model_for_lag(estimator: VAMP, model, data, lagtime):
	est = VAMP(lagtime=lagtime, dim=estimator.dim, var_cutoff=estimator.var_cutoff, scaling=estimator.scaling,
		epsilon=estimator.epsilon, observable_transform=estimator.observable_transform)

	whole_dataset = TrajectoryDataset.from_trajectories(lagtime=lag, data=data)
	whole_dataloder = DataLoader(whole_dataset, batch_size=10000, shuffle=False)
	for batch_0, batch_t in whole_dataloder:
		est.partial_fit((batch_0.numpy(), batch_t.numpy()))

	return est.fetch_model()

class VAMPKoopmanCKValidator(KoopmanChapmanKolmogorovValidator):

	def fit(self, data, n_jobs=None, progress=None, **kw):
		return super().fit(data, n_jobs, progress, _vamp_estimate_model_for_lag, **kw)



