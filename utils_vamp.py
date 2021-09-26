import torch
import numpy as np
import deeptime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP
import os
import pickle


plt.set_cmap('jet')

def estimate_koopman_op(trajs, tau):
	if type(trajs) == list:
		traj = np.concatenate([t[:-tau] for t in trajs], axis=0)
		traj_lag = np.concatenate([t[tau:] for t in trajs], axis=0)
	else:
		traj = trajs[:-tau]
		traj_lag = trajs[tau:]
	c_0 = np.transpose(traj)@traj
	c_tau = np.transpose(traj)@traj_lag

	eigv, eigvec = np.linalg.eig(c_0)
	include = eigv > 1e-7
	eigv = eigv[include]
	eigvec = eigvec[:, include]
	c0_inv = eigvec @ np.diag(1/eigv)@np.transpose(eigvec)

	koopman_op = c0_inv @ c_tau
	return koopman_op

def get_ck_test(traj, steps, tau):
	if type(traj) == list:
		n_states = traj[0].shape[1]
	else:
		n_states = traj.shape[1]

	predicted = np.zeros((n_states, n_states, steps))
	estimated = np.zeros((n_states, n_states, steps))

	predicted[:, :, 0] = np.identity(n_states)
	estimated[:, :, 0] = np.identity(n_states)

	for vector, i in zip(np.identity(n_states), range(n_states)):
		for n in range(1, steps):
			koop = estimate_koopman_op(traj, tau)
			koop_pred = np.linalg.matrix_power(koop, n)
			koop_est = estimate_koopman_op(traj, tau*n)

			predicted[i,:,n] = vector@koop_pred
			estimated[i,:,n] = vector@koop_est

	return [predicted, estimated]

def plot_ck_test(pred, est, n_states, steps, tau, save_folder):
	fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True)
	for index_i in range(n_states):
		for index_j in range(n_states):

			ax[index_i][index_j].plot(range(0, steps*tau, tau), pred[index_i, index_j], color='b')
			ax[index_i][index_j].plot(range(0, steps*tau, tau), est[index_i, index_j], color='r', linestyle='--')
			ax[index_i][index_j].set_title(str(index_i+1)+'->'+str(index_j+1), fontsize='small')

	ax[0][0].set_ylim((-0.1, 1.1))
	ax[0][0].set_xlim((0, steps*tau))
	ax[0][0].axes.get_xaxis().set_ticks(np.round(np.linspace(0, steps*tau, 3)))
	plt.tight_layout()
	plt.show()
	plt.savefig(save_folder+'/ck_test.png')

def get_its(traj, lags):
	'''
	implied timescales from a trajectory estiamted at a series of lag times

	parameters:
	---------------------
	traj: numpy array [traj_timesteps, traj_dimension] traj or list of trajs
	lags: numpy array with size [lagtimes] series of lag times at which the implied timescales are estimated

	Returns:
	---------------------
	its: numpy array with size [traj_dimensions-1, lag_times] implied timescales estimated for the trajectory
	'''

	if type(traj) == list:
		outputsize = traj[0].shape[1]
	else:
		outputsize = traj.shape[1]
	its = np.zeros((outputsize-1, len(lags)))

	for t, tau_lag in enumerate(lags):
		koopman_op = estimate_koopman_op(traj, tau_lag)
		k_eigvals, k_eigvec = np.linalg.eig(np.real(koopman_op))
		k_eigvals = np.sort(np.absolute(k_eigvals))
		k_eigvals = k_eigvals[:-1]
		its[:, t] = (-tau_lag/np.log(k_eigvals))

	return its

def plot_its(its, lag, save_folder,ylog=False):
	'''
	plots the implied timescales calculated by the function

	get_its:
	parameters:
	------------------------
	its: numpy array
		the its array returned by the function get_its
	lag: numpy array 
		lag times array used to estimated the implied timescales
	ylog: Boolean, optional, default=False
		if true, the plot will be a logarithmic plot, otherwize it 
		will be a semilogy plot
	'''

	if ylog:
		plt.loglog(lag, its.T[:,::-1])
		plt.loglog(lag, lag, 'k')
		plt.fill_between(lag, lag, 0.99, alpha=0.2, color='k')
	else:
		plt.semilogy(lag, its.T[:,::-1])
		plt.semilogy(lag,lag, 'k')
		plt.fill_between(lag, lag, 0.99, alpha=0.2, color='k')
	plt.show()
	plt.savefig(save_folder+'/its.png')


def plot_scores(train_scores, validation_scores, save_folder):
	plt.loglog(train_scores, label='training')
	plt.loglog(validation_scores, label='validation')
	plt.xlabel('step')
	plt.ylabel('score')
	plt.legend()
	plt.savefig(save_folder+'/scores.png')



def chapman_kolmogorov_validator(model, mlags, n_observables=None,
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

	whole_dataset = TrajectoryDataset.from_trajectories(lagtime=lagtime, data=data)
	whole_dataloder = DataLoader(whole_dataset, batch_size=10000, shuffle=False)
	for batch_0, batch_t in whole_dataloder:
		est.partial_fit((batch_0.numpy(), batch_t.numpy()))

	return est.fetch_model()


class VAMPKoopmanCKValidator(KoopmanChapmanKolmogorovValidator):

	def fit(self, data, n_jobs=None, progress=None, **kw):
		return super().fit(data, n_jobs, progress, _vamp_estimate_model_for_lag, **kw)