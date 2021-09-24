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
	plt.savefig(save_folder+'/its_2.png')


def plot_scores(train_scores, validation_scores, save_folder):
	plt.loglog(train_scores, label='training')
	plt.loglog(validation_scores, label='validation')
	plt.xlabel('step')
	plt.ylabel('score')
	plt.legend()
	plt.savefig(save_folder+'/scores.png')