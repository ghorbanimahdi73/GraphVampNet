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
from model import GraphVampNet
from deeptime.util.data import TrajectoryDataset
from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP
from copy import deepcopy
import os



if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')



args = buildParser().parse_args()

#---------------load Ala-dipeptide traj ---------------
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
#--------------------------------------------------------


dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)


n_val = int(len(dataset)*args.val_frac)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])



loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

lobe = GraphVampNet()
lobe_timelagged = deepcopy(lobe).to(device=device)
lobe = lobe.to(device)



vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device)



def train(train_loader , n_epochs, validation_loader=None):
	'''
	Parameters:
	-----------------
	train_loader: torch.utils.data.DataLoader
		The data to use for training, should yield a tuple of batches representing instantaneous and time-lagged samples
	n_epochs: the number of epochs for training
	validation_loader: torch.utils.data.DataLoader:
		The validation data should also be yielded as a two-element tuple.

	Returns:
	-----------------
	model: VAMPNet
	'''
	for epoch in tqdm(range(n_epochs)):
		for batch_0, batch_t in train_loader:
			vampnet.partial_fit((batch_0.to(device),batch_t.to(device)))

		if validation_loader is not None:
			with torch.no_grad():
				scores = []
				for val_batch in validation_loader:
					scores.append(vampnet.validate((val_batch[0].to(device), val_batch[1].to(device))))

				mean_score = torch.mean(torch.stack(scores))
				vampnet._validation_scores.append((vampnet._step, mean_score.item()))

		if args.save_checkpoints:
			torch.save({
				'epoch' : epoch,
				'state_dict': lobe.state_dict(),
				'train_scores': vampnet.train_scores.T,
				'validation_scores': vampnet.validation_scores.T,
				}, 'logs/'+'logs_'+str(epoch)+'.pt')

	return vampnet.fetch_model()



if not args.train and os.path.isfile(args.trained_model):
	print('Loading model')
	checkpoint = torch.load(args.trained_model)
	lobe.load_state_dict(checkpoint['state_dict'])
	lobe_timelagged = deepcopy(lobe).to(device=device)
	lobe = lobe.to(device)
	vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device)
	model = vampnet.fetch_model()
	vampnet.train_scores = checkpoint['train_scores']
	vampnet.validation_scores - che



elif args.train:
	model = train(train_loader=loader_train, n_epochs=args.epochs, validation_loader=loader_val)





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