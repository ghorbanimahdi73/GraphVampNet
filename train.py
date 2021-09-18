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
import pickle
import warnings


if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

# ignore deprecation warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)


args = buildParser().parse_args()
print(args)

if not os.path.exists(args.save_folder):
	print('making the folder for saving checkpoints')
	os.makedirs(args.save_folder)


#sys.stdout = Logger(args.save_folder)
#print(vars(args))


meta_file = os.path.join(args.save_folder, 'metadata.pkl')
pickle.dump({'args': args}, open(meta_file, 'wb'))



#---------------load Ala-dipeptide traj ---------------
#ala_coords_file = mdshare.fetch(
#    "alanine-dipeptide-3x250ns-heavy-atom-positions.npz", working_directory="data"
#)
#with np.load(ala_coords_file) as fh:
#    data = [fh[f"arr_{i}"].astype(np.float32) for i in range(3)]
#
#dihedral_file = mdshare.fetch(
#    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz", working_directory="data"
#)

#with np.load(dihedral_file) as fh:
#    dihedral = [fh[f"arr_{i}"] for i in range(3)]
#


#------------------- data as a list of trajectories ---------------------------
dists1, inds1 = np.load('dists_BBA_7.npz')['arr_0'], np.load('inds_BBA_7.npz')['arr_0']
dists2, inds2 = np.load('dists_BBA_7.npz')['arr_1'], np.load('inds_BBA_7.npz')['arr_1']


mydists1 = torch.from_numpy(dists1).to(device)
myinds1 = torch.from_numpy(inds1).to(device)

mydists2 = torch.from_numpy(dists2).to(device)
myinds2 = torch.from_numpy(inds2).to(device)

data = []
data.append(torch.cat((mydists1,myinds1), axis=-1))
data.append(torch.cat((mydists2,myinds2), axis=-1))

dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)


n_val = int(len(dataset)*args.val_frac)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])

loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

# data is a list of trajectories [T,N,M+M]
#---------------------------------------------------------------------------------

lobe = GraphVampNet()
lobe_timelagged = deepcopy(lobe).to(device=device)
lobe = lobe.to(device)

vampnet = VAMPNet(lobe=lobe, lobe_timelagged=lobe_timelagged, learning_rate=args.lr, device=device, optimizer='Adam', score_method='VAMP2')

def count_parameters(model):
	'''
	count the number of parameters in the model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('number of parameters', count_parameters(lobe))

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
		'''
		perform batches of data here
		'''
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
				#'train_scores': vampnet.train_scores.T,
				#'validation_scores': vampnet.validation_scores.T,
				}, args.save_folder+'/logs_'+str(epoch)+'.pt')

	return vampnet.fetch_model()



plt.set_cmap('jet')

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
	with open(args.save_folder+'/train_scores.npy','wb') as f:
		np.save(f, vampnet.train_scores)

	with open(args.save_folder+'/validation_scores.npy','wb') as f:
		np.save(f, vampnet.validation_scores)


	# plotting the training and validation scores of the model
	plt.loglog(*vampnet.train_scores.T, label='training')
	plt.loglog(*vampnet.validation_scores.T, label='validation')
	plt.xlabel('step')
	plt.ylabel('score')
	plt.legend()
	plt.savefig(args.save_folder+'/scores.png')





#state_probabilities = model.transform(data[0])
#
#f, axes = plt.subplots(3,2, figsize=(12,16))
#for i, ax in enumerate(axes.flatten()):
#	ax.scatter(*dihedral[0][::5].T, c=state_probabilities[...,i][::5])
#	ax.set_title(f'state {i+1}')
#f.savefig('state_prob.png')


#fig, ax = plt.subplots(1,1, figsize=(8,10))
#assignments = state_probabilities.argmax(1)
#plt.scatter(*dihedral[0].T, c=assignments, s=5, alpha=0.1)
#plt.title('Transformed state assignments')
#plt.savefig('assignments.png')








data_np = []
for i in range(len(data)):
	data_np.append(data[i].cpu().numpy())


# for the analysis part create an iterator for the whole dataset to feed in batches
whole_dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data_np)
whole_dataloder = DataLoader(whole_dataset, batch_size=args.batch_size, shuffle=False)



# for plotting the implied timescales
lagtimes = np.arange(1,101,2, dtype=np.int32)
timescales = []
for lag in tqdm(lagtimes):
	vamp = VAMP(lagatime=lagtime, observable_transform=model)
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
f.savefig(args.save_folder+'/ITS.png')





# for plotting the CK test
validator = vamp.chapman_kolmogorov_validator(mlags=20)

cktest = validator.fit(data, n_jobs=1, progress=tqdm).fetch_model()
n_states = len(vamp.singular_values)

tau = cktest.lagtimes[1]
steps = len(cktest.lagtimes)
fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True, constrained_layout=True)
for i in range(n_states):
    for j in range(n_states):
        pred = ax[i][j].plot(cktest.lagtimes, cktest.predictions[:, i, j], color='b')
        est = ax[i][j].plot(cktest.lagtimes, cktest.estimates[:, i, j], color='r', linestyle='--')
        ax[i][j].set_title(str(i+1)+ '->' +str(j+1),
                                       fontsize='small')
ax[0][0].set_ylim((-0.1,1.1));
ax[0][0].set_xlim((0, steps*tau));
ax[0][0].axes.get_xaxis().set_ticks(np.round(np.linspace(0, steps*tau, 3)));
fig.legend([pred[0], est[0]], ["Predictions", "Estimates"], 'lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.1));

fig.save(args.save_folder+'/cktest.png')