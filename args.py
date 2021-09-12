import argparse

def buildParser():

	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
	parser.add_argument('--batch-size', type=int, default=10000, help='batch-size for training')
	parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
	parser.add_argument('--hidden', type=int, default=64, help='number of hidden neurons')
	parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms')
	parser.add_argument('--num-classes', type=int, default=6, help='number of coarse-grained classes')
	parser.add_argument('--save-folder', type=str, default='logs', help='Where to save the trained model')
	parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
	parser.add_argument('--atom_init', type=str, default=None, help='inital embedding for atoms file')
	parser.add_argument('--h_a', type=int, default=64, help='Atom hidden embedding dimension')
	parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors for each atom in the graph')
	parser.add_argument('--n_conv', type=int, default=4, help='Number of convolution layers')
	parser.add_argument('--save_checkpoint', default=False,  action='store_true', help='If True, stores checkpoints')
	parser.add_argument('--conv_type', default='ConvLayer', type=str, help='the type of convolution layer, one of \
				        [ConvLayer, NeighborAttention]')
	parser.add_argument('--dmin', default=0., type=float, help='Minimum distance for the gaussian filter')
	parser.add_argument('--dmax', default=3., type=float, help='maximum distance for the gaussian filter')
	parser.add_argument('--step', default=0.2, type=float, help='step for the gaussian filter')
	parser.add_argument('--tau', default=1, type=int, help='lag time for the model')

	return parser