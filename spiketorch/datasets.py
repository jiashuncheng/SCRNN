import os, sys
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from torchvision import datasets, transforms
from struct import unpack

data_path = os.path.abspath(os.path.join('data'))

if not os.path.isdir(data_path):
	os.makedirs(data_path)

class MNIST(Dataset):
	def __init__(self, data, target):
		self.data = data
		self.target = target

	def __getitem__(self, index):
		image = self.data[index]
		label = self.target[index]
		return image, label

	def __len__(self):
		return len(self.data)
	
def get_MNIST(args, data_path=data_path, device=None):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as tensor.
	'''

	train_images, train_labels = load_MNIST(train=True, data_path=data_path)
	test_images, test_labels = load_MNIST(train=False, data_path=data_path)
	train_dataset = MNIST(train_images, train_labels)
	test_dataset = MNIST(test_images, test_labels)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
												batch_size=args.batch_size, 
												shuffle=True, 
												num_workers=0,
												generator=torch.Generator(device=device))
	test_dataloader = torch.utils.data.DataLoader(test_dataset, 
											   batch_size=args.batch_size, 
											   shuffle=False, 
											   num_workers=0,
											   generator=torch.Generator(device=device))

	return train_dataloader, test_dataloader	

def load_MNIST(train=True, data_path=data_path):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	fname = 'train' if train else 'test'

	if os.path.isfile(os.path.join(data_path, '%s.p' % fname)):
		# Get pickled data from disk.
		with open(os.path.join(data_path, '%s.p' % fname), 'rb') as f:
			data = pickle.load(f)
		X = data['X']
		y = data['y']
	else:
		# Open the images with gzip in read binary mode.
		if train:
			images = open(os.path.join(data_path, 'train-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(data_path, 'train-labels-idx1-ubyte'), 'rb')
		else:
			images = open(os.path.join(data_path, 't10k-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(data_path, 't10k-labels-idx1-ubyte'), 'rb')

		# Get metadata for images.
		images.read(4)
		number_of_images = unpack('>I', images.read(4))[0]
		rows = unpack('>I', images.read(4))[0]
		cols = unpack('>I', images.read(4))[0]

		# Get metadata for labels.
		labels.read(4)
		N = unpack('>I', labels.read(4))[0]

		if number_of_images != N:
			raise Exception('number of labels did not match the number of images')

		# Get the data.
		print('...Loading MNIST data from disk.')
		print('\n')

		X = np.zeros((N, rows, cols), dtype=np.uint8)
		y = np.zeros((N, 1), dtype=np.uint8)

		for i in range(N):
			if i % 1000 == 0:
				print('Progress :', i, '/', N)
			X[i] = [[unpack('>B', images.read(1))[0] for unused_col in \
							range(cols)] for unused_row in range(rows) ]
			y[i] = unpack('>B', labels.read(1))[0]

		print('Progress :', N, '/', N, '\n')

		X = X.reshape([N, 784])
		data = {'X': X, 'y': y }

		with open(os.path.join(data_path, '%s.p' % fname), 'wb') as f:
			pickle.dump(data, f)

	return X, y

def get_MemoryMNIST(args, data_path=data_path, device=None):
	X0, y0 = load_MemoryMNIST(True, data_path=data_path, n=0)
	X1, y1 = load_MemoryMNIST(True, data_path=data_path, n=1)
	return [X0[0], y0[0]], [X1[0], y1[0]]

def load_MemoryMNIST(train=True, data_path=data_path, n=0):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	fname = 'train' if train else 'test'

	if os.path.isfile(os.path.join(data_path, '%s_memory_%d.p' % (fname, n))):
		# Get pickled data from disk.
		with open(os.path.join(data_path, '%s_memory_%d.p' % (fname, n)), 'rb') as f:
			data = pickle.load(f)
		X = data['X']
		y = data['y']
	else:
		# Open the images with gzip in read binary mode.
		if train:
			images = open(os.path.join(data_path, 'train-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(data_path, 'train-labels-idx1-ubyte'), 'rb')
		else:
			images = open(os.path.join(data_path, 't10k-images-idx3-ubyte'), 'rb')
			labels = open(os.path.join(data_path, 't10k-labels-idx1-ubyte'), 'rb')

		# Get metadata for images.
		images.read(4)
		number_of_images = unpack('>I', images.read(4))[0]
		rows = unpack('>I', images.read(4))[0]
		cols = unpack('>I', images.read(4))[0]

		# Get metadata for labels.
		labels.read(4)
		N = unpack('>I', labels.read(4))[0]

		if number_of_images != N:
			raise Exception('number of labels did not match the number of images')

		# Get the data.
		print('...Loading Memory MNIST data from disk.')
		print('\n')

		X = []
		y = []
		# X = np.zeros((N, rows, cols), dtype=np.uint8)
		# y = np.zeros((N, 1), dtype=np.uint8)

		for i in range(N):
			if i % 1000 == 0:
				print('Progress :', i, '/', N)
			image = [[unpack('>B', images.read(1))[0] for unused_col in \
							range(cols)] for unused_row in range(rows) ]
			label = unpack('>B', labels.read(1))[0]
			if label == n:
				X.append(image)
				y.append(label)

		print('Progress :', N, '/', N, '\n')

		X = np.array(X, dtype=np.uint8).reshape(len(X), 784)
		y = np.array(y, dtype=np.uint8).reshape(len(y), 1)

		data = {'X': X, 'y': y }

		with open(os.path.join(data_path, '%s_memory_%d.p' % (fname, n)), 'wb') as f:
			pickle.dump(data, f)

	return X, y

def generate_memory_spike_train(args, data):

	data[0][0][data[0][0]>0] = 1.
	data[1][0][data[1][0]>0] = 1.
	X0 = torch.tensor(data[0][0]).float()
	X1 = torch.tensor(data[1][0]).float()

	length = int(args.time / args.dt)
	spikes = X0.float().unsqueeze(0).unsqueeze(0).repeat(length, args.batch_size, 1).clone()
	targets = torch.zeros((args.batch_size, 1))

	for i in range(args.batch_size):
		num_0 = int(length * np.random.rand())
		num_1 = int(length - num_0)
		targets[i] = torch.tensor(num_1 > num_0).float()
		indices = np.random.choice(range(length), size=num_1, replace=False)
		spikes[indices, i, :] = X1.unsqueeze(0).repeat(num_1, 1)
	
	return spikes, targets

def generate_memory_spike_train_binary(args, data):
	length = int(args.time / args.dt)
	spikes = torch.zeros((length, args.batch_size, 1))
	targets = torch.zeros((args.batch_size, 1))

	for i in range(args.batch_size):
		num_0 = int(length * np.random.rand())
		num_1 = int(length - num_0)
		targets[i] = torch.tensor(num_1 > num_0).float()
		indices = np.random.choice(range(length), size=num_1, replace=False)
		spikes[indices, i, :] = 1.
	return spikes, targets

def generate_spike_train(image, time):
	'''
	Generates Poisson spike trains based on image ink intensity.
	'''
	# Get number of input neurons.
	batch_size = image.shape[0]
	n_input = image.shape[1]
	
	# Image data preprocessing (divide by 4, invert (for spike rates),
	# multiply by 1000 (conversion from milliseconds to seconds).
	# image = (1 / (image / 4)) * 1000
	# image = torch.where(image == float('inf'), torch.zeros_like(image), image)
	image = image/255.
	
	# Make the spike data.
	spike_times = torch.poisson(image.unsqueeze(0).repeat(time,1,1))
	spike_times = torch.cumsum(spike_times, axis=0)
	spike_times[spike_times >= time] = 0

	# Create spikes matrix from spike times.
	spikes = torch.zeros([time, batch_size, n_input]).to(image.device)
	index = torch.arange(n_input).unsqueeze(0).unsqueeze(0)
	index = index.repeat(time, batch_size, 1).type(torch.int64)
	batch_index = torch.arange(batch_size).unsqueeze(1).unsqueeze(0)
	batch_index = batch_index.repeat(time, 1, n_input).type(torch.int64)
	spikes.index_put_((spike_times.type(torch.int64), batch_index, index), torch.tensor(1.))

	# Temporary fix: The above code forces a spike from
	# every input neuron on the first time step.
	spikes[0, :, :] = 0

	# Return the input spike occurrence matrix.
	return spikes


def generate_2d_spike_train(image, intensity, time):
	'''
	Generates Poisson spike trains based on image ink intensity.
	'''
	# Multiply image by desired intensity.
	image = image * intensity

	# Get number of input neurons.
	n_input = image.shape[0]
	n_input_sqrt = int(np.sqrt(n_input))
	
	# Image data preprocessing (divide by 4, invert (for spike rates),
	# multiply by 1000 (conversion from milliseconds to seconds).
	image = (1 / (image / 4)) * 1000
	image[np.isinf(image)] = 0
	
	# Make the spike data.
	spike_times = np.random.poisson(image, [time, n_input])
	spike_times = np.cumsum(spike_times, axis=0)
	spike_times[spike_times >= time] = 0

	# Create spikes matrix from spike times.
	spikes = np.zeros([time, n_input])
	for idx in range(time):
		spikes[spike_times[idx, :], np.arange(n_input)] = 1

	# Temporary fix: The above code forces a spike from
	# every input neuron on the first time step.
	spikes[0, :] = 0

	# Return the input spike occurrence matrix.
	return spikes.reshape([time, 1, n_input_sqrt, n_input_sqrt])
