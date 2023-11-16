import os, sys
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from torchvision import datasets, transforms
from struct import unpack
import random

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
	
def get_MNIST(args, data_path=None, device=None):
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

def load_MNIST(train=True, data_path=None):
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

def get_MemoryMNIST(args, data_path=None, device=None):
	X0, y0 = load_MemoryMNIST(True, data_path=data_path, n=0)
	X1, y1 = load_MemoryMNIST(True, data_path=data_path, n=1)
	return [X0[0], y0[0]], [X1[0], y1[0]]

def load_MemoryMNIST(train=True, data_path=None, n=0):
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

	length = int((args.time - args.delay - args.decision) / args.dt)
	delay = int((args.delay) / args.dt)
	decision = int((args.decision) / args.dt)
	spikes = X0.float().unsqueeze(0).unsqueeze(0).repeat(int(args.time/args.dt), args.batch_size, 1).clone()
	spikes[length:length+delay, :, :] = 0. # delay
	targets = torch.zeros((args.batch_size, 1))

	for i in range(args.batch_size):
		num_0 = int(length * np.random.rand())
		num_1 = int(length - num_0)
		indices = np.random.choice(range(length), size=num_1, replace=False)
		spikes[indices, i, :] = X1.unsqueeze(0).repeat(num_1, 1) # watch
		dec = np.random.rand()
		if dec > 0.5 :
			spikes[length + delay:, i, :] = X1.unsqueeze(0).repeat(decision, 1) # decision
			targets[i] = torch.tensor(num_1 > num_0).float()
		else:
			targets[i] = torch.tensor(num_1 < num_0).float()

	return spikes, targets

def get_one_zero(args, data):
	sample = int(args.sample / args.dt)
	delay = int(args.delay / args.dt)
	decision = int(args.decision / args.dt)
	time = sample + delay + decision
	spikes = torch.zeros((time, args.batch_size, 2))
	targets = torch.zeros((args.batch_size, 1))
	for i in range(args.batch_size):
		num_0 = int(sample * np.random.rand())
		num_1 = int(sample - num_0)
		indices = np.random.choice(range(sample), size=num_1, replace=False)
		spikes[range(sample), i, :] = torch.tensor([[1., 0.]]).repeat(sample, 1)
		spikes[indices, i, :] = torch.tensor([[0., 1.]]).repeat(num_1, 1) # sample
		dec = np.random.rand()
		if dec > 0.5 :
			spikes[sample+delay:, i, :] = torch.tensor([[0., 1.]]).repeat(decision, 1)# decision
			targets[i] = torch.tensor(num_1 > num_0).float()
		else:
			spikes[sample+delay:, i, :] = torch.tensor([[1., 0.]]).repeat(decision, 1)# decision
			targets[i] = torch.tensor(num_1 < num_0).float()

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

class ABC(Dataset):
	def __init__(self, data, target):
		self.data = data
		self.target = target

	def __getitem__(self, index):
		image = self.data[index]
		label = self.target[index]
		return image, label

	def __len__(self):
		return len(self.data)
	
def get_abc(args, data_path=None, device=None):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as tensor.
	'''
	args.time = 11
	args.n_input = 37
	args.n_output = 37
	train_images, train_labels = load_abc(train=True, data_path=data_path)
	test_images, test_labels = load_abc(train=False, data_path=data_path)
	train_dataset = ABC(train_images, train_labels)
	test_dataset = ABC(test_images, test_labels)

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

def load_abc(train=True, data_path=None):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	fname = 'train' if train else 'test'

	if os.path.isfile(os.path.join(data_path, '%s_abc.p' % (fname))):
		# Get pickled data from disk.
		with open(os.path.join(data_path, '%s_abc.p' % (fname)), 'rb') as f:
			data = pickle.load(f)
		X = data['X']
		y = data['y']
	else:
		num_train = 60000
		num_val = 10000
		num_test = 10000

		step_num = 4
		elem_num = 26 + 10 + 1

		x_train = np.zeros([num_train, step_num * 2 + 3, elem_num], dtype=np.float32)
		x_val = np.zeros([num_val, step_num * 2 + 3, elem_num], dtype=np.float32)
		x_test = np.zeros([num_test, step_num * 2 + 3, elem_num], dtype=np.float32)

		y_train = np.zeros([num_train, elem_num], dtype=np.float32)
		y_val = np.zeros([num_val, elem_num], dtype=np.float32)
		y_test = np.zeros([num_test, elem_num], dtype=np.float32)

		for i in range(0, num_train):
			x_train[i], y_train[i] = generate_one(step_num, elem_num)
			print('Progress train data:', i, '/', num_train, '\n')

		for i in range(0, num_test):
			x_test[i], y_test[i] = generate_one(step_num, elem_num)
			print('Progress test data:', i, '/', num_test, '\n')

		for i in range(0, num_val):
			x_val[i], y_val[i] = generate_one(step_num, elem_num)
			print('Progress val data:', i, '/', num_val, '\n')

		with open(os.path.join(data_path, '%s_abc.p' % ('train')), 'wb') as f:
			data = {'X': x_train, 'y': y_train}
			pickle.dump(data, f)
		with open(os.path.join(data_path, '%s_abc.p' % ('test')), 'wb') as f:
			data = {'X': x_test, 'y': y_test}
			pickle.dump(data, f)
		with open(os.path.join(data_path, '%s_abc.p' % ('val')), 'wb') as f:
			data = {'X': x_val, 'y': y_val}
			pickle.dump(data, f)
		X = x_train if fname=='train' else x_test
		y = y_train if fname=='train' else y_test

	return X, y

def get_one_hot(c, elem_num):
    a = np.zeros([elem_num])
    if ord('a') <= ord(c) <= ord('z'):
        a[ord(c) - ord('a')] = 1
    elif ord('0') <= ord(c) <= ord('9'):
        a[ord(c) - ord('0') + 26] = 1
    else:
        a[-1] = 1
    return a


def generate_one(step_num, elem_num):
    a = np.zeros([step_num * 2 + 3, elem_num])
    d = {}
    st = ''

    for i in range(0, step_num):
        c = random.randint(0, 25) # 26个字母
        while c in d:
            c = random.randint(0, 25)
        b = random.randint(0, 9)
        d[c] = b
        s, t = chr(c + ord('a')), chr(b + ord('0'))
        st += s + t
        a[i*2] = get_one_hot(s, elem_num)
        a[i*2+1] = get_one_hot(t, elem_num)

    s = random.choice(list(d.keys()))
    t = chr(s + ord('a'))
    r = chr(d[s] + ord('0'))
    a[step_num * 2] = get_one_hot('?', elem_num)
    a[step_num * 2 + 1] = get_one_hot('?', elem_num)
    a[step_num * 2 + 2] = get_one_hot(t, elem_num)
    st += '??' + t + r
    e = get_one_hot(r, elem_num)
    return a, e

def get_spike_abc(args, data_path=None, device=None):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as tensor.
	'''
	args.n_input = 37
	args.n_output = 37
	args.repeat = 4
	args.time = 11*args.repeat
	train_images, train_labels = load_spike_abc(train=True, data_path=data_path, repeat=args.repeat)
	test_images, test_labels = load_spike_abc(train=False, data_path=data_path, repeat=args.repeat)
	train_dataset = ABC(train_images, train_labels)
	test_dataset = ABC(test_images, test_labels)

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

def load_spike_abc(train=True, data_path=None, repeat=5):
	'''
	Read input-vector (image) and target class (label, 0-9) and return it as 
	a list of tuples.
	'''
	fname = 'train' if train else 'test'

	if os.path.isfile(os.path.join(data_path, '%s_abc_repeat_%d.p' % (fname, repeat))):
		# Get pickled data from disk.
		with open(os.path.join(data_path, '%s_abc_repeat_%d.p' % (fname, repeat)), 'rb') as f:
			data = pickle.load(f)
		X = data['X']
		y = data['y']
	else:
		num_train = 60000
		num_val = 10000
		num_test = 10000

		step_num = 4
		elem_num = 26 + 10 + 1

		x_train = np.zeros([num_train, (step_num * 2 + 3) * repeat, elem_num], dtype=np.float32)
		x_val = np.zeros([num_val, (step_num * 2 + 3) * repeat, elem_num], dtype=np.float32)
		x_test = np.zeros([num_test, (step_num * 2 + 3) * repeat, elem_num], dtype=np.float32)

		y_train = np.zeros([num_train, elem_num], dtype=np.float32)
		y_val = np.zeros([num_val, elem_num], dtype=np.float32)
		y_test = np.zeros([num_test, elem_num], dtype=np.float32)

		for i in range(0, num_train):
			x_train[i], y_train[i] = generate_spike_one(step_num, elem_num, repeat)
			print('Progress train data:', i, '/', num_train, '\n')

		for i in range(0, num_test):
			x_test[i], y_test[i] = generate_spike_one(step_num, elem_num, repeat)
			print('Progress test data:', i, '/', num_test, '\n')

		for i in range(0, num_val):
			x_val[i], y_val[i] = generate_spike_one(step_num, elem_num, repeat)
			print('Progress val data:', i, '/', num_val, '\n')

		with open(os.path.join(data_path, '%s_abc_repeat_%d.p' % ('train', repeat)), 'wb') as f:
			data = {'X': x_train, 'y': y_train}
			pickle.dump(data, f)
		with open(os.path.join(data_path, '%s_abc_repeat_%d.p' % ('test', repeat)), 'wb') as f:
			data = {'X': x_test, 'y': y_test}
			pickle.dump(data, f)
		with open(os.path.join(data_path, '%s_abc_repeat_%d.p' % ('val', repeat)), 'wb') as f:
			data = {'X': x_val, 'y': y_val}
			pickle.dump(data, f)
		X = x_train if fname=='train' else x_test
		y = y_train if fname=='train' else y_test

	return X, y

def get_spike_one_hot(c, elem_num, repeat):
    a = np.zeros([repeat, elem_num])
    if ord('a') <= ord(c) <= ord('z'):
        a[:,ord(c) - ord('a')] = 1
    elif ord('0') <= ord(c) <= ord('9'):
        a[:,ord(c) - ord('0') + 26] = 1
    else:
        a[:,-1] = 1
    return a

def generate_spike_one(step_num, elem_num, repeat):
    a = np.zeros([(step_num * 2 + 3) * repeat, elem_num])
    d = {}
    st = ''

    for i in range(0, step_num):
        c = random.randint(0, 25) # 26个字母
        while c in d:
            c = random.randint(0, 25)
        b = random.randint(0, 9)
        d[c] = b
        s, t = chr(c + ord('a')), chr(b + ord('0')) # abc, 123
        st += s + t
        a[(2*i)*repeat:(2*i+1)*repeat] = get_spike_one_hot(s, elem_num, repeat)
        a[(2*i+1)*repeat: (2*i+2)*repeat] = get_spike_one_hot(t, elem_num, repeat)

    s = random.choice(list(d.keys()))
    t = chr(s + ord('a'))
    r = chr(d[s] + ord('0'))
    a[step_num*2*repeat: (step_num*2 + 1)*repeat] = get_spike_one_hot('?', elem_num, repeat)
    a[(step_num*2 + 1)*repeat : (step_num*2 + 2)*repeat] = get_spike_one_hot('?', elem_num, repeat)
    a[(step_num*2 + 2)*repeat : (step_num*2 + 3)*repeat] = get_spike_one_hot(t, elem_num, repeat)
    st += '??' + t + r
    e = get_spike_one_hot(r, elem_num, 1)
    return a, e


if __name__ == '__main__':
	num_train = 60000
	num_val = 10000
	num_test = 10000

	step_num = 4
	elem_num = 26 + 10 + 1

	x_train = np.zeros([num_train, step_num * 2 + 3, elem_num], dtype=np.float32)
	x_val = np.zeros([num_val, step_num * 2 + 3, elem_num], dtype=np.float32)
	x_test = np.zeros([num_test, step_num * 2 + 3, elem_num], dtype=np.float32)

	y_train = np.zeros([num_train, elem_num], dtype=np.float32)
	y_val = np.zeros([num_val, elem_num], dtype=np.float32)
	y_test = np.zeros([num_test, elem_num], dtype=np.float32)
	for i in range(0, num_train):
		x_train[i], y_train[i] = generate_one()

	for i in range(0, num_test):
		x_test[i], y_test[i] = generate_one()

	for i in range(0, num_val):
		x_val[i], y_val[i] = generate_one()

	d = {
        'x_train': x_train,
        'x_test': x_test,
        'x_val': x_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val
    }
	with open('associative-retrieval.pkl', 'wb') as f:
		pickle.dump(d, f, protocol=2)
