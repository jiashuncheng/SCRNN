import os, sys
import time, timeit
import torch
import logging
import argparse
import numpy as np
import pickle
import pandas as pd
import torch.optim as optim

from struct import unpack
from datetime import datetime
import torch.nn as nn
from torchvision import datasets

import matplotlib.pyplot as plt

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../spiketorch"))))

from network import Network
from network import save_params, load_params, save_assignments, load_assignments, get_square_weights
from plotting import plot_input, plot_spikes, plot_weights, plot_performance
from classification import classify, assign_labels

from synapses import Synapses, STDPSynapses
from datasets import get_MNIST, generate_spike_train
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup

def log(info):
	logging.info(info)
	print(info)

def device_init(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	np.set_printoptions(threshold=sys.maxsize, linewidth=200)
	torch.set_printoptions(threshold=sys.maxsize, linewidth=100, edgeitems=10)
	if args.gpu == 'True':
		assert torch.cuda.is_available()
		torch.cuda.manual_seed(seed)
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")
	return device

parser = argparse.ArgumentParser(description='ETH (with LIF neurons) \
					SNN toy model simulation implemented with PyTorch.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_input', type=int, default=784)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--n_output', type=int, default=10)
parser.add_argument('--n_train', type=int, default=100)
parser.add_argument('--n_test', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=10)
parser.add_argument('--print_interval', type=int, default=10)
parser.add_argument('--nu_pre', type=float, default=1e-4)
parser.add_argument('--nu_post', type=float, default=1e-2)
parser.add_argument('--c_excite', type=float, default=22.5)
parser.add_argument('--c_inhib', type=float, default=17.5)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--rest', type=int, default=150)
parser.add_argument('--trace_tc', type=int, default=5e-2)
parser.add_argument('--wmax', type=float, default=1.0)
parser.add_argument('--dt', type=float, default=1)
parser.add_argument('--gpu', type=str, default='True')
parser.add_argument('--vote', type=str, default='True')
parser.add_argument('--plot', type=str, default='False')
parser.add_argument('--model_name', type=str, default='eth')

# Place parsed arguments in local scope.
args = parser.parse_args()

# Log argument values.
print('Optional argument values:')
for key, value in vars(args).items():
	print('--', key, ':', value)

logs_path = os.path.join('logs', args.model_name)
data_path = os.path.join('data', args.model_name)
params_path = os.path.join('params', args.model_name)
results_path = os.path.join('results', args.model_name)
assign_path = os.path.join('assignments', args.model_name)
perform_path = os.path.join('performances', args.model_name)

# Build filename from command-line arguments.
fname = '_'.join([ str(args.n_hidden), str(args.n_train), str(args.seed), str(args.c_inhib), str(args.c_excite), str(args.wmax) ])

# Set logging configuration.
logging.basicConfig(format='%(message)s', 
					filename=os.path.join(logs_path, '%s.log' % fname),
					level=logging.DEBUG,
					filemode='w')

# Set random number generator.
device = device_init(args.seed)

for path in [logs_path, params_path, assign_path, results_path, perform_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

def net(mode):
	if mode == 'train':
		w_in = None
		w_hidden = None
		w_out = None
		traces = True
	elif mode == 'test':
		w_in = torch.from_numpy(load_params(params_path, fname, 'x_h')).to(device)
		w_hidden = 0
		w_out = 0
		traces = False
	# Initialize the spiking neural network.
	network = Network(dt=args.dt)

	# Add neuron populations.
	network.add_group(InputGroup(args.n_input, traces=traces), 'x')
	network.add_group(LIFGroup(args.n_hidden, traces=traces, rest=-60.0, reset=-45.0,
				threshold=-52.0, voltage_decay=1e-2, refractory=5, trace_tc=args.trace_tc), 'h')
	network.add_group(LIFGroup(args.n_output, traces=traces, rest=-60.0, reset=-45.0,
			threshold=-52.0, voltage_decay=1e-2, refractory=5, trace_tc=args.trace_tc), 'y')

	network.layer1 = network.add_synapses(Synapses(network.groups['x'], network.groups['h'], w=None), source='x', target='h')
	network.layer2 = network.add_synapses(STDPSynapses(network.groups['h'], network.groups['h'],
						w=w_hidden, wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post), source='h', target='h')
	network.layer3 = network.add_synapses(Synapses(network.groups['h'], network.groups['y'], w=w_out), source='h', target='y')

	return network

network = net(args.mode)
optimizer = optim.Adam(network.parameters(), lr=0.001)
loss_fun = torch.nn.MSELoss()
# Get training and test data from disk.
train_data, test_data = get_MNIST(device=device)

# Run network simulation.
start = timeit.default_timer()

def train():
	# Count spikes from each neuron on each example (between update intervals).
	outputs = torch.zeros_like(torch.Tensor(args.update_interval, args.n_output))
	best_accuracy = -torch.inf
	# Voting schemes and neuron label assignments.
	rates = torch.zeros((args.n_output, args.n_output))
	performances = []
	assignments = -1 * torch.ones_like(torch.Tensor(args.n_output)).to(device)

	# Keep track of correct classifications for performance monitoring.
	correct = 0
	total_correct = 0
	
	intensity = 1
	for _, target in train_data : 
		shape=target.shape
		break
	targets = torch.zeros((args.update_interval*shape[0], shape[1])).to(device)

	for idx, (image, target) in enumerate(train_data):
		image, target = image.to(device), target.to(device)
		targets[idx*shape[0]:(idx+1)*shape[0]] = target
		if idx % args.print_interval == 0:
			# Log progress through dataset.
			log('Training progress: (%d / %d) - Elapsed time: %.1f h' % (idx, args.n_train, (timeit.default_timer() - start)/3600))

		if idx > 0 and idx % args.update_interval == 0:
			# Assign labels to neurons based on network spiking activity.
			rates, assignments = assign_labels(targets, outputs, rates, args)
			# Assess performance of network on last `update_interval` examples.
			performances.append(correct / args.update_interval)  # Calculate percent correctly classified.
			correct = 0  # Reset number of correct examples.
			
			log(' (current) : %.4f | (best) : %.4f | (average) : %.4f' % (performances[-1], max(performances), torch.tensor(performances).mean()))

			# Save best accuracy.
			if performances[-1] > best_accuracy:
				best_accuracy = performances[-1]
				
				save_params(params_path, network.get_weights(('x', 'h')), fname, 'x_h')
				save_params(params_path, network.get_weights(('h', 'h')), fname, 'h_h')
				save_params(params_path, network.get_weights(('h', 'y')), fname, 'h_y')
				save_assignments(assign_path, assignments, fname)

			# Save sequence of performance estimates to file.
			with open(os.path.join(perform_path, fname), 'wb') as f:
				pickle.dump(performances, f)
			targets = torch.zeros((args.update_interval*shape[0], shape[1])).to(device)

		inpts = {}

		# Encode current input example as Poisson spike trains.
		inpts['x'] = generate_spike_train(image, intensity * args.dt, int(args.time / args.dt))
		inpts['x'] = inpts['x'].to(device)

		# Run network on Poisson-encoded image data.
		spikes = network(args.mode, inpts, args.time)

		# Re-run image if there isn't any network activity.
		n_retries = 0
		while torch.sum(spikes['y']) < 5 and n_retries < 3:
			intensity += 1; n_retries += 1
			inpts['x'] = generate_spike_train(image, intensity * args.dt, int(args.time / args.dt))
			spikes = network(mode=args.mode, inpts=inpts, time=args.time)

		# Reset input intensity after any retries.
		intensity = 1

		# Classify network output (spikes) based on historical spiking activity.
		predictions = classify(spikes['y'], assignments, args)

		if args.vote == True:
			loss = loss_fun(predictions, target)
		else:
			loss = loss_fun(predictions, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# If correct, increment counter variable.
		if predictions[0] == target[0]:
			correct += 1
			total_correct += 1

		# Run zero image on network for `rest_time`.
		network.reset()

		# Add spikes from this iteration to the spike monitor
		outputs[idx % args.update_interval] = torch.sum(spikes['y'], 0)

	log('Training progress: (%d / %d) - Elapsed time: %.4f\n' % (args.n_train, args.n_train, timeit.default_timer() - start))

	results = 100 * total_correct / args.n_train
	log('Training accuracy: %.4f\n' % (results))

	# Save out network parameters and assignments for the test phase.
	save_params(params_path, network.get_weights(('x', 'h')), fname, 'x_h')
	save_params(params_path, network.get_weights(('h', 'h')), fname, 'h_h')
	save_params(params_path, network.get_weights(('h', 'y')), fname, 'h_y')
	save_assignments(assign_path, assignments, fname)

def test():
	# Count spikes from each neuron on each example (between update intervals).
	outputs = torch.zeros_like(torch.Tensor(args.update_interval, args.n_output))
	# Voting schemes and neuron label assignments.
	assignments = torch.from_numpy(load_assignments(assign_path, fname)).to(device)

	# Keep track of correct classifications for performance monitoring.
	correct = 0
	total_correct = 0
	intensity = 1
	
	for image, _ in train_data : 
		shape=image.shape
		break
	targets = torch.zeros((args.update_interval*shape[0], shape[1])).to(device)

	for idx, (image, target) in enumerate(train_data):
		image, target = image.to(device), target.to(device)
		targets[idx*shape[0]:(idx+1)*shape[0]] = target
		if idx % args.print_interval == 0:
			# Log progress through dataset.
			log('Training progress: (%d / %d) - Elapsed time: %.1f h' % (idx, args.n_train, (timeit.default_timer() - start)/3600))

		inpts = {}

		# Encode current input example as Poisson spike trains.
		inpts['x'] = generate_spike_train(image, intensity * args.dt, int(args.time / args.dt))
		inpts['x'] = inpts['x'].to(device)

		# Run network on Poisson-encoded image data.
		spikes = network(args.mode, inpts, args.time)

		# Re-run image if there isn't any network activity.
		n_retries = 0
		while torch.sum(spikes['y']) < 5 and n_retries < 3:
			intensity += 1; n_retries += 1
			inpts['x'] = generate_spike_train(image, intensity * args.dt, int(args.time / args.dt))
			spikes = network(mode=args.mode, inpts=inpts, time=args.time)

		# Reset input intensity after any retries.
		intensity = 1

		# Classify network output (spikes) based on historical spiking activity.
		predictions = classify(spikes['y'], assignments, args)

		# If correct, increment counter variable.
		if predictions[0] == target[0]:
			correct += 1
			total_correct += 1

		# Run zero image on network for `rest_time`.
		network.reset()

		# Add spikes from this iteration to the spike monitor
		outputs[idx % args.update_interval] = torch.sum(spikes['y'], 0)

	log('Test progress: (%d / %d) - Elapsed time: %.4f\n' % (args.n_test, args.n_test, timeit.default_timer() - start))

	results = 100 * total_correct / args.n_test
	log('Test accuracy: %.4f\n' % (results))

	# Save out network parameters and assignments for the test phase.

	results = pd.DataFrame([[datetime.now(), fname] + list(results)], columns=['date', 'parameters'] + list(results.keys()))
	results_fname = '_'.join([str(args.n_hidden), str(args.n_train), 'results.csv'])
	
	if not results_fname in os.listdir(results_path):
		results.to_csv(os.path.join(results_path, results_fname), index=False)
	else:
		all_results = pd.read_csv(os.path.join(results_path, results_fname))
		all_results = pd.concat([all_results, results], ignore_index=True)
		all_results.to_csv(os.path.join(results_path, results_fname), index=False)

if __name__ == "__main__":
	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		test()