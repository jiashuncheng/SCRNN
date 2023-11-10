import os, sys
import time, timeit
import torch
import logging
import argparse
import numpy as np
import pickle
import pandas as pd

from struct import unpack
from datetime import datetime
from torchvision import datasets

import matplotlib.pyplot as plt

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../spiketorch"))))

from network import Network
from network import save_params, load_params, save_assignments, load_assignments, get_square_weights
from plotting import plot_input, plot_spikes, plot_weights, plot_performance
from classification import classify, assign_labels

from monitors import Monitor
from synapses import Synapses, STDPSynapses
from datasets import get_MNIST, generate_spike_train
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup

model_name = 'eth'

logs_path = os.path.join('logs', model_name)
data_path = os.path.join('data', model_name)
params_path = os.path.join('params', model_name)
results_path = os.path.join('results', model_name)
assign_path = os.path.join('assignments', model_name)
perform_path = os.path.join('performances', model_name)

for path in [logs_path, params_path, assign_path, results_path, perform_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

np.set_printoptions(threshold=sys.maxsize, linewidth=200)
torch.set_printoptions(threshold=sys.maxsize, linewidth=100, edgeitems=10)


parser = argparse.ArgumentParser(description='ETH (with LIF neurons) \
					SNN toy model simulation implemented with PyTorch.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_input', type=int, default=784)
parser.add_argument('--n_neurons', type=int, default=100)
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
parser.add_argument('--plot', type=str, default='False')

# Place parsed arguments in local scope.
args = parser.parse_args()

# Convert string arguments into boolean datatype.
plot = args.plot == 'True'
gpu = args.gpu == 'True'

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	assert torch.cuda.is_available()
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

# Set random number generator.
np.random.seed(args.seed)

# Record decaying spike traces to use STDP.
traces = args.mode == 'train'

# Build filename from command-line arguments.
fname = '_'.join([ str(args.n_neurons), str(args.n_train), str(args.seed), str(args.c_inhib), str(args.c_excite), str(args.wmax) ])

# Set logging configuration.
logging.basicConfig(format='%(message)s', 
					filename=os.path.join(logs_path, '%s.log' % fname),
					level=logging.DEBUG,
					filemode='w')

# Log argument values.
print('\nOptional argument values:')
for key, value in vars(args).items():
	print('-', key, ':', value)

# Initialize the spiking neural network.
network = Network(dt=args.dt)

# Add neuron populations.
network.add_group(InputGroup(args.n_input, traces=traces), 'X')
network.add_group(AdaptiveLIFGroup(args.n_neurons, traces=traces, rest=-65.0, reset=-65.0,
			threshold=-52.0, voltage_decay=1e-2, refractory=5, trace_tc=args.trace_tc), 'Ae')
network.add_group(LIFGroup(args.n_neurons, traces=traces, rest=-60.0, reset=-45.0,
		threshold=-40.0, voltage_decay=1e-1, refractory=2, trace_tc=args.trace_tc), 'Ai')

# Add synaptic connections between populations
if args.mode == 'train':
	network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae'],
			wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post), source='X', target='Ae')
elif args.mode == 'test':
	network.add_synapses(STDPSynapses(network.groups['X'], network.groups['Ae'],
					w=torch.from_numpy(load_params(params_path, fname, 'X_Ae')).to(device),
						wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post), name=('X', 'Ae'))

network.add_synapses(Synapses(network.groups['Ae'], network.groups['Ai'], 
					w=torch.diag(args.c_excite * torch.ones_like(torch.Tensor(args.n_neurons)))), source='Ae', target='Ai')
network.add_synapses(Synapses(network.groups['Ai'], network.groups['Ae'], w=-args.c_inhib * \
									(torch.ones_like(torch.Tensor(args.n_neurons, args.n_neurons)) - torch.diag(1 \
											* torch.ones_like(torch.Tensor(args.n_neurons))))), source='Ai', target='Ae')

# network.add_monitor(Monitor(obj=network.groups['Ae'], state_vars=['v', 'theta']), name=('Ae', ('v', 'theta')))
# network.add_monitor(Monitor(obj=network.groups['Ai'], state_vars=['v']), name=('Ai', 'v'))
# Get training or test data from disk.
train_data, test_data = get_MNIST(device=device)

# Count spikes from each neuron on each example (between update intervals).
outputs = torch.zeros_like(torch.Tensor(args.update_interval, args.n_neurons))

# Network simulation times.
image_time = args.time

# Run network simulation.
start = timeit.default_timer()

def log(info):
	logging.info(info)
	print(info)

def train():
	best_accuracy = -torch.inf
	# Voting schemes and neuron label assignments.
	voting_schemes = ['all']
	rates = torch.zeros((args.n_neurons, 10))
	performances = { scheme : [] for scheme in voting_schemes }
	assignments = -1 * torch.ones_like(torch.Tensor(args.n_neurons)).to(device)

	# Keep track of correct classifications for performance monitoring.
	correct = { scheme : 0 for scheme in voting_schemes }
	total_correct = { scheme : 0 for scheme in voting_schemes }
	
	intensity = 1
	for _, target in train_data : 
		shape=target.shape
		break
	inputs = torch.zeros((args.update_interval*shape[0], shape[1])).to(device)

	for idx, (image, target) in enumerate(train_data):
		image, target = image.to(device), target.to(device)
		inputs[idx*shape[0]:(idx+1)*shape[0]] = target
		if idx % args.print_interval == 0:
			# Log progress through dataset.
			log('Training progress: (%d / %d) - Elapsed time: %.1f h' % (idx, args.n_train, (timeit.default_timer() - start)/3600))

		if idx > 0 and idx % args.update_interval == 0:
			# Assign labels to neurons based on network spiking activity.
			rates, assignments = assign_labels(inputs, outputs, rates, assignments, args)
			# Assess performance of network on last `update_interval` examples.
			for scheme in performances.keys():
				performances[scheme].append(correct[scheme] / args.update_interval)  # Calculate percent correctly classified.
				correct[scheme] = 0  # Reset number of correct examples.
				
				log('%s -> (current) : %.4f | (best) : %.4f | (average) : %.4f' % (scheme,
					performances[scheme][-1], max(performances[scheme]), torch.tensor(performances[scheme]).mean()))

				# Save best accuracy.
				if performances[scheme][-1] > best_accuracy:
					best_accuracy = performances[scheme][-1]
					
					save_params(params_path, network.get_weights(('X', 'Ae')), fname, 'X_Ae')
					save_params(params_path, network.get_theta('Ae'), fname, 'theta')
					save_assignments(assign_path, assignments, fname)

			# Save sequence of performance estimates to file.
			with open(os.path.join(perform_path, fname), 'wb') as f:
				pickle.dump(performances, f)
			inputs = torch.zeros((args.update_interval*shape[0], shape[1])).to(device)

		inpts = {}

		# Encode current input example as Poisson spike trains.
		inpts['X'] = generate_spike_train(image, intensity * args.dt, int(image_time / args.dt))
		inpts['X'] = inpts['X'].to(device)

		# Run network on Poisson-encoded image data.
		spikes = network.run(args.mode, inpts, image_time)

		# Re-run image if there isn't any network activity.
		n_retries = 0
		while torch.sum(spikes['Ae']) < 5 and n_retries < 3:
			intensity += 1; n_retries += 1
			inpts['X'] = generate_spike_train(image, intensity * args.dt, int(image_time / args.dt))
			spikes = network.run(mode=args.mode, inpts=inpts, time=image_time)

		# Reset input intensity after any retries.
		intensity = 1

		# Classify network output (spikes) based on historical spiking activity.
		predictions = classify(spikes['Ae'], voting_schemes, assignments, args)

		# If correct, increment counter variable.
		for scheme in predictions.keys():
			if predictions[scheme][0] == target[0]:
				correct[scheme] += 1
				total_correct[scheme] += 1

		# Run zero image on network for `rest_time`.
		network.reset()

		# Add spikes from this iteration to the spike monitor
		outputs[idx % args.update_interval] = torch.sum(spikes['Ae'], 0)

	log('Training progress: (%d / %d) - Elapsed time: %.4f\n' % (args.n_train, args.n_train, timeit.default_timer() - start))

	results = {}
	for scheme in voting_schemes:
		results[scheme] = 100 * total_correct[scheme] / args.n_train
		log('Training accuracy for voting scheme "%s": %.4f\n' % (scheme, results[scheme]))


	# Save out network parameters and assignments for the test phase.
	save_params(params_path, network.get_weights(('X', 'Ae')), fname, 'X_Ae')
	save_params(params_path, network.get_theta('Ae'), fname, 'theta')
	save_assignments(assign_path, assignments, fname)

def test():
	# Voting schemes and neuron label assignments.
	voting_schemes = ['all']
	assignments = torch.from_numpy(load_assignments(assign_path, fname)).to(device)

	# Keep track of correct classifications for performance monitoring.
	correct = { scheme : 0 for scheme in voting_schemes }
	total_correct = { scheme : 0 for scheme in voting_schemes }
	intensity = 1
	
	for image, _ in train_data : 
		shape=image.shape
		break
	inputs = torch.zeros((args.update_interval*shape[0], shape[1])).to(device)

	for idx, (image, target) in enumerate(train_data):
		image, target = image.to(device), target.to(device)
		inputs[idx*shape[0]:(idx+1)*shape[0]] = image
		if idx % args.print_interval == 0:
			# Log progress through dataset.
			log('Training progress: (%d / %d) - Elapsed time: %.1f h' % (idx, args.n_train, (timeit.default_timer() - start)/3600))

		inpts = {}

		# Encode current input example as Poisson spike trains.
		inpts['X'] = generate_spike_train(image, intensity * args.dt, int(image_time / args.dt))
		inpts['X'] = inpts['X'].to(device)

		# Run network on Poisson-encoded image data.
		spikes = network.run(args.mode, inpts, image_time)

		# Re-run image if there isn't any network activity.
		n_retries = 0
		while torch.sum(spikes['Ae']) < 5 and n_retries < 3:
			intensity += 1; n_retries += 1
			inpts['X'] = generate_spike_train(image, intensity * args.dt, int(image_time / args.dt))
			spikes = network.run(mode=args.mode, inpts=inpts, time=image_time)

		# Reset input intensity after any retries.
		intensity = 1

		# Classify network output (spikes) based on historical spiking activity.
		predictions = classify(spikes['Ae'], voting_schemes, assignments, args)

		# If correct, increment counter variable.
		for scheme in predictions.keys():
			if predictions[scheme][0] == target[0]:
				correct[scheme] += 1
				total_correct[scheme] += 1

		# Run zero image on network for `rest_time`.
		network.reset()

		# Add spikes from this iteration to the spike monitor
		outputs[idx % args.update_interval] = torch.sum(spikes['Ae'], 0)

	log('Test progress: (%d / %d) - Elapsed time: %.4f\n' % (args.n_test, args.n_test, timeit.default_timer() - start))

	results = {}
	for scheme in voting_schemes:
		results[scheme] = 100 * total_correct[scheme] / args.n_test
		log('Test accuracy for voting scheme "%s": %.4f\n' % (scheme, results[scheme]))

	# Save out network parameters and assignments for the test phase.

	results = pd.DataFrame([[datetime.now(), fname] + list(results.values())], columns=['date', 'parameters'] + list(results.keys()))
	results_fname = '_'.join([str(args.n_neurons), str(args.n_train), 'results.csv'])
	
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