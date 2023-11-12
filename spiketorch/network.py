import torch
import os, sys
import numpy as np
import torch.nn as nn
from collections import OrderedDict

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "networks"))))

from synapses import Synapses, STDPSynapses
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup
 
class Network(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(Network, self).__init__()
		traces = args.mode = 'train'
		self.dt = args.dt
		self.batch_size = args.batch_size
		self.neuron_i = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_h = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.layer_i = Synapses(self.neuron_i, self.neuron_h)
		self.layer_h = STDPSynapses(self.neuron_h, self.neuron_h, wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post, batch_size=args.batch_size)
		# self.layer_h = Synapses(self.neuron_h, self.neuron_h)
		self.layer_o = Synapses(self.neuron_h, self.neuron_o)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		y_out_list = torch.zeros((int(time/self.dt), self.batch_size,self.neuron_o.n))
		h = torch.zeros((self.batch_size,self.neuron_h.n))

		for timestep in range(int(time / self.dt)):
			x = self.neuron_i(x_in[timestep, :])
			h = self.neuron_h(self.layer_i(x) + self.layer_h(h))
			y = self.neuron_o(self.layer_o(h))
			y_out_list[timestep,:, :] = y

			# Update synapse weights if we're in training mode with STDP.
			if mode == 'train' and isinstance(self.layer_h, STDPSynapses):
				self.layer_h.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_h, STDPSynapses):
		# 	self.layer_h.normalize()
			
		return y_out_list

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_h.v = self.neuron_h.rest * torch.zeros((self.neuron_h.batch_size, self.neuron_h.n))
		self.neuron_o.v = self.neuron_o.rest * torch.zeros((self.neuron_o.batch_size, self.neuron_o.n))

		self.neuron_h.s[:] = 0.
		self.neuron_o.s[:] = 0.

		self.neuron_i.x[:] = 0
		self.neuron_h.x[:] = 0
		self.neuron_o.x[:] = 0

def save_params(params_path, params, fname, prefix):
	'''
	Save network params to disk.

	Arguments:
		- params (numpy.ndarray): Array of params to save.
		- fname (str): File name of file to write to.
	'''
	torch.save(params, os.path.join(params_path, '_'.join([prefix, fname]) + '.pth'))


def load_params(params_path, fname, prefix):
	'''
	Load network params from disk.

	Arguments:
		- fname (str): File name of file to read from.
		- prefix (str): Name of the parameters to read from disk.

	Returns:
		- params (numpy.ndarray): Params stored in file `fname`.
	'''
	return torch.load(os.path.join(params_path, '_'.join([prefix, fname]) + '.pth'))


def save_assignments(assign_path, assignments, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- assignments (numpy.ndarray): Array of assignments to save.
		- fname (str): File name of file to write to.
	'''
	torch.save(assignments, os.path.join(assign_path, '_'.join(['assignments', fname]) + '.pth'))


def load_assignments(assign_path, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- fname (str): File name of file to read from.

	Returns:
		- assignments (numpy.ndarray): Assignments stored in file `fname`.
	'''
	return torch.load(os.path.join(assign_path, '_'.join(['assignments', fname]) + '.pth'))


def get_square_weights(weights, n_input_sqrt, n_neurons_sqrt):
	'''
	Get the weights from the input to excitatory layer and reshape them.
	'''
	square_weights = torch.zeros_like(torch.Tensor([n_input_sqrt * n_neurons_sqrt,
												n_input_sqrt * n_neurons_sqrt]))

	for n in range(n_neurons_sqrt ** 2):
		filtr = weights[:, n]
		square_weights[(n % n_neurons_sqrt) * n_input_sqrt : \
					((n % n_neurons_sqrt) + 1) * n_input_sqrt, \
					((n // n_neurons_sqrt) * n_input_sqrt) : \
					((n // n_neurons_sqrt) + 1) * n_input_sqrt] = \
						filtr.reshape([n_input_sqrt, n_input_sqrt])
	
	return square_weights


def get_conv_weights(weights, kernel_size, stride, n_patches, n_patch_neurons):
	rearranged = torch.zeros_like(torch.Tensor([kernel_size * n_patch_neurons, kernel_size * n_patches]))

	for patch in range(n_patches):
		for neuron in range(n_patch_neurons):
			rearranged[kernel_size * neuron : kernel_size * (neuron + 1), kernel_size * patch : kernel_size * (patch + 1)] = weights[patch]

	return rearranged.T


def get_convolution_locations(neuron, n_patch_neurons_sqrt, n_input_sqrt, kernel_size, stride):
	convolution_locations = [0] * (n_input_sqrt ** 2)

	for x in range(kernel_size):
		for y in range(kernel_size):
			convolution_locations[(((neuron % n_patch_neurons_sqrt) * stride + (neuron // \
				n_patch_neurons_sqrt) * n_input_sqrt * stride) + (x * n_input_sqrt) + y)] = 1

	return convolution_locations