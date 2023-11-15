import torch
import os, sys
import numpy as np
import torch.nn as nn
from collections import OrderedDict

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "networks"))))

from synapses import Synapses, STDPSynapses
from groups import InputGroup, LIFGroup, AdaptiveLIFGroup

class MemoryNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork, self).__init__()
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		# Parameters of others
		# self.neuron_i = nn.ReLU()
		# self.neuron_i.n = self.n_input
		# self.neuron_s = nn.ReLU()
		# self.neuron_s.n = 50
		# self.neuron_z = nn.ReLU()
		# self.neuron_z.n = 100
		# self.neuron_h = nn.ReLU()
		# self.neuron_h.n = self.n_hidden
		# self.relu_hs = nn.ReLU()
		# self.neuron_ho = nn.ReLU()
		# self.neuron_ho.n = 100
		# self.neuron_o = nn.ReLU()
		# self.neuron_o.n = self.n_output

		self.neuron_i = nn.ReLU()
		self.neuron_i.n = self.n_input
		self.neuron_s = nn.ReLU()
		self.neuron_s.n = self.n_hidden
		self.neuron_z = nn.ReLU()
		self.neuron_z.n = self.n_hidden
		self.neuron_h = nn.ReLU()
		self.neuron_h.n = self.n_hidden
		self.relu_hs = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_ho.n = self.n_hidden
		self.neuron_o = nn.ReLU()
		self.neuron_o.n = self.n_output

		# Parameter of others
		# self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='uniform', std=np.sqrt(20))
		# self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='xavier', factor=1)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='xavier', factor=1)
		self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='xavier', factor=1)
		self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='xavier', factor=1)
		self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='xavier', factor=1)
		self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='xavier', factor=1)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		h = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))

		for timestep in range(int(time / self.dt)):
			x = self.neuron_i(x_in[timestep, :])
			s = self.neuron_s(self.layer_i(x))
			z = self.neuron_z(self.layer_z(s))
			h = self.neuron_h(self.layer_c(z) + self.layer_h(h))
			hs = h.reshape(self.batch_size, 1, self.n_hidden)
			hh = hs
			a = self.lambda_ * a + self.eta * hs.transpose(1,2) @ hs

			for step in range(1):
				hs = self.layer_h(h).reshape(hh.shape) + self.layer_c(z).reshape(hh.shape) + hs @ a
				mu = torch.mean(hs, 0)
				sig = torch.sqrt(torch.mean(torch.pow((hs - mu), 2), 0))
				hs = self.relu_hs(torch.div(self.g * (hs - mu), sig) + self.b)
			h = hs.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(h))
		y = self.neuron_o(self.layer_o(o))
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class SpikeMemoryNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SpikeMemoryNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		self.neuron_i = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = AdaptiveLIFGroup(args.batch_size, 50, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = AdaptiveLIFGroup(args.batch_size, 100, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_h = AdaptiveLIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.relu_hs = nn.ReLU()
		self.neuron_ho = AdaptiveLIFGroup(args.batch_size, 100, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = AdaptiveLIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='uniform', std=np.sqrt(20))
		# self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='xavier', factor=1)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='xavier', factor=1)
		self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='xavier', factor=1)
		self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='xavier', factor=1)
		self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='xavier', factor=1)
		self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='xavier', factor=1)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		h = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))
		y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			x = self.neuron_i(x_in[timestep, :])
			s = self.neuron_s(self.layer_i(x))
			z = self.neuron_z(self.layer_z(s))
			h = self.neuron_h(self.layer_c(z) + self.layer_h(h))
			hs = h.reshape(self.batch_size, 1, self.n_hidden)
			hh = hs
			a = self.lambda_ * a + self.eta * hs.transpose(1,2) @ hs

			for step in range(1):
				hs = self.layer_h(h).reshape(hh.shape) + self.layer_c(z).reshape(hh.shape) + hs @ a
			h = hs.reshape(self.batch_size, self.n_hidden)

			o = self.neuron_ho(self.layer_ho(h))
			y = self.neuron_o(self.layer_o(o))
			y_out[timestep, :,:] = y
   
			# if timestep >= int(time / self.dt) - self.repeat:
			# 	o = self.neuron_ho(self.layer_ho(h))
			# 	y = self.neuron_o(self.layer_o(o))
			# 	y_out[timestep + self.repeat - int(time / self.dt), :,:] = y
    
			# # Update synapse weights if we're in training mode with STDP.
			# if mode == 'train' and isinstance(self.layer_h, STDPSynapses):
			# 	self.layer_h.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_h, STDPSynapses):
		# 	self.layer_h.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_i.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_h.resets()
		self.neuron_o.resets()

class Network(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(Network, self).__init__()
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		# Parameters of others
		# self.neuron_i = nn.ReLU()
		# self.neuron_i.n = self.n_input
		# self.neuron_s = nn.ReLU()
		# self.neuron_s.n = 50
		# self.neuron_z = nn.ReLU()
		# self.neuron_z.n = 100
		# self.neuron_h = nn.ReLU()
		# self.neuron_h.n = self.n_hidden
		# self.relu_hs = nn.ReLU()
		# self.neuron_ho = nn.ReLU()
		# self.neuron_ho.n = 100
		# self.neuron_o = nn.ReLU()
		# self.neuron_o.n = self.n_output

		self.neuron_i = nn.ReLU()
		self.neuron_i.n = self.n_input
		self.neuron_s = nn.ReLU()
		self.neuron_s.n = self.n_hidden
		self.neuron_z = nn.ReLU()
		self.neuron_z.n = self.n_hidden
		self.neuron_h = nn.ReLU()
		self.neuron_h.n = self.n_hidden
		self.relu_hs = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_ho.n = self.n_hidden
		self.neuron_o = nn.ReLU()
		self.neuron_o.n = self.n_output

		# Parameter of others
		# self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='uniform', std=np.sqrt(20))
		# self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='xavier', factor=1)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='xavier', factor=1)
		self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='xavier', factor=1)
		self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='xavier', factor=1)
		self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='xavier', factor=1)
		self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='xavier', factor=1)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		h = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))

		for timestep in range(int(time / self.dt)):
			x = self.neuron_i(x_in[timestep, :])
			s = self.neuron_s(self.layer_i(x))
			z = self.neuron_z(self.layer_z(s))
			h = self.neuron_h(self.layer_c(z) + self.layer_h(h))

		o = self.neuron_ho(self.layer_ho(h))
		y = self.neuron_o(self.layer_o(o))
		y = y.reshape([1, self.batch_size, self.n_output])
			
		return y

	def reset(self):
		pass

class SpikeNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SpikeNetwork, self).__init__()
		traces = args.mode = 'train'
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		self.neuron_i = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, 50, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, 100, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_h = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.relu_hs = nn.ReLU()
		self.neuron_ho = LIFGroup(args.batch_size, 100, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='uniform', std=np.sqrt(20))
		# self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_i = Synapses(self.neuron_i, self.neuron_s, init='xavier', factor=1)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='xavier', factor=1)
		self.layer_c = Synapses(self.neuron_z, self.neuron_h, init='xavier', factor=1)
		self.layer_h = Synapses(self.neuron_h, self.neuron_h, init='xavier', factor=1)
		self.layer_ho = Synapses(self.neuron_h, self.neuron_ho, init='xavier', factor=1)
		self.layer_o = Synapses(self.neuron_ho, self.neuron_o, init='xavier', factor=1)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		h = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))

		for timestep in range(int(time / self.dt)):
			x = self.neuron_i(x_in[timestep, :])
			s = self.neuron_s(self.layer_i(x))
			z = self.neuron_z(self.layer_z(s))
			h = self.neuron_h(self.layer_c(z) + self.layer_h(h))
			hs = h.reshape(self.batch_size, 1, self.n_hidden)
			hh = hs
			a = self.lambda_ * a + self.eta * hs.transpose(1,2) @ hs

			for step in range(1):
				hs = self.layer_h(h).reshape(hh.shape) + self.layer_c(z).reshape(hh.shape) + hs @ a
				mu = torch.mean(hs, 0)
				sig = torch.sqrt(torch.mean(torch.pow((hs - mu), 2), 0))
				hs = self.relu_hs(torch.div(self.g * (hs - mu), sig) + self.b)
			h = hs.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(h))
		y = self.neuron_o(self.layer_o(o))
		y = y.reshape([1, self.batch_size, self.n_output])

			# # Update synapse weights if we're in training mode with STDP.
			# if mode == 'train' and isinstance(self.layer_h, STDPSynapses):
			# 	self.layer_h.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_h, STDPSynapses):
		# 	self.layer_h.normalize()
			
		return y

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_i.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_h.resets()
		self.neuron_o.resets()

class Network_old(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(Network_old, self).__init__()
		traces = args.mode = 'train'
		self.dt = args.dt
		self.batch_size = args.batch_size
		self.neuron_i = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_h = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.layer_i = Synapses(self.neuron_i, self.neuron_h)
		# self.layer_h = STDPSynapses(self.neuron_h, self.neuron_h, wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post, batch_size=args.batch_size)
		self.layer_h = Synapses(self.neuron_h, self.neuron_h)
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

class SpikeNetwork_old(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SpikeNetwork_old, self).__init__()
		traces = args.mode = 'train'
		self.dt = args.dt
		self.batch_size = args.batch_size
		self.neuron_i = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_h = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.layer_i = Synapses(self.neuron_i, self.neuron_h)
		# self.layer_h = STDPSynapses(self.neuron_h, self.neuron_h, wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post, batch_size=args.batch_size)
		self.layer_h = Synapses(self.neuron_h, self.neuron_h)
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