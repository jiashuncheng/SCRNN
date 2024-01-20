import torch
import os, sys
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import pickle

sys.path.insert(0, os.path.dirname(__file__))

from synapses import *
from groups import *

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
		# self.neuron_s = nn.ReLU()
		# self.neuron_s.n = self.n_input
		# self.neuron_s = nn.ReLU()
		# self.neuron_s.n = 50
		# self.neuron_z = nn.ReLU()
		# self.neuron_z.n = 100
		# self.neuron_r = nn.ReLU()
		# self.neuron_r.n = self.n_hidden
		# self.neuron_o = nn.ReLU()
		# self.neuron_ho = nn.ReLU()
		# self.neuron_ho.n = 100
		# self.neuron_o = nn.ReLU()
		# self.neuron_o.n = self.n_output

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		# Parameter of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ o

			for step in range(2):
				o = self.layer_r(r).reshape(o.shape) + self.layer_c(z).reshape(o.shape) + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		# y_out[timestep, :, :] = y
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class MemoryNetwork_1(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork_1, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ o

			for step in range(2):
				o = self.layer_r(o).reshape(o.shape) + self.layer_c(z).reshape(o.shape) + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		# y_out[timestep, :, :] = y
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class MemoryNetwork_2(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork_2, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			hh = o.clone()

			for step in range(1):
				o = self.layer_ro(r).reshape(o.shape) + self.layer_c(z).reshape(o.shape) + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ hh
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class MemoryNetwork_3(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork_3, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_x = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ma = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			zz = self.layer_c(z)
			r = self.neuron_r(zz + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			hh = o.clone()

			for step in range(1):
				o = self.layer_ro(r).reshape(o.shape) + self.layer_ma(self.neuron_x(zz)).reshape(o.shape) + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ hh
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class MemoryNetwork_4(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork_4, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_x = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ma = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)
		self.a = nn.Parameter(torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)))
		nn.init.xavier_uniform_(self.a)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		hh = torch.zeros((self.batch_size, 1, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			zz = self.layer_c(z)
			r = self.neuron_r(zz + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)

			for step in range(1):
				o = self.layer_ro(r).reshape(o.shape) + self.layer_ma(self.neuron_x(zz)).reshape(o.shape) + o @ self.a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class MemoryNetwork_5(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork_5, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_x = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ma = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			zz = self.layer_c(z)
			r = self.neuron_r(zz + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			hh = o.clone()

			for step in range(1):
				o = self.layer_ro(r).reshape(o.shape) + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ hh
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class MemoryNetwork_6(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(MemoryNetwork_6, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_x = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ma = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, 1, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))

		for timestep in range(int(time / self.dt)):
			if x_in.shape[2] == 6:
				s = self.neuron_s(x_in[timestep, :][:,:self.n_input])
			else:
				s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			zz = self.layer_c(z)
			r = self.neuron_r(zz + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			hh = o.clone()

			for step in range(1):
				if x_in.shape[2] == 6:
					a = self.layer_ma(self.neuron_x(self.layer_a(x_in[timestep, :][:,self.n_input:]))).reshape(o.shape)  + a
				else:
					a = self.layer_ma(self.neuron_x(zz)).reshape(o.shape)  + a
				o = self.layer_ro(r).reshape(o.shape) + a + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ hh
			r = o.reshape(self.batch_size, self.n_hidden)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		pass

class RNNNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(RNNNetwork, self).__init__()
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

		self.neuron_s = nn.ReLU()
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_o = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		# Parameter of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier', factor=5)
		self.layer_z = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_c = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_ho = Synapses(self.n_hidden, self.n_hidden, init='xavier', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))
			mu = torch.mean(r, 0)
			sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0)) + 1
			r = self.neuron_o(torch.div(self.g * (r - mu), sig) + self.b)

		o = self.neuron_ho(self.layer_ho(r))
		y = self.layer_y(o)
		# y_out[timestep, :, :] = y
		y_out = y.reshape([1, self.batch_size, self.n_output])

		return y_out

	def reset(self):
		pass

class SimpleMemoryNetwork_6(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_6, self).__init__()
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output
		self.layer_A = args.layer_A
		self.analyse_pre = args.analyse_pre // self.dt
		self.sample = args.sample // self.dt
		self.repeat = args.repeat // self.dt
		self.decision = args.decision // self.dt
		self.experiment = args.experiment
		self.store_A_state = args.store_A_state
		self.cut_atn_to_rsc = args.cut_atn_to_rsc

		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		self.neuron_a = nn.ReLU() # ACC
		self.neuron_r = nn.ReLU() # ATN
		self.neuron_o = nn.ReLU() # RSC
		self.neuron_o.h = []

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ma = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ao = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		a = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		o = torch.zeros((self.batch_size, 1, self.n_hidden)).to(self.device)
		A = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)).to(self.device)
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output)).to(self.device)
		aa = []

		for timestep in range(int(time / self.dt)):
			if x_in.shape[2] == 6:
				r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_r(r))
				a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a))
			else:
				r = self.neuron_r(self.layer_sr(x_in[timestep, :]) + self.layer_r(r))
				a = self.neuron_a(self.layer_ma(x_in[timestep, :]) + self.layer_a(a))				

			o = self.layer_ro(r).reshape(o.shape) + r.reshape(o.shape) @ A + self.layer_ao(a).reshape(o.shape)
			mu = torch.mean(o, 0)
			sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0) + 1e-1)
			o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			self.neuron_o.h.append(o.cpu().detach().numpy())
			if self.layer_A and timestep >= self.analyse_pre and timestep <= (self.sample * self.repeat):
				A = self.lambda_ * A + self.eta * o.transpose(1,2) @ r.reshape(o.shape)

			if self.decision > 1 and timestep >= int(time / self.dt) - self.decision:
				y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y.reshape([self.batch_size, self.n_output])

			if mode == 'analyse' and self.store_A_state:
				aa.append(A.cpu().detach())
			if mode == 'analyse' and self.cut_atn_to_rsc:
				A *= 0.
		if mode == 'analyse' and self.store_A_state:
			with open('/home/jiashuncheng/code/MANN/plot/data/A_7.pkl', 'wb') as a:
				pickle.dump(np.stack(aa), a)
			sys.exit()

		if self.decision == 1:
			y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
			y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		self.neuron_o.h = []
##
class SimpleMemoryNetwork_6_no_norm(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_6_no_norm, self).__init__()
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output
		self.layer_A = args.layer_A
		self.analyse_pre = args.analyse_pre // self.dt
		self.sample = args.sample // self.dt
		self.repeat = args.repeat // self.dt
		self.decision = args.decision // self.dt
		self.experiment = args.experiment
		self.store_A_state = args.store_A_state
		self.cut_atn_to_rsc = args.cut_atn_to_rsc

		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		self.neuron_a = nn.ReLU() # ACC
		self.neuron_r = nn.ReLU() # ATN
		self.neuron_o = nn.ReLU() # RSC
		self.neuron_o.h = []

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ma = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ao = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		a = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		o = torch.zeros((self.batch_size, 1, self.n_hidden)).to(self.device)
		A = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)).to(self.device)
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output)).to(self.device)
		aa = []

		for timestep in range(int(time / self.dt)):
			if x_in.shape[2] == 6:
				r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_r(r))
				a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a))
			else:
				r = self.neuron_r(self.layer_sr(x_in[timestep, :]) + self.layer_r(r))
				a = self.neuron_a(self.layer_ma(x_in[timestep, :]) + self.layer_a(a))				

			o = self.layer_ro(r).reshape(o.shape) + r.reshape(o.shape) @ A + self.layer_ao(a).reshape(o.shape)
			o = self.neuron_o(o)
			# mu = torch.mean(o, 0)
			# sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0) + 1e-1)
			# o = self.neuron_o(torch.div(self.g * (o - mu), sig) + self.b)
			self.neuron_o.h.append(o.cpu().detach().numpy())
			if self.layer_A and timestep >= self.analyse_pre and timestep <= (self.sample * self.repeat):
				A = self.lambda_ * A + self.eta * o.transpose(1,2) @ r.reshape(o.shape)

			if self.decision > 1 and timestep >= int(time / self.dt) - self.decision:
				y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y.reshape([self.batch_size, self.n_output])

			if mode == 'analyse' and self.store_A_state:
				aa.append(A.cpu().detach())
			if mode == 'analyse' and self.cut_atn_to_rsc:
				A *= 0.
		if mode == 'analyse' and self.store_A_state:
			with open('/home/jiashuncheng/code/MANN/plot/data/A_7.pkl', 'wb') as a:
				pickle.dump(np.stack(aa), a)
			sys.exit()

		if self.decision == 1:
			y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
			y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		self.neuron_o.h = []

class SimpleRNNNetwork(nn.Module):
	def __init__(self, args, device):
		super(SimpleRNNNetwork, self).__init__()
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		self.neuron_r = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)

		for timestep in range(int(time / self.dt)):
			r = self.layer_sr(x_in[timestep, :]) + self.layer_r(r)
			mu = torch.mean(r, 0)
			sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0) + 1e-1)
			r = self.neuron_r(torch.div(self.g * (r - mu), sig) + self.b)

		y = self.layer_y(r)
		y_out = y.reshape([1, self.batch_size, self.n_output])

		return y_out

	def reset(self):
		pass

class SimpleRNNNetwork_no_norm(nn.Module):
	def __init__(self, args, device):
		super(SimpleRNNNetwork_no_norm, self).__init__()
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))

		self.neuron_r = nn.ReLU()

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)

		for timestep in range(int(time / self.dt)):
			r = self.layer_sr(x_in[timestep, :]) + self.layer_r(r)
			r = self.neuron_r(r)
			# mu = torch.mean(r, 0)
			# sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0) + 1e-1)
			# r = self.neuron_r(torch.div(self.g * (r - mu), sig) + self.b)

		y = self.layer_y(r)
		y_out = y.reshape([1, self.batch_size, self.n_output])

		return y_out

	def reset(self):
		pass

class SimpleMemoryNetwork_two_neurons_new_stdp(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_two_neurons_new_stdp, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = nn.ReLU()
		self.neuron_z = nn.ReLU()
		self.neuron_r = nn.ReLU()
		self.neuron_r.r = None
		self.neuron_r.n = self.n_hidden
		self.neuron_hs = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_o = nn.ReLU()

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_r, init='rand', factor=3)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=3)
		self.layer_a = HebbSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=3)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		self.neuron_r.r = r
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision - 5, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			# r = self.neuron_r(self.layer_sr(s))
			# s = self.neuron_s(self.layer_sr(s))
			# if timestep == int(time / self.dt) - self.decision - self.delay:
			# 	r = r * torch.zeros((self.batch_size, self.n_hidden))
			# 	self.neuron_r.r = r
			# r = self.neuron_r(self.layer_c(s) + self.layer_r(r))
			r = self.neuron_r(self.layer_r(r) + self.layer_sr(s))
			self.neuron_r.r = r
			o = r
			# if mode == 'train' and timestep < self.sample * self.repeat and timestep >= 5:
			# 	self.layer_a.update()

			for step in range(1):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				o = self.layer_r(r) + self.layer_a(o) + self.layer_sr(s)
				# r = (self.layer_a(r)) + s
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0)) + 0.01
				o = self.neuron_hs(torch.div(self.g * (o - mu), sig) + self.b)
			r = o
			self.neuron_r.r = r
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
			if timestep >= int(time / self.dt) - self.decision + 5:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep - 5 + self.decision - int(time / self.dt), :,:] = y
    
		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		pass

class SpikeMemoryNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SpikeMemoryNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.decision = args.decision
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
		self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGrouphs(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = nn.ReLU() #LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threhold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=5)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=5)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=5)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=5)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=5)
		self.layer_y = Synapses(args.n_hidden, self.n_output, init='rand', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))
			o = r.reshape(self.batch_size, 1, self.n_hidden)
			a = self.lambda_ * a + self.eta * o.transpose(1,2) @ o

			for step in range(3):
				o = self.layer_r(r).reshape(o.shape) + self.layer_c(z).reshape(o.shape) + o @ a
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0)) + 1.
				o = self.neuron_hs(torch.div(self.g * (o - mu), 1.) + self.b)
			r = o.reshape(self.batch_size, self.n_hidden)

			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
   
			if timestep >= int(time / self.dt) - self.decision:
				o = self.neuron_ho(self.layer_ho(r))
				y = self.neuron_o(self.layer_y(o))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    
			# # Update synapse weights if we're in training mode with STDP.
			# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
			# 	self.layer_r.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		# self.neuron_o.resets()

class STDPSpikeMemoryNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(STDPSpikeMemoryNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = nn.ReLU() #LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threhold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=5)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=5)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=5)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=5)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))

			for step in range(5):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				self.layer_a.update()
				r = (self.layer_r(r) + self.layer_a(r)) + self.layer_c(z)
				mu = torch.mean(r, 0)
				sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0))
				r = self.neuron_hs(torch.div(self.g * (r - mu), sig) + self.b)
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)

			if timestep >= int(time / self.dt) - self.decision:
				o = self.neuron_ho(self.layer_ho(r))
				y = self.neuron_o(self.layer_y(o))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		# self.neuron_o.resets()

class SimpleSTDPSpikeMemoryNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSTDPSpikeMemoryNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			if timestep == int(time / self.dt) - self.decision - self.delay:
				self.neuron_r.v *= 0.
				r = torch.zeros((self.batch_size, self.n_hidden))
			# r = self.neuron_r(self.layer_c(s) + self.layer_r(r))
			r = self.neuron_r(s + self.layer_r(r))
			if timestep <= self.sample or timestep >= int(time / self.dt) - self.decision:
				self.layer_a.update()

			for step in range(5):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				# r = self.layer_r(r) + s
				# r = (self.layer_a(r)) + s
				r = self.neuron_hs((self.layer_r(r) + self.layer_a(r)) + s)
				# mu = torch.mean(r, 0)
				# sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0))
				# r = self.neuron_r(torch.div(self.g * (r - mu), sig) + self.b)
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
			if timestep >= int(time / self.dt) - self.decision:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()


class SimpleSTDPSpikeMemoryNetwork_two_neurons(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSTDPSpikeMemoryNetwork_two_neurons, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			# r = self.neuron_r(self.layer_sr(s))
			s = self.neuron_s(self.layer_sr(s))
			if timestep == int(time / self.dt) - self.decision - self.delay:
				self.neuron_r.v *= 0.
				r = torch.zeros((self.batch_size, self.n_hidden))
			r = self.neuron_r(self.layer_c(s) + self.layer_r(r))
			# r = self.neuron_r(s + self.layer_r(r))
			o = r
			if mode == 'train' and timestep <= self.sample * self.repeat:
				self.layer_a.update()
				# TODO stdp 用t和t+repeat的值
				# 只使用一次inner loop

			# self.layer_a.update()

			for step in range(5):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				o = self.layer_r(r) + self.layer_a(o)
				# r = (self.layer_a(r)) + s
				mu = torch.mean(o, 0)
				sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				o = self.neuron_hs(torch.div(self.g * (o - mu), sig) + self.b)
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
			if timestep >= int(time / self.dt) - self.decision + 5:
				y = self.neuron_o(self.layer_y(o))
				y_out[timestep - 5 + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()

class SimpleSTDPSpikeMemoryNetwork_two_neurons_new_stdp(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSTDPSpikeMemoryNetwork_two_neurons_new_stdp, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_r, init='rand', factor=3)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=3)
		self.layer_a = STDPSynapses_delta(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=3)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision - 5, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			# r = self.neuron_r(self.layer_sr(s))
			# s = self.neuron_s(self.layer_sr(s))
			# if timestep == int(time / self.dt) - self.decision - self.delay:
			# 	self.neuron_r.v *= 0.
			# 	r = torch.zeros((self.batch_size, self.n_hidden))
			# r = self.neuron_r(self.layer_c(s) + self.layer_r(r))
			r = self.neuron_r(self.layer_sr(s))
			o = r
			if mode == 'train' and timestep <= self.sample * self.repeat and timestep >= 5:
				self.layer_a.update()

			for step in range(1):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				o = self.layer_r(r) + self.layer_a(o) + self.layer_sr(s)
				# r = (self.layer_a(r)) + s
				# mu = torch.mean(o, 0)
				# sig = torch.sqrt(torch.mean(torch.pow((o - mu), 2), 0))
				# o = self.neuron_hs(torch.div(self.g * (o - mu), sig) + self.b)
				o = self.neuron_hs(o)
			r = o
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
			if timestep >= int(time / self.dt) - self.decision + 5:
				o = o.detach()
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep - 5 + self.decision - int(time / self.dt), :,:] = y
    
		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()

class SimpleSTDPSpikeMemoryNetwork_two_neurons_h(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSTDPSpikeMemoryNetwork_two_neurons_h, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			if timestep == int(time / self.dt) - self.decision - self.delay:
				self.neuron_r.v *= 0.
				r = torch.zeros((self.batch_size, self.n_hidden))
			# r = self.neuron_r(self.layer_c(s) + self.layer_r(r))
			r = self.neuron_r(s + self.layer_r(r))
			# if timestep <= self.sample * self.repeat or timestep >= int(time / self.dt) - self.decision:
			self.layer_a.update()
			# self.layer_a.update()

			for step in range(3):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				r = self.layer_r(r) + self.layer_a(r) + s
				# r = (self.layer_a(r)) + s
				mu = torch.mean(r, 0)
				sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0))
				r = self.neuron_hs(torch.div(self.g * (r - mu), sig) + self.b)
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
			if timestep >= int(time / self.dt) - self.decision:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()

class SimpleSTDPSpikeMemoryNetwork_one_neurons_A_W(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSTDPSpikeMemoryNetwork_one_neurons_A_W, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			if timestep == int(time / self.dt) - self.decision - self.delay:
				self.neuron_r.v *= 0.
				r = torch.zeros((self.batch_size, self.n_hidden))

			if timestep <= self.sample * self.repeat or timestep >= int(time / self.dt) - self.decision:
				self.layer_a.update()
			r = self.neuron_r(s + self.layer_r(r) + self.layer_a(r))

			if timestep >= int(time / self.dt) - self.decision:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()

class SimpleSTDPSpikeMemoryNetwork_one_neurons_AW(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSTDPSpikeMemoryNetwork_one_neurons_AW, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ha = MixtureSynapses(self.neuron_r, self.neuron_r, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			if timestep == int(time / self.dt) - self.decision - self.delay:
				self.neuron_r.v *= 0.
				r = torch.zeros((self.batch_size, self.n_hidden))

			# if timestep <= self.sample * self.repeat or timestep >= int(time / self.dt) - self.decision:
			self.layer_a.update()
			r = self.neuron_r(s + self.layer_ha(r))

			if timestep >= int(time / self.dt) - self.decision:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()

class STDPSpikeNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(STDPSpikeNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = nn.ReLU() #LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threhold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=5)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=5)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=5)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=5)
		self.layer_a = STDPSynapses(self.neuron_r, self.neuron_r, init='zeros', factor=1, batch_size=self.batch_size)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=5)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))
			# self.layer_a.update()

			for step in range(5):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				r = self.layer_r(r) + self.layer_c(z)
				mu = torch.mean(r, 0)
				sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0))
				r = self.neuron_hs(torch.div(self.g * (r - mu), sig) + self.b)
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)

			if timestep >= int(time / self.dt) - self.decision:
				o = self.neuron_ho(self.layer_ho(r))
				y = self.neuron_o(self.layer_y(o))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		# self.neuron_o.resets()

class SimpleSpikeMemoryNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSpikeMemoryNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = nn.ReLU() #LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threhold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			if timestep == int(time / self.dt) - self.decision:
				r = r * 0.
			r = self.neuron_r(self.layer_c(s) + self.layer_r(r))

			for step in range(5):
				# o = (self.layer_r(r).reshape(o.shape) + o @ a) + self.layer_c(z).reshape(o.shape)
				r = self.layer_r(r) + self.layer_c(s)
				mu = torch.mean(r, 0)
				sig = torch.sqrt(torch.mean(torch.pow((r - mu), 2), 0))
				r = self.neuron_hs(torch.div(self.g * (r - mu), sig) + self.b)
    
			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
			if timestep >= int(time / self.dt) - self.decision:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		# self.neuron_o.resets()


class SimpleSpikeNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleSpikeNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.sample = args.sample
		self.delay = args.delay
		self.decision = args.decision
		self.dt = args.dt
		self.device = device
		self.batch_size = args.batch_size
		self.n_input = args.n_input
		self.n_hidden = args.n_hidden
		self.n_output = args.n_output

		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		# self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		# self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=2)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=2)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=2)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_a = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=2)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=2)
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='rand', factor=2)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.layer_sr(s)
			# if timestep == int(time / self.dt) - self.decision - self.delay:
			# 	self.neuron_r.v *= 0.
			# 	r = torch.zeros((self.batch_size, self.n_hidden))
			# if timestep <= int(time / self.dt) - self.decision:
			# 	r = torch.zeros((self.batch_size, self.n_hidden))
			if timestep == int(time / self.dt) - self.decision:
				print(end='')
			# r = self.neuron_r(self.layer_c(s) + self.layer_r(r))
			r = self.neuron_r(s + self.layer_r(r))
			# self.layer_a.update()
   
			if timestep >= int(time / self.dt) - self.decision:
				y = self.neuron_o(self.layer_y(r))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		self.neuron_o.resets()


class Network(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(Network, self).__init__()
		self.dt = args.dt
		self.decision = args.decision
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
		# self.neuron_s = nn.ReLU()
		# self.neuron_s.n = self.n_input
		# self.neuron_s = nn.ReLU()
		# self.neuron_s.n = 50
		# self.neuron_z = nn.ReLU()
		# self.neuron_z.n = 100
		# self.neuron_r = nn.ReLU()
		# self.neuron_r.n = self.n_hidden
		# self.neuron_o = nn.ReLU()
		# self.neuron_ho = nn.ReLU()
		# self.neuron_ho.n = 100
		# self.neuron_o = nn.ReLU()
		# self.neuron_o.n = self.n_output

		self.neuron_s = nn.ReLU()
		self.neuron_s.n = self.n_input
		self.neuron_s = nn.ReLU()
		self.neuron_s.n = self.n_hidden
		self.neuron_z = nn.ReLU()
		self.neuron_z.n = self.n_hidden
		self.neuron_r = nn.ReLU()
		self.neuron_r.n = self.n_hidden
		self.neuron_o = nn.ReLU()
		self.neuron_ho = nn.ReLU()
		self.neuron_ho.n = self.n_hidden
		self.neuron_o = nn.ReLU()
		self.neuron_o.n = self.n_output

		# Parameter of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=5)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=5)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=5)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=5)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=5)
		self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='rand', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))

			if timestep >= int(time / self.dt) - self.decision:
				o = self.neuron_ho(self.layer_ho(r))
				y = self.neuron_o(self.layer_y(o))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
			
		return y_out

	def reset(self):
		pass

class SpikeNetwork(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SpikeNetwork, self).__init__()
		traces = args.mode = 'train'
		self.repeat = args.repeat
		self.decision = args.decision
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
		self.gamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		self.beta = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_s = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_z = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_hs = LIFGrouphs(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_ho = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = nn.ReLU() #LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threhold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)

		# Parameters of others
		# self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='uniform', std=np.sqrt(0.02))
		# self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='uniform', std=np.sqrt(0.01))
		# self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='uniform', std=np.sqrt(20))
		# self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='eye', factor=0.05)
		# self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='uniform', std=np.sqrt(0.01))
		# self.layer_y = Synapses(self.neuron_ho, self.neuron_o, init='uniform', std=np.sqrt(1.0/self.n_output))

		self.layer_sr = Synapses(self.neuron_s, self.neuron_s, init='rand', factor=5)
		self.layer_z = Synapses(self.neuron_s, self.neuron_z, init='rand', factor=5)
		self.layer_c = Synapses(self.neuron_z, self.neuron_r, init='rand', factor=5)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r, init='rand', factor=5)
		self.layer_ho = Synapses(self.neuron_r, self.neuron_ho, init='rand', factor=5)
		self.layer_y = Synapses(args.n_hidden, self.n_output, init='rand', factor=5)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		r = torch.zeros((self.batch_size, self.n_hidden))
		a = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden))
		# y_out = torch.zeros((int(time / self.dt), self.batch_size, self.n_output))
		y_out = torch.zeros((self.decision, self.batch_size, self.n_output))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			s = self.neuron_s(self.layer_sr(s))
			z = self.neuron_z(self.layer_z(s))
			r = self.neuron_r(self.layer_c(z) + self.layer_r(r))

			# o = self.neuron_ho(self.layer_ho(r))
			# y = self.neuron_o(self.layer_y(o))
			# y_out = y.reshape(1, self.batch_size, self.n_output)
   
			if timestep >= int(time / self.dt) - self.decision:
				o = self.neuron_ho(self.layer_ho(r))
				y = self.neuron_o(self.layer_y(o))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y
    
			# # Update synapse weights if we're in training mode with STDP.
			# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
			# 	self.layer_r.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_s.resets()
		self.neuron_s.resets()
		self.neuron_z.resets()
		self.neuron_ho.resets()
		self.neuron_hs.resets()
		self.neuron_r.resets()
		# self.neuron_o.resets()

class Network_old(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(Network_old, self).__init__()
		traces = args.mode = 'train'
		self.dt = args.dt
		self.batch_size = args.batch_size
		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.layer_sr = Synapses(self.neuron_s, self.neuron_r)
		# self.layer_r = STDPSynapses(self.neuron_r, self.neuron_r, wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post, batch_size=args.batch_size)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r)
		self.layer_y = Synapses(self.neuron_r, self.neuron_o)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		y_out_list = torch.zeros((int(time/self.dt), self.batch_size,self.neuron_o.n))
		r = torch.zeros((self.batch_size,self.neuron_r.n))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			r = self.neuron_r(self.layer_sr(s) + self.layer_r(r))
			y = self.neuron_o(self.layer_y(r))
			y_out_list[timestep,:, :] = y



			# Update synapse weights if we're in training mode with STDP.
			if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
				self.layer_r.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out_list

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_r.v = self.neuron_r.rest * torch.zeros((self.neuron_r.batch_size, self.neuron_r.n))
		self.neuron_o.v = self.neuron_o.rest * torch.zeros((self.neuron_o.batch_size, self.neuron_o.n))

		self.neuron_r.s[:] = 0.
		self.neuron_o.s[:] = 0.

		self.neuron_s.s[:] = 0
		self.neuron_r.s[:] = 0
		self.neuron_o.s[:] = 0

class SpikeNetwork_old(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SpikeNetwork_old, self).__init__()
		traces = args.mode = 'train'
		self.dt = args.dt
		self.batch_size = args.batch_size
		self.neuron_s = InputGroup(args.batch_size, args.n_input, traces=traces, dt=self.dt)
		self.neuron_r = LIFGroup(args.batch_size, args.n_hidden, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.neuron_o = LIFGroup(args.batch_size, args.n_output, traces=traces, rest=args.rest, reset=args.reset, threshold=args.threshold, voltage_decay=args.voltage_decay, refractory=args.refractory, trace_tc=args.trace_tc, dt=self.dt)
		self.layer_sr = Synapses(self.neuron_s, self.neuron_r)
		# self.layer_r = STDPSynapses(self.neuron_r, self.neuron_r, wmax=args.wmax, nu_pre=args.nu_pre, nu_post=args.nu_post, batch_size=args.batch_size)
		self.layer_r = Synapses(self.neuron_r, self.neuron_r)
		self.layer_y = Synapses(self.neuron_r, self.neuron_o)

	def forward(self, mode, x_in, time):
		'''
		Run network for a single iteration.
		'''
		# # Simulate neuron and synapse activity for `time` timesteps.
		y_out_list = torch.zeros((int(time/self.dt), self.batch_size,self.neuron_o.n))
		r = torch.zeros((self.batch_size,self.neuron_r.n))

		for timestep in range(int(time / self.dt)):
			s = self.neuron_s(x_in[timestep, :])
			r = self.neuron_r(self.layer_sr(s) + self.layer_r(r))
			y = self.neuron_o(self.layer_y(r))
			y_out_list[timestep,:, :] = y



			# Update synapse weights if we're in training mode with STDP.
			if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
				self.layer_r.update()

		# # Normalize synapse weights if we're in training mode.
		# if mode == 'train' and isinstance(self.layer_r, STDPSynapses):
		# 	self.layer_r.normalize()
			
		return y_out_list

	def reset(self):
		'''
		Resets certain state variables.
		'''
		self.neuron_r.v = self.neuron_r.rest * torch.zeros((self.neuron_r.batch_size, self.neuron_r.n))
		self.neuron_o.v = self.neuron_o.rest * torch.zeros((self.neuron_o.batch_size, self.neuron_o.n))

		self.neuron_r.s[:] = 0.
		self.neuron_o.s[:] = 0.

		self.neuron_s.s[:] = 0
		self.neuron_r.s[:] = 0
		self.neuron_o.s[:] = 0

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

	for s in range(kernel_size):
		for y in range(kernel_size):
			convolution_locations[(((neuron % n_patch_neurons_sqrt) * stride + (neuron // \
				n_patch_neurons_sqrt) * n_input_sqrt * stride) + (s * n_input_sqrt) + y)] = 1

	return convolution_locations