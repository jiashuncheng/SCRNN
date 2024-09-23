import torch
import os, sys
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import pickle

sys.path.insert(0, os.path.dirname(__file__))

from synapses import *
from groups import *

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

class SimpleMemoryNetwork_2024(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_2024, self).__init__()
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
		self.mu = args.mu
		self.sigma = args.sigma

		self.alpha = torch.tensor([args.alpha], dtype=torch.float32).to(device)
		self.r0 = torch.tensor([args.r0], dtype=torch.float32).to(device)
		self.kd = torch.tensor([args.kd], dtype=torch.float32).to(device)
		self.u = torch.tensor([args.u], dtype=torch.float32).to(device)

		self.lambda_ = torch.tensor([args.lambda_], dtype=torch.float32).to(device)
		self.eta = torch.tensor([args.eta], dtype=torch.float32).to(device)

		self.eta0 = torch.tensor([args.eta0], dtype=torch.float32).to(device)
		self.wg = nn.Parameter(torch.ones((1, 1), dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.zeros([1, self.n_hidden], dtype=torch.float32))

		self.neuron_a = nn.ReLU() # ACC
		self.neuron_r = nn.ReLU() # ATN
		self.neuron_o = nn.ReLU() # RSC
		self.neuron_o.h = []

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ma = Synapses(self.n_input+1, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ao = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ar = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ra = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		a = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		o = torch.zeros((self.batch_size, 1, self.n_hidden)).to(self.device)
		A = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)).to(self.device)
		g = torch.zeros((self.batch_size, 1)).to(self.device)
		c = torch.zeros((self.batch_size, 1)).to(self.device)

		y_out = torch.zeros((self.decision, self.batch_size, self.n_output)).to(self.device)
		aa = []

		for timestep in range(int(time / self.dt)):
			# if timestep == 30:
			# 	print()
			if x_in.shape[2] == 5:
				a_t = a
				# a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r) + torch.normal(mean=self.mu, std=self.sigma, size=a.shape).to(self.device))
				a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r))
				# r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t) + torch.normal(mean=self.mu, std=self.sigma, size=r.shape).to(self.device))
				r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t))
			else:
				a_t = a
				a = self.neuron_a(self.layer_ma(x_in[timestep, :]) + self.layer_a(a) + self.layer_ra(r))
				r = self.neuron_r(self.layer_sr(x_in[timestep, :]) + self.layer_ar(a_t))				

			o = self.layer_ro(r).reshape(o.shape) + r.reshape(o.shape) @ A + self.layer_ao(a).reshape(o.shape)
			mu = torch.mean(o, 0)
			o = self.neuron_o(self.g * (o - mu) + self.b)
			self.neuron_o.h.append(o.cpu().detach().numpy())
			# # eta c g i 是scalar
			# g = (1-self.alpha) * g + self.alpha * i
			# k = self.r0/(2 * self.kd)
			# c = (1-self.kd) * c + self.r0 * g
			# eta = self.u*c/(k+c) + self.eta0
			if self.eta is None:
				eta = eta.unsqueeze(2)
			else:
				eta = self.eta
			if timestep > self.sample: # 没有noise了，所以这里是不是0也都可以了
				eta = 0
			# print(A.max(), A.min())
			if self.layer_A:
				A = self.lambda_ * A + eta * o.transpose(1,2) @ r.reshape(o.shape)
			# A = A + 1e-6
			# A = A / torch.sum(A, dim=1, keepdim=True) # 去掉A的norm

			if self.decision > 1 and timestep >= int(time / self.dt) - self.decision:
				y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y.reshape([self.batch_size, self.n_output])

			if mode == 'analyse' and self.store_A_state:
				aa.append(A.cpu().detach())
			if mode == 'analyse' and self.cut_atn_to_rsc:
				A *= 0.
		if mode == 'analyse' and self.store_A_state:
			with open('/home/jiashuncheng/code/MANN/plot/data/A_seed9_monitor_a.pkl', 'wb') as a:
				pickle.dump(np.stack(aa), a)
			sys.exit()

		if self.decision == 1:
			y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
			y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		self.neuron_o.h = []

class SimpleMemoryNetwork_20241(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_20241, self).__init__()
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
		self.mu = args.mu
		self.sigma = args.sigma

		self.alpha = torch.tensor([args.alpha], dtype=torch.float32).to(device)
		self.r0 = torch.tensor([args.r0], dtype=torch.float32).to(device)
		self.kd = torch.tensor([args.kd], dtype=torch.float32).to(device)
		self.u = torch.tensor([args.u], dtype=torch.float32).to(device)

		self.lambda_ = torch.tensor([args.lambda_], dtype=torch.float32).to(device)
		self.eta = torch.tensor([args.eta], dtype=torch.float32).to(device)

		self.eta0 = torch.tensor([args.eta0], dtype=torch.float32).to(device)
		self.wg = nn.Parameter(torch.ones((1, 1), dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.zeros([1, self.n_hidden], dtype=torch.float32))

		self.neuron_a = nn.ReLU() # ACC
		self.neuron_r = nn.ReLU() # ATN
		self.neuron_o = nn.ReLU() # RSC
		self.neuron_o.h = []
		self.neuron_a.h = []
		self.neuron_r.h = []

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ma = Synapses(self.n_input+1, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ao = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ar = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ra = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		a = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		o = torch.zeros((self.batch_size, 1, self.n_hidden)).to(self.device)
		A = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)).to(self.device)
		g = torch.zeros((self.batch_size, 1)).to(self.device)
		c = torch.zeros((self.batch_size, 1)).to(self.device)

		y_out = torch.zeros((self.decision, self.batch_size, self.n_output)).to(self.device)
		aa = []

		for timestep in range(int(time / self.dt)):
			# if timestep == 30:
			# 	print()
			if x_in.shape[2] == 5:
				a_t = a
				# a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r) + torch.normal(mean=self.mu, std=self.sigma, size=a.shape).to(self.device))
				a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r))
				# r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t) + torch.normal(mean=self.mu, std=self.sigma, size=r.shape).to(self.device))
				r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t))
			else:
				a_t = a
				a = self.neuron_a(self.layer_ma(x_in[timestep, :]) + self.layer_a(a) + self.layer_ra(r))
				r = self.neuron_r(self.layer_sr(x_in[timestep, :]) + self.layer_ar(a_t))				

			o = self.layer_ro(r).reshape(o.shape) + r.reshape(o.shape) @ A + self.layer_ao(a).reshape(o.shape)
			mu = torch.mean(o, 0)
			o = self.neuron_o(self.g * (o - mu) + self.b)
			self.neuron_o.h.append(o.cpu().detach().numpy())
			self.neuron_a.h.append(a.cpu().detach().numpy())
			self.neuron_r.h.append(r.cpu().detach().numpy())
			# # eta c g i 是scalar
			# g = (1-self.alpha) * g + self.alpha * i
			# k = self.r0/(2 * self.kd)
			# c = (1-self.kd) * c + self.r0 * g
			# eta = self.u*c/(k+c) + self.eta0
			if self.eta is None:
				eta = eta.unsqueeze(2)
			else:
				eta = self.eta
			if timestep > self.sample: # 没有noise了，所以这里是不是0也都可以了
				eta = 0
			# print(A.max(), A.min())
			if self.layer_A:
				A = self.lambda_ * A + eta * o.transpose(1,2) @ r.reshape(o.shape)
			# A = A + 1e-6
			# A = A / torch.sum(A, dim=1, keepdim=True) # 去掉A的norm

			if self.decision > 1 and timestep >= int(time / self.dt) - self.decision:
				y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y.reshape([self.batch_size, self.n_output])

			if mode == 'analyse' and self.store_A_state:
				aa.append(A.cpu().detach())
			if mode == 'analyse' and self.cut_atn_to_rsc:
				A *= 0.
		if mode == 'analyse' and self.store_A_state:
			with open('/home/jiashuncheng/code/MANN/plot/data/A_seed9_monitor_a.pkl', 'wb') as a:
				pickle.dump(np.stack(aa), a)
			sys.exit()

		if self.decision == 1:
			y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
			y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		self.neuron_o.h = []

class SimpleMemoryNetwork_20242(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_20242, self).__init__()
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
		self.mu = args.mu
		self.sigma = args.sigma

		self.alpha = torch.tensor([args.alpha], dtype=torch.float32).to(device)
		self.r0 = torch.tensor([args.r0], dtype=torch.float32).to(device)
		self.kd = torch.tensor([args.kd], dtype=torch.float32).to(device)
		self.u = torch.tensor([args.u], dtype=torch.float32).to(device)

		self.lambda_ = torch.tensor([args.lambda_], dtype=torch.float32).to(device)
		self.eta = torch.tensor([args.eta], dtype=torch.float32).to(device)

		self.eta0 = torch.tensor([args.eta0], dtype=torch.float32).to(device)
		self.wg = nn.Parameter(torch.ones((1, 1), dtype=torch.float32))
		self.g = nn.Parameter(torch.ones([1, self.n_hidden], dtype=torch.float32))
		self.b = nn.Parameter(torch.zeros([1, self.n_hidden], dtype=torch.float32))

		self.neuron_a = nn.ReLU() # ACC
		self.neuron_r = nn.ReLU() # ATN
		self.neuron_o = nn.ReLU() # RSC
		'''
		self.neuron_a = nn.Tanh() # ACC
		self.neuron_r = nn.Tanh() # ATN
		self.neuron_o = nn.Sigmoid() # RSC
		'''
		self.neuron_o.h = []
		self.neuron_a.h = []
		self.neuron_r.h = []

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ma = Synapses(self.n_input+1, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ao = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ar = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ra = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		a = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		o = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		A = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)).to(self.device)
		g = torch.zeros((self.batch_size, 1)).to(self.device)
		c = torch.zeros((self.batch_size, 1)).to(self.device)

		y_out = torch.zeros((self.decision, self.batch_size, self.n_output)).to(self.device)
		aa = []

		for timestep in range(int(time / self.dt)):
			# if timestep == 30:
			# 	print()
			if x_in.shape[2] == 5:
				a_t = a
				# a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r) + torch.normal(mean=self.mu, std=self.sigma, size=a.shape).to(self.device))
				a = self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r)
				a = self.neuron_a(a)
				# r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t) + torch.normal(mean=self.mu, std=self.sigma, size=r.shape).to(self.device))
				r = self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t)
				r = self.neuron_r(r)
			else:
				a_t = a
				a = self.neuron_a(self.layer_ma(x_in[timestep, :]) + self.layer_a(a) + self.layer_ra(r))
				r = self.neuron_r(self.layer_sr(x_in[timestep, :]) + self.layer_ar(a_t))				

			o = self.layer_ro(r) + (A @ r.unsqueeze(2)).squeeze(2) + self.layer_ao(a)
			# mu = torch.mean(o, 0, keepdim=True)
			# sigma = torch.std(o, 1, keepdim=True)
			# o = self.neuron_o(self.g / sigma * (o - mu) + self.b)
			# o = self.neuron_o(self.g * (o - mu) + self.b)
			o = self.neuron_o(o)
			self.neuron_o.h.append(o.cpu().detach().numpy())
			self.neuron_a.h.append(a.cpu().detach().numpy())
			self.neuron_r.h.append(r.cpu().detach().numpy())
			# # eta c g i 是scalar
			# g = (1-self.alpha) * g + self.alpha * i
			# k = self.r0/(2 * self.kd)
			# c = (1-self.kd) * c + self.r0 * g
			# eta = self.u*c/(k+c) + self.eta0
			if self.eta is None:
				eta = eta.unsqueeze(2)
			else:
				eta = self.eta
			if timestep > self.sample: # 没有noise了，所以这里是不是0也都可以了
				eta = 0
			# print(A.max(), A.min())
			if self.layer_A:
				A = self.lambda_ * A + eta * o.unsqueeze(2) @ r.unsqueeze(1) 
			# A = A + 1e-6
			# A = A / torch.sum(A, dim=1, keepdim=True) # 去掉A的norm

			if self.decision > 1 and timestep >= int(time / self.dt) - self.decision:
				y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y.reshape([self.batch_size, self.n_output])

			if mode == 'analyse' and self.store_A_state:
				aa.append(A.cpu().detach())
			if mode == 'analyse' and self.cut_atn_to_rsc:
				A *= 0.
		if mode == 'analyse' and self.store_A_state:
			with open('/home/jiashuncheng/code/MANN/plot/data/A_seed9_monitor_a.pkl', 'wb') as a:
				pickle.dump(np.stack(aa), a)
			sys.exit()

		if self.decision == 1:
			y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
			y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		self.neuron_o.h = []

class SimpleMemoryNetwork_20243(nn.Module):
	'''
	Combines neuron groups and synapses into a spiking neural network.
	'''
	def __init__(self, args, device):
		super(SimpleMemoryNetwork_20243, self).__init__()
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
		self.mu = args.mu
		self.sigma = args.sigma

		self.alpha = args.alpha
		self.r0 = args.r0
		self.kd = args.kd 
		self.u = args.u 
		self.eta0 = args.eta0 
		self.wg = args.wg 
		self.rc = args.rc 
		self.eta = args.eta

		self.lambda_ = torch.tensor([args.lambda_], dtype=torch.float32).to(device)

		self.neuron_a = nn.ReLU() # ACC
		self.neuron_r = nn.ReLU() # ATN
		self.neuron_o = nn.ReLU() # RSC

		self.neuron_o.h = []
		self.neuron_a.h = []
		self.neuron_r.h = []
		self.gs = []

		self.layer_sr = Synapses(self.n_input, self.n_hidden, init='xavier')
		self.layer_ma = Synapses(self.n_input+1, self.n_hidden, init='xavier')
		self.layer_ro = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ao = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ar = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_ra = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_r = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_a = Synapses(self.n_hidden, self.n_hidden, init='xavier')
		self.layer_y = Synapses(self.n_hidden, self.n_output, init='xavier')

	def forward(self, mode, x_in, time):
		r = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		a = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		o = torch.zeros((self.batch_size, self.n_hidden)).to(self.device)
		A = torch.zeros((self.batch_size, self.n_hidden, self.n_hidden)).to(self.device)
		g = torch.zeros((self.batch_size, 1, 1)).to(self.device)
		c = torch.zeros((self.batch_size, 1, 1)).to(self.device)

		y_out = torch.zeros((self.decision, self.batch_size, self.n_output)).to(self.device)
		aa = []

		for timestep in range(int(time / self.dt)):
			# if timestep == 30:
			# 	print()
			if x_in.shape[2] == 5:
				a_t = a
				# a = self.neuron_a(self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r) + torch.normal(mean=self.mu, std=self.sigma, size=a.shape).to(self.device))
				a = self.layer_ma(x_in[timestep, :][:,self.n_input:]) + self.layer_a(a) + self.layer_ra(r)
				a = self.neuron_a(a)
				# r = self.neuron_r(self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t) + torch.normal(mean=self.mu, std=self.sigma, size=r.shape).to(self.device))
				r = self.layer_sr(x_in[timestep, :][:,:self.n_input]) + self.layer_ar(a_t)
				r = self.neuron_r(r)
			else:
				a_t = a
				a = self.neuron_a(self.layer_ma(x_in[timestep, :]) + self.layer_a(a) + self.layer_ra(r))
				r = self.neuron_r(self.layer_sr(x_in[timestep, :]) + self.layer_ar(a_t))				

			o = self.layer_ro(r) + (A @ r.unsqueeze(2)).squeeze(2) + self.layer_ao(a)
			o = self.neuron_o(o)
			self.neuron_o.h.append(o.cpu().detach().numpy())
			self.neuron_a.h.append(a.cpu().detach().numpy())
			self.neuron_r.h.append(r.cpu().detach().numpy())
			# # eta c g i 是scalar
			if self.eta == -1:
				g = (1-self.alpha) * g + self.alpha * np.tanh(self.wg * self.rc) + \
					torch.normal(mean=self.mu, std=self.sigma, size=g.shape).to(self.device)
				k = self.r0/(2 * self.kd)
				c = (1-self.kd) * c + self.r0 * g
				eta = self.u*c/(k+c) + self.eta0
			else:
				eta = self.eta
			if timestep > self.sample: # 没有noise了，所以这里是不是0也都可以了
				eta = 0
			# print(A.max(), A.min())
			if self.layer_A:
				A = self.lambda_ * A + eta * o.unsqueeze(2) @ r.unsqueeze(1) 
			# A = A + 1e-6
			# A = A / torch.sum(A, dim=1, keepdim=True) # 去掉A的norm
			self.gs.append(g.cpu().detach().numpy())

			if self.decision > 1 and timestep >= int(time / self.dt) - self.decision:
				y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
				y_out[timestep + self.decision - int(time / self.dt), :,:] = y.reshape([self.batch_size, self.n_output])

			if mode == 'analyse' and self.store_A_state:
				aa.append(A.cpu().detach())
			if mode == 'analyse' and self.cut_atn_to_rsc:
				A *= 0.
		if mode == 'analyse' and self.store_A_state:
			with open('/home/jiashuncheng/code/MANN/plot/data/A_seed9_monitor_a.pkl', 'wb') as a:
				pickle.dump(np.stack(aa), a)
			sys.exit()

		if self.decision == 1:
			y = self.layer_y(o.reshape(self.batch_size, self.n_hidden))
			y_out = y.reshape([1, self.batch_size, self.n_output])
			
		return y_out

	def reset(self):
		self.neuron_o.h = []
		self.neuron_a.h = []
		self.neuron_r.h = []
		self.gs = []

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
