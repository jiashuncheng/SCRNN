import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from abc import ABC, abstractmethod

class ActionFun(Function):
	@staticmethod
	def forward(ctx, v, threshold, window):
		ctx.save_for_backward(v, threshold, window)
		return v.gt(threshold).float()

	@staticmethod
	def backward(ctx, grad_output):
		v, threshold, window, = ctx.saved_tensors
		grad_input = grad_output.clone()
		temp = abs(v - threshold) < window
		return grad_input * temp.float(), None, None

class Group(ABC):
	'''
	Abstract base class for groups of neurons.
	'''
	def __init__(self):
		super().__init__()

	def step(self, inpts, mode):
		pass

	def get_spikes(self):
		return self.s

	def get_voltages(self):
		return self.v

	def get_traces(self):
		return self.x


class InputGroup(nn.Module, Group):
	'''
	Group of neurons clamped to input spikes.
	'''
	def __init__(self, batch_size, n, traces=False, trace_tc=5e-2, dt=1):
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.s = torch.zeros((batch_size, n))  # Spike occurences.
		self.dt = dt
		
		if self.traces:
			self.x = torch.zeros((batch_size, n))  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

	def forward(self, inpts):
		'''
		On each simulation step, set the spikes of the
		population equal to the inputs.
		'''
		self.s = inpts

		if self.traces:
			# Decay spike traces.
			self.x = self.x - self.dt * self.trace_tc * self.x
			# Setting synaptic traces.
			self.x[inpts.bool()] = 1.0
		
		return self.s


class LIFGroup(nn.Module, Group):
	'''
	Group of leaky integrate-and-fire neurons.
	'''
	def __init__(self, batch_size, n, traces=False, rest=0., reset=0., threshold=0.5, 
								refractory=2, voltage_decay=1., trace_tc=5e-2, window=0.2, dt=1., tau=3):
		
		super().__init__()
		self.batch_size = batch_size
		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = Variable(torch.tensor(threshold))  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.
		self.window =  Variable(torch.tensor(window))
		self.dt = dt
		self.tau = tau

		self.v = self.rest * torch.zeros((batch_size, n)) # Neuron voltages.
		self.s = torch.zeros((batch_size, n))  # Spike occurences.

		if traces:
			self.x = torch.zeros((batch_size, n)) # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros((batch_size, n))  # Refractory period counters.

	def forward(self, inpts):	
		# Decrement refractory counters.
		# self.inpts = torch.ones_like(self.inpts) #test
		# print(inpts.max(), inpts.mean(), inpts.min())
		self.refrac_count[self.refrac_count != 0] = self.refrac_count[self.refrac_count != 0] - self.dt
		# Integrate input and decay voltages.
		self.v =  self.voltage_decay * self.v + (self.dt/self.tau * ( -self.v + self.rest + inpts)) * (self.refrac_count == 0).float()
		# Check for spiking neurons.
		self.s = ActionFun.apply(self.v, self.threshold, self.window)
		self.refrac_count[self.s.bool()] = self.refractory
		self.v = self.v * (self.s == 0.).float() + self.reset * (self.s == 1.).float()

		if self.traces:
			# Decay spike traces.
			self.x = self.x - self.dt * self.trace_tc * self.x
			# Setting synaptic traces.
			self.x[self.s.bool()] = 1.0

		return self.s

if __name__ == "__main__":
	inpts = torch.ones((100, 1, 1)) * 1
	inpts[50:,:,:] = -1
	inpts[80:,:,:] = 1
	neuron = LIFGroup(1,1)
	v = []
	for i in range(100):
		v.append(neuron.v[0,0])
		neuron(inpts[i])
	import matplotlib.pyplot as plt
	print(max(v))
	plt.plot(range(100), v)
	plt.show()

# TODO
class AdaptiveLIFGroup(nn.Module, Group):
	'''
	Group of leaky integrate-and-fire neurons with adaptive thresholds.
	'''
	def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, refractory=5,
							voltage_decay=1e-2, theta_plus=0.05, theta_decay=1e-7, trace_tc=5e-2):
		
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.
		self.theta_plus = theta_plus  # Constant mV to raise threshold potential post-firing.
		self.theta_decay = theta_decay  # Rate of decay of adaptive threshold potential.

		self.v = self.rest * torch.ones_like(torch.Tensor(n))  # Neuron voltages.
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.
		self.theta = torch.zeros_like(torch.Tensor(n))  # Adaptive threshold parameters.

		self.action_fun = ActionFun.apply

		if traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros_like(torch.Tensor(n))  # Refractory period counters.

	def forward(self, inpts, mode, dt):
		# Decay voltages.
		self.v = self.v - dt * self.voltage_decay * (self.v - self.rest)

		if self.traces:
			# Decay spike traces and adaptive thresholds.
			self.x = self.x - dt * self.trace_tc * self.x
			self.theta = self.theta - dt * self.theta_decay * self.theta

		# Decrement refractory counters.
		self.refrac_count = self.refrac_count - dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold + self.theta) * (self.refrac_count <= 0)
		self.refrac_count[self.s] = dt * self.refractory
		self.v[self.s] = self.reset

		# Choose only a single neuron to spike (ETH replication).
		if torch.sum(self.s) > 0:
			s = torch.zeros_like(torch.Tensor(self.s.size()))
			s[torch.multinomial(self.s.float(), 1)] = 1

		# Integrate inputs.
		self.v = self.v + inpts

		if self.traces:
			# Update adaptive thresholds, synaptic traces.
			self.theta[self.s] = self.theta[self.s] + self.theta_plus
			self.x[self.s] = 1.0

		return self.s
