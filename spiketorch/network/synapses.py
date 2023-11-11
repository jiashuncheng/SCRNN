import torch
import torch.nn as nn

class Synapses(nn.Module):
	'''
	Specifies constant synapses between two populations of neurons.
	'''
	def __init__(self, source, target, w=None):
		super().__init__()
		self.source = source
		self.target = target

		if w is None:
			self.w = nn.Parameter(torch.rand(source.n, target.n))
		else:
			self.w = w

	def get_weights(self):
		return self.w.data

	def set_weights(self, w):
		self.w = w

class STDPSynapses(nn.Module):
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, w=None, nu_pre=1e-4, nu_post=1e-2, wmax=1.0, norm=78.0):
		super().__init__()
		self.source = source
		self.target = target

		if w is None:
			self.w = nn.Parameter(torch.rand(source.n, target.n))
		else:
			self.w = w

		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm

	def get_weights(self):
		return self.w

	def set_weights(self, w):
		self.w = w

	def get_source(self):
		return self.source

	def set_source(self, source):
		self.source = source

	def get_target(self):
		return self.target

	def set_target(self, target):
		self.target = target

	def normalize(self):
		'''
		Normalize weights to have average value `self.norm`.
		'''
		self.w.data *= self.norm / self.w.sum(0).view(1, -1)

	def update(self):
		'''
		Perform STDP weight update.
		'''
		# Post-synaptic.
		self.w.data += self.nu_post * (self.source.x.view(self.source.n, 1) * self.target.s.float().view(1, self.target.n))
		# Pre-synaptic.
		self.w.data -= self.nu_pre * (self.source.s.float().view(self.source.n, 1) * self.target.x.view(1, self.target.n))

		# Ensure that weights are within [0, self.wmax].
		self.w.data.clamp_(0, self.wmax)	
