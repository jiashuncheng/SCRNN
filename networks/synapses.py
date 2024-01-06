import torch
import torch.nn as nn

class Synapses(nn.Module):
	'''
	Specifies constant synapses between two populations of neurons.
	'''
	def __init__(self, source, target, init=None, std=None, factor=None):
		super().__init__()
		self.source = source
		self.target = target
		n_source = self.source if isinstance(self.source, int) else self.source.n
		n_target = self.target if isinstance(self.target, int) else self.target.n

		if init == 'uniform':
			self.w = nn.Parameter(torch.empty(n_source, n_target))
			nn.init.uniform_(self.w, a=-std, b=std)
			self.b = nn.Parameter(torch.zeros([1, n_target]))
		elif init == 'eye':
			self.w = nn.Parameter(factor * torch.eye(n_source, n_target))
			self.b = nn.Parameter(torch.zeros([1, n_target]))
		elif init == 'xavier':
			self.w = nn.Parameter((torch.empty(n_source, n_target)))
			nn.init.xavier_uniform_(self.w)
			self.b = nn.Parameter(torch.zeros([1, n_target]))
		elif init == 'rand':
			self.w = nn.Parameter((torch.rand(n_source, n_target) - 0.5) * factor)
			self.b = nn.Parameter(torch.zeros([1, n_target]))
		elif init == 'rand_nozeros':
			self.w = nn.Parameter((torch.rand(n_source, n_target) - 0.5) * factor)
			self.b = nn.Parameter(torch.zeros([1, n_target]))

	def forward(self, spike):
		return spike @ self.w

class STDPSynapses(nn.Module):
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, batch_size=1, nu_pre=1e-2, nu_post=1e-2, wmax=2.5, norm=78.0, init=None, factor=None):
		super().__init__()
		self.source = source
		self.target = target
		self.batch_size = batch_size

		# self.w = nn.Parameter(torch.rand(source.n, target.n))
		if init == 'xavier':
			self.w = torch.rand(source.n, target.n)
			nn.init.xavier_uniform_(self.w)
			# self.w.data *= 2
		elif init == 'zeros':
			self.w = torch.zeros(source.n, target.n)
		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm
		self.stdp_lr = 1e-1

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
		post = self.nu_post * (self.source.x.view(self.source.n, self.batch_size) @ self.target.s.float().view(self.batch_size, self.target.n))
		# Pre-synaptic.
		pre = -self.nu_pre * (self.source.s.float().view(self.source.n, self.batch_size) @ self.target.x.view(self.batch_size, self.target.n))

		self.delta_w = self.stdp_lr * post + self.stdp_lr * pre
		self.w.data = self.lambda_ * self.w.data + self.eta * self.delta_w
		
		# Ensure that weights are within [0, self.wmax].
		self.w.data.clamp_(-self.wmax, self.wmax)	

	def forward(self, spike):
		return spike.float() @ self.w
	

class STDPSynapses_delta(nn.Module):
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, batch_size=1, nu_pre=1e-2, nu_post=1e-2, wmax=2.5, norm=78.0, init=None, factor=None):
		super().__init__()
		self.source = source
		self.target = target
		self.batch_size = batch_size

		# self.w = nn.Parameter(torch.rand(source.n, target.n))
		if init == 'xavier':
			self.w = torch.rand(source.n, target.n)
			nn.init.xavier_uniform_(self.w)
			# self.w.data *= 2
		elif init == 'zeros':
			self.w = torch.zeros(source.n, target.n)
		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm
		self.stdp_lr = 2

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
		post = self.nu_post * (self.source.x_list[self.source.inner_clock - 5].view(self.source.n, self.batch_size) @ self.target.s.float().view(self.batch_size, self.target.n))
		# Pre-synaptic.
		pre = -self.nu_pre * (self.source.s_list[self.source.inner_clock - 5].float().view(self.source.n, self.batch_size) @ self.target.x.view(self.batch_size, self.target.n))

		self.delta_w = self.stdp_lr * post + self.stdp_lr * pre
		self.w.data = self.lambda_ * self.w.data + self.eta * self.delta_w
		
		# Ensure that weights are within [0, self.wmax].
		self.w.data.clamp_(-self.wmax, self.wmax)	

	def forward(self, spike):
		return spike.float() @ self.w

class HebbSynapses(nn.Module):
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, batch_size=1, nu_pre=1e-2, nu_post=1e-2, wmax=2.5, norm=78.0, init=None, factor=None):
		super().__init__()
		self.source = source
		self.target = target
		self.batch_size = batch_size

		# self.w = nn.Parameter(torch.rand(source.n, target.n))
		if init == 'xavier':
			self.w = torch.rand(source.n, target.n)
			nn.init.xavier_uniform_(self.w)
			# self.w.data *= 2
		elif init == 'zeros':
			self.w = torch.zeros(source.n, target.n)
		self.lambda_ = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm
		self.stdp_lr = 2

	def normalize(self):
		'''
		Normalize weights to have average value `self.norm`.
		'''
		self.w.data *= self.norm / self.w.sum(0).view(1, -1)

	def update(self):
		self.delta_w = self.stdp_lr * self.source.h.t() @ self.source.h
		self.w.data = self.lambda_ * self.w.data + self.eta * self.delta_w
		
		# Ensure that weights are within [0, self.wmax].
		self.w.data.clamp_(-self.wmax, self.wmax)	

	def forward(self, spike):
		return spike.float() @ self.w

class MixtureSynapses(nn.Module):
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, batch_size=1, nu_pre=1e-2, nu_post=1e-2, wmax=2.5, norm=78.0, init='zeros', factor=1):
		super().__init__()
		self.source = source
		self.target = target
		self.batch_size = batch_size

		# STDP
		if init == 'xavier':
			self.w = torch.rand(source.n, target.n)
			nn.init.xavier_uniform_(self.w)
			# self.w.data *= 2
		elif init == 'zeros':
			self.w = torch.zeros(source.n, target.n)
		self.lambda_ = 0.9#nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
		self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm
		self.stdp_lr = 1e-1

		# BP
		# self.w_bp = nn.Parameter(torch.rand(source.n, target.n))
		self.w_bp = nn.Parameter((torch.rand(source.n, target.n) - 0.5) * 2)


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
		post = self.nu_post * (self.source.x.view(self.source.n, self.batch_size) @ self.target.s.float().view(self.batch_size, self.target.n))
		# Pre-synaptic.
		pre = -self.nu_pre * (self.source.s.float().view(self.source.n, self.batch_size) @ self.target.x.view(self.batch_size, self.target.n))

		self.delta_w = self.stdp_lr * post + self.stdp_lr * pre
		self.w.data = self.lambda_ * self.w.data + self.eta * self.delta_w
		
		# Ensure that weights are within [0, self.wmax].
		self.w.data.clamp_(-self.wmax, self.wmax)	

	def forward(self, spike):
		return spike.float() @ (self.w * self.w_bp)
	

if __name__ == "__main__":
	from groups import LIFGroup
	n_x = LIFGroup(1,1,traces=True, refractory=2)
	n_y = LIFGroup(1,1,traces=True, refractory=2)
	w = STDPSynapses(n_x,n_y, nu_pre=1e-1, nu_post=1e-1)
	list_ = []
	
	input_x = torch.zeros((50,1,1))
	input_y = torch.zeros((50,1,1))
	input_x[25] = 5
	for delta_t in range(-25,25): # post
		input_y[25+delta_t] = 5

		for time in range(50):
			n_x(input_x[time])
			n_y(input_y[time])
			
			delta_w = w.nu_post * (w.source.x.view(w.source.n, 1) * w.target.s.float().view(1, w.target.n)) - w.nu_pre * (w.target.x.view(1, w.target.n) * w.source.s.float().view(w.source.n, 1))

			if delta_t >=0 and time == 25+delta_t:
				list_.append(delta_w[0][0].item())
			if delta_t < 0 and time == 25:
				list_.append(delta_w[0][0].item())
	print(list_)

	import matplotlib.pyplot as plt
	plt.plot(range(0,50), list_)
	plt.show()



			
		

