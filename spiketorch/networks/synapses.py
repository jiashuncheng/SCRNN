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

		self.w = nn.Parameter(torch.rand(source.n, target.n))
		nn.init.xavier_uniform_(self.w)
		self.w.data *= 5

	def forward(self, spike):
		return spike @ self.w

class STDPSynapses(nn.Module):
	'''
	Specifies STDP-adapted synapses between two populations of neurons.
	'''
	def __init__(self, source, target, batch_size=1, nu_pre=1e-2, nu_post=1e-2, wmax=1.0, norm=78.0):
		super().__init__()
		self.source = source
		self.target = target
		self.batch_size = batch_size

		# self.w = nn.Parameter(torch.rand(source.n, target.n))
		self.w = torch.rand(source.n, target.n)
		nn.init.xavier_uniform_(self.w)
		self.w.data *= 5

		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmax = wmax
		self.norm = norm
		self.stdp_lr = 1e-3

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
		self.w.data += self.delta_w

		# Ensure that weights are within [0, self.wmax].
		self.w.data.clamp_(-self.wmax, self.wmax)	

	def forward(self, spike):
		return spike.float() @ self.w
	

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

	



			
		

