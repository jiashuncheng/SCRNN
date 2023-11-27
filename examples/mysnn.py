import json
import os, sys
from pathlib import Path
import timeit
import torch
import logging
import argparse
import numpy as np
import pickle
import pandas as pd
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

p = Path(__file__)
sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../spiketorch"))))

from network import Network, MemoryNetwork, SpikeMemoryNetwork, SpikeNetwork, STDPSpikeMemoryNetwork
from network import save_params, load_params
from datasets import get_MNIST, get_spike_abc, generate_spike_train, generate_memory_spike_train, get_one_zeros, get_abc

def log(info):
	logging.info(info)
	print(info)

parser = argparse.ArgumentParser(description='ETH (with LIF neurons) \
					SNN toy model simulation implemented with PyTorch.')

parser.add_argument('--experiment', type=str, default='abc', choices=['mnist', 'memory_mnist', 'one_zero', 'spike_abc', 'abc'])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_hidden', type=int, default=50)
parser.add_argument('--n_train', type=int, default=100)
parser.add_argument('--n_test', type=int, default=1)
parser.add_argument('--nu_pre', type=float, default=1e-2)
parser.add_argument('--nu_post', type=float, default=1e-2)
parser.add_argument('--rest', type=float, default=0.)
parser.add_argument('--reset', type=float, default=-0.1)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--voltage_decay', type=float, default=1)
parser.add_argument('--refractory', type=int, default=1)
parser.add_argument('--trace_tc', type=int, default=5e-2)
parser.add_argument('--wmax', type=float, default=10.0)
parser.add_argument('--dt', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--model_name', type=str, default='eth')
parser.add_argument('--network', type=str, default='MemoryNetwork', choices=['MemoryNetwork', 'SpikeMemoryNetwork', 'Network', 'SpikeNetwork', 'STDPSpikeMemoryNetwork'])

# one_zero task
parser.add_argument('--sample', type=int, default=11)
parser.add_argument('--delay', type=int, default=4)
parser.add_argument('--decision', type=int, default=2)

# Place parsed arguments in local scope.
args = parser.parse_args()

# Set random number generator.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
torch.set_printoptions(threshold=sys.maxsize, linewidth=100, edgeitems=10)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.gpu is not None:
	assert torch.cuda.is_available()
	torch.cuda.manual_seed(args.seed)
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

logs_path = os.path.join(p.parent.parent, 'results',args.model_name, 'logs')
params_path = os.path.join(p.parent.parent, 'results',args.model_name, 'params')
results_path = os.path.join(p.parent.parent, 'results',args.model_name, 'results')
assign_path = os.path.join(p.parent.parent, 'results',args.model_name, 'assignments')
perform_path = os.path.join(p.parent.parent, 'results',args.model_name, 'performances')
data_path = os.path.join(p.parent.parent, 'data', args.experiment)

# Build filename from command-line arguments.
fname = '_'.join([ str(args.n_hidden), str(args.n_train), str(args.seed), str(args.wmax)])
for path in [logs_path, data_path, params_path, assign_path, results_path, perform_path]:
	if not os.path.isdir(path):
		os.makedirs(path)
  
# Init network
network = {'MemoryNetwork': MemoryNetwork, 
           'SpikeMemoryNetwork': SpikeMemoryNetwork,
           'Network': Network,
           'SpikeNetwork': SpikeNetwork,
           'STDPSpikeMemoryNetwork': STDPSpikeMemoryNetwork}

if args.experiment == 'abc' and 'Spike' not in args.network:
	train_data, test_data = get_abc(args, data_path=data_path,device=device)
	model = network[args.network](args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
 
elif args.experiment == 'abc' and 'Spike' in args.network:
	train_data, test_data = get_abc(args, data_path=data_path,device=device)
	model = network[args.network](args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# loss_fun = torch.nn.MSELoss()
 
elif args.experiment == 'spike_abc' and 'Spike' in args.network:
	train_data, test_data = get_spike_abc(args, data_path=data_path,device=device)
	model = network[args.network](args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# loss_fun = torch.nn.MSELoss()
 
elif args.experiment == 'spike_abc' and 'Spike' not in args.network:
	train_data, test_data = get_spike_abc(args, data_path=data_path,device=device)
	model = network[args.network](args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# loss_fun = torch.nn.MSELoss()

elif args.experiment == 'one_zero' and 'Spike' not in args.network:
	train_data, test_data = get_one_zeros(args, data_path=data_path, device=device)
	model = network[args.network](args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	action_fun = torch.nn.Sigmoid()
	loss = torch.nn.BCELoss()
	loss_fun = lambda prediction, target : loss(action_fun(prediction), target.float())
 
elif args.experiment == 'one_zero' and 'Spike' in args.network:
	train_data, test_data = get_one_zeros(args, data_path=data_path, device=device)
	model = network[args.network](args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# action_fun = torch.nn.Sigmoid()
	# loss = torch.nn.BCELoss()
	# loss_fun = lambda prediction, target : loss(action_fun(prediction), target.float())

# Log argument values.
print('Optional argument values:')
for key, value in vars(args).items():
	print('--', key, ':', value)
with open(os.path.join(params_path, '_'.join(['params', fname]) + '.json'), 'w') as f:
	params = json.dumps(vars(args), ensure_ascii=False, indent=2)
	f.write(params)
	f.close()
 
# Set logging configuration.
logging.basicConfig(format='%(message)s', 
					filename=os.path.join(logs_path, '%s.log' % fname),
					level=logging.DEBUG,
					filemode='w')

def train():
	# Run model simulation.
	start = timeit.default_timer()
	for i in range(args.n_train):
		total_correct = 0
		best_accuracy = -1
		correct = 0
		with tqdm(total=len(train_data), ncols=100) as _tqdm:
			_tqdm.set_description('epoch: {}/{}'.format(i+1, args.n_train))
			for idx, (image, target) in enumerate(train_data):
				x_in, target = image.permute(1,0,2).to(device), target.to(device)
				y_out = model(args.mode, x_in, args.time)
				predictions = torch.mean(y_out, dim=0)
				loss = loss_fun(predictions, target)
				# save_params(params_path, model, fname, 'model')
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				# print(model.layer_h.w.max(), model.layer_h.w.mean())
				# If correct, increment counter variable.
				correct = (predictions.argmax(1) == target.argmax(1)).float().sum()
				correct = correct / args.batch_size
				total_correct += (predictions.argmax(1) == target.argmax(1)).float().sum()
				model.reset()
				_tqdm.set_postfix(loss='{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(loss, model.layer_a.w.max(), model.layer_a.w.mean(), model.layer_h.w.max()))
				_tqdm.update(1)
		total_correct = total_correct / (args.batch_size * len(train_data)) 
		if total_correct > best_accuracy:
			best_accuracy = correct
			save_params(params_path, model, fname, 'model')

		log('Training progress (%d/%d): Finish - Elapsed time: %.4f h\n' % (i+1, args.n_train, (timeit.default_timer() - start)/3600))
		log('Current training total accuracy: %.4f\n' % (total_correct))

def train_MNIST():
	train_data, test_data = get_MNIST(args, device=device)
	# Keep track of correct classifications for performance monitoring.
	for i in range(args.n_train):
		total_correct = 0
		best_accuracy = -1
		correct = 0
		length = len(train_data)
		with tqdm(total=length, ncols=80) as _tqdm:
			_tqdm.set_description('epoch: {}/{}'.format(i+1, args.n_train))
			for idx, (image, target) in enumerate(train_data):
				image, target = image.to(device), target.to(device)
				# Encode current input example as Poisson spike trains.
				x_in = generate_spike_train(image, int(args.time / args.dt))
				# Run network on Poisson-encoded image data.
				y_out = network(args.mode, x_in, args.time)
				predictions = torch.mean(y_out, dim=0)
				target_onehot = F.one_hot(target.long(), num_classes=args.n_output).float()[:,0,:]
				loss = loss_fun(predictions, target_onehot)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				# If correct, increment counter variable.
				correct = (predictions.argmax(1, keepdim=True) == target).float().sum()
				correct = correct / args.batch_size
				total_correct += (predictions.argmax(1, keepdim=True) == target).float().sum()
				network.reset()
				_tqdm.set_postfix(loss='{:.4f}'.format(loss))
				_tqdm.update(1)
		total_correct = total_correct / (args.batch_size * len(train_data)) 
		if total_correct > best_accuracy:
			best_accuracy = correct
			save_params(params_path, network, fname, 'model')

		log('Training progress (%d/%d): Finish - Elapsed time: %.4f\n' % (i+1, args.n_train, (timeit.default_timer() - start)/3600))
		log('Current training total accuracy: %.4f\n' % (total_correct))

def test_MNIST():
	train_data, test_data = get_MNIST(args, device=device)
	with torch.no_grad():
		# network = load_params(params_path, fname, 'model')
		Accuracy = 0
		for idx, (image, target) in enumerate(test_data):
			image, target = image.to(device), target.to(device)
			# Encode current input example as Poisson spike trains.
			x_in = generate_spike_train(image, int(args.time / args.dt))
			# Run network on Poisson-encoded image data.
			y_out = network(args.mode, x_in, args.time)
			predictions = torch.mean(y_out, dim=0)
			total_correct = (predictions.argmax(1, keepdim=True) == target[0]).float().sum()
			correct = total_correct / args.batch_size
			Accuracy += correct

		Accuracy = Accuracy / (idx + 1)

		log('Test progress: (%d / %d) - Elapsed time: %.4f\n' % (args.n_test, args.n_test, (timeit.default_timer() - start)/3600))
		log('Test accuracy: %.4f\n' % (Accuracy))

		# Save out network parameters.
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
		test_MNIST()