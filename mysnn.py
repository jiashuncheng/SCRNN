import json
import os, sys
from pathlib import Path
import timeit
import torch
import logging
import argparse
import numpy as np
import pickle
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch
import random

p = Path(__file__)
sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "networks"))))

from network import *
from datasets import *

def log(info):
	logging.info(info)
	print(info)

parser = argparse.ArgumentParser(description='ETH (with LIF neurons) \
					SNN toy model simulation implemented with PyTorch.')

parser.add_argument('--experiment', type=str, default='abc', help='Dataset enviroment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_hidden', type=int, default=400)
parser.add_argument('--n_train', type=int, default=100)
parser.add_argument('--n_test', type=int, default=1)
parser.add_argument('--nu_pre', type=float, default=1e-2)
parser.add_argument('--nu_post', type=float, default=1e-2)
parser.add_argument('--rest', type=float, default=0.)
parser.add_argument('--reset', type=float, default=0.)
parser.add_argument('--threshold', type=float, default=0.2)
parser.add_argument('--voltage_decay', type=float, default=1)
parser.add_argument('--refractory', type=int, default=2)
parser.add_argument('--trace_tc', type=int, default=5e-2)
parser.add_argument('--wmax', type=float, default=10.0)
parser.add_argument('--dt', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--model_name', type=str, default='eth', help='Name of saved training model')
parser.add_argument('--network', type=str, default='MemoryNetwork', help='Network architecture')
parser.add_argument('--layer_A', action='store_true')
parser.add_argument('--n_trials', type=int, help='Number of dataset samples.', default=1000)

# one_zero task
parser.add_argument('--analyse_pre', type=int, default=0)
parser.add_argument('--sample', type=int, default=20)
parser.add_argument('--delay', type=int, default=100)
parser.add_argument('--decision', type=int, default=1)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--prop', type=float, default=0.5)
parser.add_argument('--prop_0_a', type=float, default=0.8)
parser.add_argument('--prop_1_a', type=float, default=0.4)
parser.add_argument('--mu', type=float, default=0.)
parser.add_argument('--sigma', type=float, default=0.6)

# analyse
parser.add_argument('--store_h_state', action='store_true')
parser.add_argument('--name_of_saved_file', type=str, help='filename of store_h_state')
parser.add_argument('--store_A_state', action='store_true')
parser.add_argument('--cut_atn_to_rsc', action='store_true')
parser.add_argument('--cut_acc_to_rsc', action='store_true')
parser.add_argument('--lambda_', type=float, default=0.9)
parser.add_argument('--eta0', type=float, default=0)
parser.add_argument('--u', type=float, default=0.5)
parser.add_argument('--kd', type=float, default=0.5)
parser.add_argument('--r0', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--eta', type=float, default=-1.)
parser.add_argument('--wg', type=float, default=0.1)
parser.add_argument('--rc', type=float, default=-1.)


# Place parsed arguments in local scope.
args = parser.parse_args()
if 'analyse' in args.experiment:
	args.mode = 'analyse'

# Set random number generator.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
torch.set_printoptions(threshold=sys.maxsize, linewidth=100, edgeitems=10)
if args.gpu is not None:
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	assert torch.cuda.is_available()
	torch.cuda.manual_seed(args.seed)
	device = torch.device("cuda:0")
	torch.set_default_dtype(torch.float32)
	# torch.set_default_device(device)
else:
	device = torch.device("cpu")

logs_path = os.path.join(p.parent, 'results', args.model_name, 'logs')
params_path = os.path.join(p.parent, 'results', args.model_name, 'params')
model_path = os.path.join(p.parent, 'results', args.model_name, 'model')
data_path = os.path.join(p.parent, 'data', args.experiment)

# Build filename from command-line arguments.
fname = '_'.join([str(args.n_hidden), str(args.n_train), str(args.seed), str(args.experiment), str(args.cut_acc_to_rsc), str(args.cut_atn_to_rsc), str(args.layer_A), str(args.eta0), str(args.lambda_), str(args.delay)])
for path in [logs_path, data_path, params_path, model_path]:
	if not os.path.isdir(path):
		os.makedirs(path)
  
# Init network
if args.experiment == 'abc' and 'Spike' not in args.network:
	train_data, test_data = get_abc(args, data_path=data_path,device=device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
 
elif args.experiment == 'abc' and 'Spike' in args.network:
	train_data, test_data = get_abc(args, data_path=data_path,device=device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# loss_fun = torch.nn.MSELoss()
 
elif args.experiment == 'spike_abc' and 'Spike' in args.network:
	train_data, test_data = get_spike_abc(args, data_path=data_path,device=device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# loss_fun = torch.nn.MSELoss()
 
elif args.experiment == 'spike_abc' and 'Spike' not in args.network:
	train_data, test_data = get_spike_abc(args, data_path=data_path,device=device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# loss_fun = torch.nn.MSELoss()

elif args.experiment == 'one_zero' and 'Spike' in args.network:
	train_data, test_data = get_one_zeros(args, data_path=data_path, device=device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()
	# action_fun = torch.nn.Sigmoid()
	# loss = torch.nn.BCELoss()
	# loss_fun = lambda prediction, target : loss(action_fun(prediction), target.float())

elif args.experiment == 'one_zero' and 'Spike' not in args.network:
	train_data, test_data = get_one_zeros(args, data_path=data_path, device=device)
	# model = network[args.network](args, device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()

elif args.experiment == 'one_zero_ab' and 'Spike' not in args.network:
	train_data, test_data = get_one_zeros_ab(args, data_path=data_path, device=device)
	if 'Memory' in args.network:
		args.n_input = 2
	model = eval(args.network)(args, device)
	'''
	first_recurrent_params = list(model.layer_ar.parameters()) + \
		list(model.layer_ra.parameters()) + \
		list(model.layer_a.parameters()) + \
		list(model.layer_r.parameters())
	other_params = list(model.layer_sr.parameters()) + \
		list(model.layer_ma.parameters()) + \
		list(model.layer_ao.parameters()) + \
		list(model.layer_ro.parameters()) + \
		list(model.layer_y.parameters())
	optimizer = optim.Adam([{'params': first_recurrent_params, 'weight_decay': 0.01},
				{'params': other_params, 'weight_decay': 0}], lr=args.lr)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	'''
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
	# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()

elif args.experiment == 'one_zero_ab_analyse' and 'Spike' not in args.network:
	train_data, test_data = get_one_zeros_ab_analyse(args, data_path=data_path, device=device)
	if 'Memory' in args.network:
		args.n_input = 2
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()

elif args.experiment == 'one_zero_vanillia' and 'Spike' in args.network:
	train_data, test_data = get_one_zeros_vanillia(args, data_path=data_path, device=device)
	model = eval(args.network)(args, device)
	# optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# loss_fun = torch.nn.MSELoss()
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()

elif args.experiment == 'one_zero_vanillia' and 'Spike' not in args.network:
	train_data, test_data = get_one_zeros_vanillia(args, data_path=data_path, device=device)
	model = eval(args.network)(args, device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	loss_fun = lambda prediction, target : torch.sum(-target * F.log_softmax(prediction, -1), -1).mean()


model = model.to(device)

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
		correct = 0
		with tqdm(total=len(train_data), ncols=100) as _tqdm:
			_tqdm.set_description('epoch: {}/{}'.format(i+1, args.n_train))
			for idx, (image, target) in enumerate(train_data):
				x_in, target = image.permute(1,0,2).to(device), target.to(device)
				y_out = model('train', x_in, args.time)
				predictions = torch.mean(y_out, dim=0)
				loss = loss_fun(predictions, target.float())
				assert loss is not torch.nan, "Loss is nan!"
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				correct = (predictions.argmax(1) == target.argmax(1)).float().sum()
				correct = correct / args.batch_size
				total_correct += (predictions.argmax(1) == target.argmax(1)).float().sum()
				model.reset()
				_tqdm.set_postfix(loss='{:.4f}, model={}'.format(loss, args.network))
				_tqdm.update(1)
			if 'scheduler' in globals():
				print('sc')
				scheduler.step()
		total_correct = total_correct / (args.batch_size * len(train_data))

		log('Training progress (%d/%d): Finish - Elapsed time: %.4f h' % (i+1, args.n_train, (timeit.default_timer() - start)/3600))
		log('Current training total accuracy: %.4f' % (total_correct))
		test(i, model)
		print('Save model ...')
		torch.save(model.state_dict(), model_path + '/save.pt')

counter = 0
def test(i, model):
	model.eval()
	with torch.no_grad():
		total_correct = 0
		correct = 0
		with tqdm(total=len(test_data), ncols=100) as _tqdm:
			_tqdm.set_description('epoch: {}/{}'.format(i+1, args.n_train))
			for idx, (image, target) in enumerate(test_data):
				x_in, target = image.permute(1,0,2).to(device), target.to(device)
				y_out = model('test', x_in, args.time)
				predictions = torch.mean(y_out, dim=0)
				correct = (predictions.argmax(1) == target.argmax(1)).float().sum()
				correct = correct / args.batch_size
				total_correct += (predictions.argmax(1) == target.argmax(1)).float().sum()
				model.reset()
				_tqdm.update(1)
		total_correct = total_correct / (args.batch_size * len(test_data))

		log('Test accuracy: %.4f\n' % (total_correct))

		global counter
		if total_correct >= 0.99:
			counter += 1
		if counter >= 10:
			sys.exit()

def analyse():
	model.load_state_dict(torch.load(model_path + '/save.pt'))
	# model.load_state_dict(torch.load('/home/jiashuncheng/code2/MANN2/results_20240301/a_20240229_seed1/model/save.pt'))
	if args.mode == "analyse" and args.cut_acc_to_rsc:
		model.layer_ao.w.data *= 0. # 切断ACC
	if args.mode == "analyse" and args.cut_atn_to_rsc:
		model.layer_ro.w.data *= 0. # 切断ATN
	model.eval()
	with torch.no_grad():
		total_correct = 0
		correct = 0
		records = []
		with tqdm(total=len(test_data), ncols=100) as _tqdm:
			for idx, (image, target) in enumerate(test_data):
				x_in, target = image.permute(1,0,2).to(device), target.to(device)
				x_in_origin = x_in.clone()
				# x_in = x_in + torch.normal(mean=args.mu, std=args.sigma, size=x_in.shape).to(device)
				y_out = model('analyse', x_in, args.time)
				predictions = torch.mean(y_out, dim=0)
				correct = (predictions.argmax(1) == target.argmax(1)).float().sum()
				# correct = correct / args.batch_size
				total_correct += (predictions.argmax(1) == target.argmax(1)).float().sum()
				if args.mode == "analyse" and args.store_h_state:
					# with open('/home/jiashuncheng/code/MANN/plot/data/fixed_eta_h_seed25.pkl', 'wb') as a:
					#LINK - /home/jiashuncheng/code/MANN/plot/data/fixed_eta_h_seed25_20240325.pkl
					# with open('/home/jiashuncheng/code2/MANN2/plot/data/test_20240828_{}.pkl'.format(args.sample), 'wb') as a: # 用于补充1~20的实验测试
					records.append({"RSC": model.neuron_o.h, 
								"ACC": model.neuron_a.h, 
								"ATN": model.neuron_r.h,
								"g": model.gs,
								"acc": correct.cpu().numpy()})
					model.reset()
				_tqdm.update(1)
		total_correct = total_correct / (args.batch_size * len(test_data))
		with open('/home/jiashuncheng/liuchenghao/MANN/MANN2/plot/data2/test_eta/{}.pkl'.format(args.name_of_saved_file), 'wb') as a:
			pickle.dump(records, a)

		log('Analyse accuracy: %.4f\n' % (total_correct))

def train_MNIST():
	train_data, test_data = get_MNIST(args, device=device)
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

if __name__ == "__main__":
	if args.mode == 'train':
		train()
	elif args.mode == 'analyse':
		analyse()
