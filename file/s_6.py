from config import *

for x in lam[:]:
    for y in eta[:]:
        list_.append(['{:.2f}'.format(x), '{:.3f}'.format(y)])

def args(i):
    _args = ["--experiment", "one_zero_ab", "--network", "SimpleMemoryNetwork_6", "--lr", "1e-3", "--model_name", "lambda{}_eta{}_hid30".format(list_[i][0], list_[i][1]), "--n_hidden", "30", "--delay", "{:d}".format(100), "--seed", "2", "--layer_A", "--n_train", "20", "--eta", "{}".format(list_[i][1]), "--lambda_", "{}".format(list_[i][0]), "--gpu", "1"]
    return _args

app = Auto_Run(TIME,CMD, args) 