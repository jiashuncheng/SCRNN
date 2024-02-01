from config import *

for x in lam[:]:
    for y in eta[:5]:
        list_.append(['{}'.format(x), '{}'.format(y)])

# def args(i):
#     _args = ["--experiment", "one_zero_ab", "--network", "SimpleMemoryNetwork_10", "--lr", "1e-3", "--model_name", "scale_a_v1_lambda{}_eta{}_hid30".format(list_[i][0], list_[i][1]), "--n_hidden", "30", "--delay", "{:d}".format(100), "--seed", "5", "--layer_A", "--n_train", "20", "--eta", "{}".format(list_[i][1]), "--lambda_", "{}".format(list_[i][0]), "--gpu", "1"]
#     return _args

def args(i):
    _args = ["--experiment", "one_zero_ab", "--network", "SimpleMemoryNetwork_10", "--lr", "1e-3", "--model_name", "scale_a_10_v1_lambda{}_eta{}_hid30".format(list_[i][0], list_[i][1]), "--n_hidden", "30", "--delay", "{:d}".format(100), "--seed", "5", "--layer_A", "--n_train", "20", "--eta", "{}".format(list_[i][1]), "--lambda_", "{}".format(list_[i][0]), "--gpu", "0", "--prop_0_a", "0.8", "--prop_1_a", "0.4"]
    return _args

app = Auto_Run(TIME,CMD, args) 