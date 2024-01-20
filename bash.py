import os
list_ = range(20, 10020, 100)
with open('/home/jiashuncheng/code/MANN/plot/data/delay_a_hi50_de20_no_norm.txt', 'w') as a:
    pass
for delay in list_:
    print(delay, '/', 10000, ':')
    args = ["--experiment", "one_zero_ab_analyse", "--network", "SimpleMemoryNetwork_6_no_norm", "--lr", "1e-3", "--model_name", "A_hidden50_seed1_delay20_no_norm", "--n_hidden", "50", "--delay", f"{delay}", "--seed", "1", "--layer_A", "--mode", "analyse", "--repeat", "1", "--analyse_pre", "0", "--decision", "1"]
    args = " ".join(args)
    cmd = 'python /home/jiashuncheng/code/MANN/mysnn.py ' + args
    res = os.popen(cmd)
    output_str = res.readlines()[-2]  # 获得输出字符串
    with open('/home/jiashuncheng/code/MANN/plot/data/delay_a_hi50_de20_no_norm.txt', 'a+') as a:
        a.write('delay={}: '.format(delay) + output_str)


# list_ = range(20, 10020, 100)
# with open('/home/jiashuncheng/code/MANN/plot/data/delay_rnn_hi50_de20_no_norm.txt', 'w') as a:
#     pass
# for delay in list_:
#     print(delay, '/', 10000, ':')
#     args = ["--experiment", "one_zero_ab_analyse", "--network", "SimpleRNNNetwork_no_norm", "--lr", "1e-3", "--model_name", "RNN_hidden50_seed1_delay20_no_norm", "--n_hidden", "50", "--delay", f"{delay}", "--seed", "1", "--layer_A", "--mode", "analyse", "--repeat", "1", "--analyse_pre", "0", "--decision", "1"]
#     args = " ".join(args)
#     cmd = 'python /home/jiashuncheng/code/MANN/mysnn.py ' + args
#     res = os.popen(cmd)
#     output_str = res.readlines()[-2]  # 获得输出字符串
#     with open('/home/jiashuncheng/code/MANN/plot/data/delay_rnn_hi50_de20_no_norm.txt', 'a+') as a:
#         a.write('delay={}: '.format(delay) + output_str)