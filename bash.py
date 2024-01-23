import os
import time

seed = 5
list_ = [30]+list(range(100,10100,100))


# with open(f'/home/jiashuncheng/code/MANN/plot/data/delay_a_hi10_de30_new_{seed}.txt', 'w') as a:
#     pass
# for delay in list_:
#     print(delay, '/', 200, ':')
#     args = ["--experiment", "one_zero_ab_analyse", "--network", "SimpleMemoryNetwork_6", "--lr", "1e-3", "--model_name", f"A_hidden10_seed{seed}_delay30_new_1", "--n_hidden", "10", "--delay", f"{delay}", "--seed", f"{seed}", "--layer_A", "--mode", "analyse", "--repeat", "1", "--analyse_pre", "0", "--decision", "1"]
#     args = " ".join(args)
#     cmd = 'python /home/jiashuncheng/code/MANN/mysnn.py ' + args
#     res = os.popen(cmd)
#     output_str = res.readlines()[-2]  # 获得输出字符串
#     print(output_str)
#     with open(f'/home/jiashuncheng/code/MANN/plot/data/delay_a_hi10_de30_new_{seed}.txt', 'a+') as a:
#         a.write('delay={}: '.format(delay) + output_str)

with open(f'/home/jiashuncheng/code/MANN/plot/data/delay_rnn_hi21_de30_no_norm_new_{seed}.txt', 'w') as a:
    pass
for delay in list_:
    print(delay, '/', 1000, ':')
    args = ["--experiment", "one_zero_ab_analyse", "--network", "SimpleRNNNetwork_no_norm", "--lr", "1e-3", "--model_name", f"RNN_hidden21_seed{seed}_delay30_no_norm_new_1", "--n_hidden", "21", "--delay", f"{delay}", "--seed", f"{seed}", "--layer_A", "--mode", "analyse", "--repeat", "1", "--analyse_pre", "0", "--decision", "1"]
    args = " ".join(args)
    cmd = 'python /home/jiashuncheng/code/MANN/mysnn.py ' + args
    res = os.popen(cmd)
    output_str = res.readlines()[-2]  # 获得输出字符串
    print(output_str)
    # break
    with open(f'/home/jiashuncheng/code/MANN/plot/data/delay_rnn_hi21_de30_no_norm_new_{seed}.txt', 'a+') as a:
        a.write('delay={}: '.format(delay) + output_str)