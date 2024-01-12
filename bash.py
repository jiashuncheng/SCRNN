import os
list_ = range(100, 10100, 100)
with open('/home/jiashuncheng/code/MANN/plot/data/delay.txt', 'w') as a:
    pass
for delay in list_:
    print(delay, '/', 10000, ':')
    args = ["--experiment", "one_zero_ab_analyse", "--network", "SimpleMemoryNetwork_6", "--lr", "1e-3", "--model_name", "analyse_100_norm_without_b", "--n_hidden", "10", "--delay", f"{delay}", "--seed", "2", "--layer_A", "--mode", "analyse", "--repeat", "1", "--analyse_pre", "0", "--decision", "1"]
    args = " ".join(args)
    cmd = 'python /home/jiashuncheng/code/MANN/mysnn.py ' + args
    res = os.popen(cmd)
    output_str = res.readlines()[-2]  # 获得输出字符串
    with open('/home/jiashuncheng/code/MANN/plot/data/delay.txt', 'a+') as a:
        a.write('delay={}: '.format(delay) + output_str)