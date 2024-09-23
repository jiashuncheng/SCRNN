main_file = "/home/jiashuncheng/code/MANN/mysnn.py"

# -*- coding:utf-8 -*-
import subprocess,time,sys
import logging
from logging import handlers
from config import *

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
 
class Auto_Run():
    def __init__(self,sleep_time,cmd,args):
        self.log = Logger('log/le.log',level='debug')
        self.sleep_time = sleep_time
        self.cmd = cmd
        self.ext = (cmd[-3:]).lower()        #判断文件的后缀名，全部换成小写
        self.p = None             #self.p为subprocess.Popen()的返回值，初始化为None

        self.index = 0
        self.args = args

        self.run()                           #启动时先执行一次程序
 
        try:
            while 1:
                time.sleep(sleep_time * 30)  #休息1分钟，判断程序状态
                self.poll = self.p.poll()    #判断程序进程是否存在，None：表示程序正在运行 其他值：表示程序已退出
                if self.poll is None:
                    self.log.logger.info("第{}个程序运行正常".format(self.index))
                else:
                    self.index+=1
                    self.log.logger.info("未检测到程序运行状态，准备启动第{}个程序".format(self.index))
                    if self.index < len(list_):
                        self.run()
                    else:
                        break
        except KeyboardInterrupt as e:
            self.log.logger.info("检测到CTRL+C，准备退出程序!")

        self.log.logger.info('end.')
 
    def run(self):
        if self.ext == ".py":
            self.log.logger.info('start OK!')
            self.p = subprocess.Popen(['python','%s'%self.cmd] + self.args(self.index), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
        else:
            pass

TIME = 1                        #程序状态检测间隔（单位：分钟）
CMD = main_file                 #需要执行程序的绝对路径，支持jar 如：D:\\calc.exe 或者D:\\test.jar

list_ = []
# lam = ['0.90', '0.91', '0.92', '0.93', '0.94', '0.95', '0.96', '0.97', '0.98', '0.99', '1.00', '0.85', '0.86', '0.87', '0.88', '0.89']
# lam = ['0.90', '0.99', '1.00']
# lam = [0.02, 0.05, 0.1, 0.5, 1.0]
# eta = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '3.0', '4.0', '5.0']
# eta = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
# eta = ['0', '0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.007', '0.008', '0.009', '0.01', '0.0001', '0.0002', '0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008', '0.0009', '0.1', '1', '10']
# eta = ['0.0001', '0.0002', '0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008', '0.0009', '0.1', '1', '10']
# eta = ['100', '1000']
# eta = ['0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.007', '0.008', '0.009']
# eta = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9']
# eta = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

seed = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# eta = ['0.1', '0.2', '0.3']
# eta = ['0.4', '0.5', '0.6']
# eta = ['0.7', '0.8', '0.9']
# eta = ['0.01', '1.0']
# eta = ['0.02', '0.03', '0.04']
# eta = ['0.05', '0.06', '0.07']
# eta = ['0.001', '0.002', '0.004', '0.006', '0.008', '0.01', '0.02', '0.04', '0.06', '0.08', '0.1']
# eta = ['0.0001', '0.0002', '0.0004', '0.0006', '0.0008', '0.001', '0.002', '0.004', '0.006', '0.008', '0.01']
eta = ['0.02', '0.04', '0.06', '0.08', '0.1']
# eta = ['0.01']
