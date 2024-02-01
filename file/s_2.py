from config import *

for x in lam[3:6]:
    for y in eta[:]:
        list_.append(['{:.2f}'.format(x), '{:.2f}'.format(y)])

app = Auto_Run(TIME,CMD) 