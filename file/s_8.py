from config import *

for x in lam[5:10]:
    for y in eta[:]:
        list_.append(['{:.2f}'.format(x), '{:.1f}'.format(y)])

app = Auto_Run(TIME,CMD) 