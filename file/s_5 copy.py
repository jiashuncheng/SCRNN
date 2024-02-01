from config import *

for x in lam[11:]:
    for y in eta[6:10]:
        list_.append(['{:.2f}'.format(x), '{:.1f}'.format(y)])

app = Auto_Run(TIME,CMD) 