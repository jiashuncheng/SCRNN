from config import *

for x in lam:
    for y in eta[6:]:
        list_.append(['{:.3f}'.format(x), '{:.1f}'.format(y)])

app = Auto_Run(TIME,CMD) 