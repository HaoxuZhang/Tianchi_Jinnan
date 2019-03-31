import numpy as np
import os
import random

list=os.listdir('./jinnan2_round1_train_20190305/normal')

random.shuffle(list)
num=0
for i in range(len(list)):
    if num <400:
        #if list[i] not in exist_file:
        os.system('cp {} ./normal_aug/'.format('./jinnan2_round1_train_20190305/normal/'+list[i]))
        num+=1
    else:
        break
