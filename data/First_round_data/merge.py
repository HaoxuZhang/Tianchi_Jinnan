import numpy as np
import os
import random

list=os.listdir('./jinnan2_round1_train_20190305/restricted')
for i in range(len(list)):
    os.system('cp {} ./train_val/'.format('./jinnan2_round1_train_20190305/restricted/'+list[i]))

list=os.listdir('./normal_aug')
for i in range(len(list)):
    os.system('cp {} ./train_val/'.format('./normal_aug/'+list[i]))