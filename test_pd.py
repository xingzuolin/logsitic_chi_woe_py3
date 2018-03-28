#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 16:04
# @Author  : Jun
# @File    : test_pd.py
import os
from binning import find_best_bin
import pandas as pd

src = r'E:/dev_code/logistic_data/modeling/data/'
seg = 1

unit_weight = 'UNIT_WEIGHT'
doll_weight = 'DOLL_WEIGHT'
target_var = 'IS_COLLECTION_BAD'

prefix = 'collection'

if not os.path.isdir(src+'s'+str(seg)):
    os.mkdir(src+'s'+str(seg))
if not os.path.isdir(src+'s'+str(seg)+'/woe_files'):
    os.mkdir(src+'s'+str(seg)+'/woe_files')

tgt = src+'s'+str(seg)

h5 = pd.HDFStore(tgt + '/s' + str(seg) + '_dev_oot.h5', 'r')
dev = h5['dev']
oot = h5['oot']
h5.close()

var_name ='MP_FPD_RATE_AMT_3M'
data = dev[[var_name, target_var]]

find_best_bin(data=data, subset=[0, 1], y=target_var,var_name=var_name,groups=5,rate=0.05)