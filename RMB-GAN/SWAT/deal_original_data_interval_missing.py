import argparse
import numpy as np
import time
import random
import os
import sklearn.model_selection as skl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
parser = argparse.ArgumentParser()


parser.add_argument("--expPATH", type=str, default=os.path.expanduser(''),
                    help="Experiment path")

opt = parser.parse_args()
DATASETDIR = os.path.expanduser('')
df = pd.read_csv(os.path.join(DATASETDIR, 'SWaT_Dataset_Normal_v1.csv'))
df=df.drop([' Timestamp'], axis=1)
df=df.drop(['n'], axis=1)
df=df.drop(['P601'], axis=1)
df=df.drop(['P603'], axis=1)
pd.set_option('display.max_columns', None)
# Train/test split
train_df=df[0:396000]
test_df=df[396000:]

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
test_df = test_df.astype(np.float32)
np.save(os.path.join(opt.expPATH, 'dataTest_ground_truth.npy'), test_df, allow_pickle=False)

for i in range(1,6):
    train_df[i:train_df.shape[0]:6,1:8:3]=-1.0
    test_df[i:test_df.shape[0]:6,1:8:3]=-1.0
for i in range(1,11):
    train_df[i:train_df.shape[0]:11,10:23:3]=-1.0
    test_df[i:test_df.shape[0]:11,10:23:3]=-1.0
for i in range(1,16):
    train_df[i:train_df.shape[0]:16,25:47:3]=-1.0
    test_df[i:test_df.shape[0]:16,25:47:3]=-1.0

np.save(os.path.join(opt.expPATH,'dataTrain.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'dataTest.npy'), test_df, allow_pickle=False)

