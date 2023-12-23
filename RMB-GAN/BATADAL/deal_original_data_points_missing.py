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
df = pd.read_csv(os.path.join(DATASETDIR, 'BATADAL_dataset03.csv'))

df=df.drop(['DATETIME'], axis=1)
df=df.drop(['ATT_FLAG'], axis=1)
df=df.drop(['S_PU11'], axis=1)
df=df.drop(['F_PU11'], axis=1)
df=df.drop(['S_PU9'], axis=1)
df=df.drop(['F_PU9'], axis=1)
df=df.drop(['S_PU3'], axis=1)
df=df.drop(['F_PU3'], axis=1)
df=df.drop(['S_PU1'], axis=1)
# Train/test split
train_df=df[0:6961]
test_df=df[6961:]

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
test_df = test_df.astype(np.float32)
np.save(os.path.join(opt.expPATH, 'dataTest_ground_truth.npy'), test_df, allow_pickle=False)

fill_train = np.random.uniform(size=(train_df.shape[0], train_df.shape[1]))
train_mask = torch.rand(train_df.shape)
train_mask=train_mask.numpy()
fill_train[fill_train<= 0.1] = -1
train_mask[fill_train!= -1] = 1
train_mask[fill_train== -1] = 0
train_df=train_df*train_mask+fill_train*(1-train_mask)

fill_test = np.random.uniform(size=(test_df.shape[0], test_df.shape[1]))
test_mask = torch.rand(test_df.shape)
test_mask=test_mask.numpy()
fill_test[fill_test<= 0.1] = -1
test_mask[fill_test!= -1] = 1
test_mask[fill_test== -1] = 0
test_df=test_df*test_mask+fill_test*(1-test_mask)
train_df=train_df.astype(np.float32)
test_df=test_df.astype(np.float32)

np.save(os.path.join(opt.expPATH,'dataTrain.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'dataTest.npy'), test_df, allow_pickle=False)

