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
df = pd.read_csv(os.path.join(DATASETDIR, 'EpicLog_Scenario 6_19_Oct_2018_16_06.csv'))
df=df.drop(['Timestamp'], axis=1)

# Train/test split
train_df=df[0:683]
test_df=df[683:]

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
test_df = test_df.astype(np.float32)
np.save(os.path.join(opt.expPATH, 'dataTest_ground_truth.npy'), test_df, allow_pickle=False)

x=0
for i in range(7):
    idx=random.randint(10,60)
    idx=idx+x
    train_df[idx:idx+30]=-1.0
    x=idx+30
x=0
for i in range(2):
    idx = random.randint(40, 90)
    idx = idx + x
    test_df[idx:idx + 30] = -1.0
    x = idx + 30


np.save(os.path.join(opt.expPATH,'dataTrain.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'dataTest.npy'), test_df, allow_pickle=False)

