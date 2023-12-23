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
df = pd.read_csv(os.path.join(DATASETDIR, 'WADI_attackdata_labelled.csv')).drop('Unnamed: 0', axis=1)

df=df.drop(['Row'], axis=1)
df=df.drop(['Date'], axis=1)
df=df.drop(['Time'], axis=1)
df=df.drop(['attack'], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_LS_001_AL"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_LS_002_AL"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_002_STATUS"], axis=1)

df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_003_STATUS"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_004_STATUS"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_001_AL"], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_002_AL'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_004_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_005_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_009_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_004_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_101_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_201_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_301_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_401_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_501_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_601_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_001_PV'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_002_PV'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_LS_001_AL'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_MV_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_MV_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_MV_003_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_003_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_004_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\PLANT_START_STOP_LOG'], axis=1)
df=df[6620:56620]

# Train/test split
train_df=df[0:41000]
test_df=df[41000:]
sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
test_df = test_df.astype(np.float32)
np.save(os.path.join(opt.expPATH, 'dataTest_ground_truth.npy'), test_df, allow_pickle=False)

x=0
for i in range(410):
    idx=random.randint(10,60)
    idx=idx+x
    train_df[idx:idx+30]=-1.0
    x=idx+30
x=0
for i in range(90):
    idx = random.randint(10, 60)
    idx = idx + x
    test_df[idx:idx + 30] = -1.0
    x = idx + 30

np.save(os.path.join(opt.expPATH,'dataTrain.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'dataTest.npy'), test_df, allow_pickle=False)

