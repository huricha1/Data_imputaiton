import argparse
import os
from gap_missing import generatorModel_x,mask_data
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from sklearn.linear_model import LinearRegression
parser = argparse.ArgumentParser()


parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--window_size", type=int, default=30, help="size of the window")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--PATH", type=str, default=os.path.expanduser(''),
                    help="Training status")
opt = parser.parse_args()
print(opt)


# Create the experiments path
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(opt.PATH)

testData = np.load('', allow_pickle=False)
testData_ground_truth = np.load('', allow_pickle=False)

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")

#####################################
#### Load model and optimizer #######
#####################################

# Loading the checkpoint
checkpoint = torch.load(os.path.join(opt.PATH, f'model_epoch_.pth'))

# Load models
generatorModel_x.load_state_dict(checkpoint['Generator_state_dict'])

generatorModel_x.eval()

#######################################################
#### Load real data and restore data #######
#######################################################

# Load real data
num=testData.shape[0]
num_fake_samples = num
# Generate a batch of samples
gen_samples=np.zeros([num, 6])
n_batches = int(num_fake_samples / opt.window_size)
for i in range(n_batches):
    test = testData[i * opt.window_size:(i + 1) * opt.window_size, :].copy()
    test = torch.from_numpy(test).to(device)
    test = test.view(1, opt.window_size, test.shape[1])
    real_mask = torch.rand(test.shape)
    real_mask[test != -1] = 1
    real_mask[test == -1] = 0
    real_mask = Variable(real_mask).to(device)
    # Sample noise as generator input
    z = torch.randn(real_mask.shape, device=device)
    impute = mask_data(test, real_mask, z)
    impu_data1 = generatorModel_x(impute,0)
    impute_data = mask_data(test, real_mask, impu_data1)

    gen_samples[i * opt.window_size:(i + 1) * opt.window_size, :] = impute_data.cpu().data.numpy()

    # Check to see if there is any nan
    assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.window_size:], 0)

# Trasnform Object array to float
gen_samples = gen_samples.astype(np.float32)
# ave synthetic data
np.save(os.path.join(opt.PATH, 'syntheticlabel.npy', gen_samples, allow_pickle=False))

gen = np.load(os.path.join(opt.PATH, 'syntheticlabel.npy'), allow_pickle=False)
#######################################################
#### test regression#######
#######################################################

X_train_r = gen[0:300].copy()
X_test_r = testData_ground_truth[300:].copy()

a = X_train_r[:, :2].copy()
b = X_train_r[:, 3:].copy()
x_train_lr = np.concatenate((a, b), axis=1)
y_train_lr = X_train_r[:, 2].copy()

a = X_test_r[:, :2].copy()
b = X_test_r[:, 3:].copy()
x_test_lr = np.concatenate((a, b), axis=1)
y_test_lr = X_test_r[:, 2].copy()
linear_regressor = LinearRegression()
linear_regressor.fit(x_train_lr, y_train_lr)
y_pre = linear_regressor.predict(x_test_lr)
loss = nn.MSELoss(reduction='sum')
l = loss(torch.from_numpy(y_pre), torch.from_numpy(y_test_lr))




