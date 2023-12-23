import argparse
import os

import numpy as np
import time

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from module import (Generator_x, Encoder_x,Discriminator_x,Discriminator_i)


parser = argparse.ArgumentParser()


parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--window_size", type=int, default=30, help="size of the window")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")


parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--save_model", type=int, default=, help="model save")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--PATH", type=str, default=os.path.expanduser(''),
                    help="Training status")
opt = parser.parse_args()
print(opt)


# Create the experiments path
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(opt.PATH)


########## Dataset Processing ###########

class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = len(data)
        self.featureSize = data[0].shape[1]

    def return_data(self):
        return self.data
        # return np.clip(self.data, 0, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            pass

        return torch.from_numpy(sample)


trainData = np.load('', allow_pickle=False)
testData = np.load('', allow_pickle=False)
testData_ground_truth = np.load('', allow_pickle=False)

samples_list = list()
for i in range(0,trainData.shape[0]-opt.window_size+1):
    samples_list.append(trainData[i:i+opt.window_size])

# Train data loader
dataset_train_object = Dataset(data=samples_list, transform=False)

samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")


criterion = nn.BCELoss()

def mse_loss(x_output, y_target):
    loss=nn.MSELoss(reduction='sum')
    l=loss(x_output,y_target)
    return l

def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau

def impute_data(data, mask, tau):
    return (1-mask) * data + mask * tau

def weights_init(m):
    """
    Custom weight initialization.
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.LSTMCell:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Initialize generator and discriminator
generatorModel_x = Generator_x()
discriminatorModel_i = Discriminator_i()
discriminatorModel_x = Discriminator_x()
EncoderModel_x=Encoder_x()


if torch.cuda.device_count() > 1 and opt.multiplegpu:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  generatorModel_x = nn.DataParallel(generatorModel_x, list(range(opt.num_gpu)))
  discriminatorModel_i = nn.DataParallel(discriminatorModel_i, list(range(opt.num_gpu)))
  discriminatorModel_x = nn.DataParallel(discriminatorModel_x, list(range(opt.num_gpu)))
  EncoderModel_x = nn.DataParallel(EncoderModel_x, list(range(opt.num_gpu)))

# Put models on the allocated device
generatorModel_x.to(device)
discriminatorModel_i.to(device)
discriminatorModel_x.to(device)
EncoderModel_x.to(device)

# Weight initialization
generatorModel_x.apply(weights_init)
discriminatorModel_i.apply(weights_init)
discriminatorModel_x.apply(weights_init)
EncoderModel_x.apply(weights_init)

# Optimizers
optimizer_Gx = torch.optim.Adam(generatorModel_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

optimizer_Di = torch.optim.Adam(discriminatorModel_i.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_Dx = torch.optim.Adam(discriminatorModel_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_Ex = torch.optim.Adam(EncoderModel_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

# Define cuda Tensor
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

if opt.training:
    gen_iterations = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i_batch, samples in enumerate(dataloader_train):
            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]*samples.shape[1]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]*samples.shape[1]).fill_(0.0), requires_grad=False)
            # Configure input
            real_samples = Variable(samples.type(Tensor)).to(device)
            real_mask = torch.rand(real_samples.shape)
            real_mask[real_samples!= -1] = 1
            real_mask[real_samples== -1] = 0
            real_mask = Variable(real_mask).to(device)
            # Sample noise as generator input
            z = torch.randn(real_samples.shape,device=device)
            real_mask_data=mask_data(real_samples,real_mask,z)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # reset gradients of discriminator
            optimizer_Dx.zero_grad()

            for p in discriminatorModel_i.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in discriminatorModel_x.parameters():  # reset requires_grad
                p.requires_grad = True

            fake_samples = generatorModel_x(real_mask_data,1)
            fake_mask_samples = mask_data(real_samples, real_mask, fake_samples)

            z_real = EncoderModel_x(fake_mask_samples)
            z_mask_real = mask_data(z_real, real_mask, z)
            out_realx = discriminatorModel_x(fake_mask_samples.detach(), z_mask_real.detach()).view(-1)
            realx_loss = criterion(out_realx, valid)

            out_fakex = discriminatorModel_x(fake_samples.detach(), real_mask_data).view(-1)

            fakex_loss = criterion(out_fakex, fake)
            # total loss and calculate the backprop
            dx_loss = (realx_loss + fakex_loss)

            dx_loss.backward()
            optimizer_Dx.step()

            optimizer_Di.zero_grad()

            out_reali = discriminatorModel_i(fake_mask_samples.detach())
            reali_loss = criterion(out_reali, real_mask)
            # total loss and calculate the backprop
            reali_loss.backward()
            optimizer_Di.step()
            # -----------------
            #  Train Generator
            # -----------------

            for p in discriminatorModel_x.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in discriminatorModel_i.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_Gx.zero_grad()

            fake_samples = generatorModel_x(real_mask_data,1)
            fake_mask_samples = mask_data(real_samples, real_mask, fake_samples)
            one_matrix = Variable(Tensor(real_samples.shape).fill_(1.0), requires_grad=False)
            dx = discriminatorModel_i(fake_mask_samples)
            gi_loss = criterion(dx, one_matrix)

            dx=discriminatorModel_x(fake_samples,real_mask_data).view(-1)
            gx_loss = criterion(dx, valid)
            mse = mse_loss(fake_samples, fake_mask_samples)/real_mask.sum()

            loss=gi_loss+gx_loss+mse
            loss.backward()
            optimizer_Gx.step()

            optimizer_Ex.zero_grad()
            fake_samples = generatorModel_x(real_mask_data,1)
            fake_mask_samples = mask_data(real_samples, real_mask, fake_samples)
            z_real = EncoderModel_x(fake_mask_samples)
            z_mask_real = mask_data(z_real, real_mask,z)
            dz = discriminatorModel_x(fake_mask_samples, z_mask_real).view(-1)
            ex_loss = criterion(dz, fake)
            ex_loss.backward()
            optimizer_Ex.step()

            gen_iterations += 1
            batches_done = epoch * len(dataloader_train) + i_batch + 1
            if batches_done % opt.sample_interval == 0:
                print(
                    'TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_Dx: %.6f  Loss_Di: %.6f Loss_Gx: %.6f  Loss_Gi: %.6f  Loss_E: %.6f Loss_Rec: %.6f'
                    % (epoch + 1, opt.n_epochs, i_batch + 1, len(dataloader_train), dx_loss.item(), reali_loss.item(),gx_loss.item(), gi_loss.item(), ex_loss.item(), mse.item()), flush=True)
        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.save_model == 0:
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel_x.state_dict(),
                'optimizer_G_state_dict': optimizer_Gx.state_dict(),
            }, os.path.join(opt.PATH, "model_epoch_%d.pth" % (epoch + 1)))



