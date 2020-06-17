#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os

from tqdm import tqdm
from PIL import Image
import math
import numpy as np

from statistics import mean 



def normal_init(m, mean, std): 
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
def calc_pad(k,d):
    return int((k-1)*d/2)

class ResidualBlock(nn.Module):
    def __init__(self, channels,  in_channels=None,dilation=None, kernel_size=3):
        super(ResidualBlock, self).__init__()

        if dilation is None:
            padding=calc_pad(kernel_size,1)
            
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel_size,padding=padding)
            self.bn1 = nn.InstanceNorm2d(channels)
            self.relu1 = nn.PReLU()        
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size,padding=padding)
            self.relu2 = nn.PReLU()
            self.bn2 = nn.InstanceNorm2d(channels)
            self.dropout = nn.Dropout(0.25)
        else:
            padding=calc_pad(kernel_size,dilation)
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation= dilation)
            self.bn1 = nn.InstanceNorm2d(channels)
            self.relu1 = nn.PReLU()            
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation= dilation)
            self.relu2 = nn.PReLU()
            self.bn2 = nn.InstanceNorm2d(channels)
            self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.bn2(residual)
        residual = self.dropout(residual)
        return x + residual
    

def getLRitem(lr_filename):
    lr_image = np.loadtxt(lr_filename)
    return lr_image

def update_exp(exp):
    return (exp+1) if (exp<=8) else 8


class getDataset(Dataset):
    def __init__(self, lrd1, lrd2):
        super(getDataset, self).__init__()
        self.lr1_imageFileNames = [os.path.join(lrd1, x) for x in os.listdir(lrd1)]
        self.lr2_imageFileNames = [os.path.join(lrd2, x) for x in os.listdir(lrd2)]
        self.lr1_imageFileNames.sort()
        self.lr2_imageFileNames.sort()
        self.nameNoPath = [x for x in os.listdir(lrd1)]
        self.nameNoPath.sort()

    def __getitem__(self,index):
        lr1_filename = self.lr1_imageFileNames[index]
        lr2_filename = self.lr2_imageFileNames[index]
        protein = self.nameNoPath[index].split(".")[0]
        lr1_rowdata=getLRitem(lr1_filename)       
        lr2_rowdata=getLRitem(lr2_filename)   

        lr1_image = lr1_rowdata
        lr2_image = lr2_rowdata
        return lr1_image, lr2_image, protein
    
    def __len__(self):
        return len(self.lr1_imageFileNames)



class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=64, gn=False):
        super(Discriminator, self).__init__()

        if not gn:
            self.conv1 = nn.Conv2d(1, d, 3, 1, 1)
            self.relu1 = nn.PReLU()
            self.conv11 = nn.Conv2d(d, d, 3, 2, 1)
            self.relu11 = nn.PReLU()
            self.conv11_bn = nn.InstanceNorm2d(d)
            self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1)
            self.relu2 = nn.PReLU()
            self.conv2_bn = nn.InstanceNorm2d(d*2)
            self.conv22 = nn.Conv2d(d*2, d*2, 3, 2, 1)
            self.relu22 = nn.PReLU()
            self.conv22_bn = nn.InstanceNorm2d(d*2)
            self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
            self.relu3 = nn.PReLU()
            self.conv3_bn = nn.InstanceNorm2d(d*4)
            self.conv33 = nn.Conv2d(d*4, d*4, 3, 2, 1)
            self.relu33 = nn.PReLU()
            self.conv33_bn = nn.InstanceNorm2d(d*4)
            self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
            self.relu4 = nn.PReLU()
            self.conv4_bn = nn.InstanceNorm2d(d*8)
            self.conv44 = nn.Conv2d(d*8, d*8, 3, 2, 1)
            self.relu44 = nn.PReLU()
            self.conv44_bn = nn.InstanceNorm2d(d*8)
            self.conv5 = nn.Conv2d(d*8, d*16, 3, 1, 1)
            self.relu5 = nn.PReLU()
            self.dropout5 = nn.Dropout(0.25)
            
            
            self.conv6 = nn.Conv2d(d*16, 1, 1)

        else:
            self.conv1 = nn.Conv2d(1, d, 3, 1, 1)
            self.relu1 = nn.PReLU()
            self.conv11 = nn.Conv2d(d, d, 3, 2, 1)
            self.relu11 = nn.PReLU()
            self.conv11_bn = nn.GroupNorm(16,d)
            self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1)
            self.relu2 = nn.PReLU()
            self.conv2_bn = nn.GroupNorm(32,d*2)
            self.conv22 = nn.Conv2d(d*2, d*2, 3, 2, 1)
            self.relu22 = nn.PReLU()
            self.conv22_bn = nn.GroupNorm(32,d*2)

            self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
            self.relu3 = nn.PReLU()
            self.conv3_bn = nn.GroupNorm(32,d*4)
            self.conv33 = nn.Conv2d(d*4, d*4, 3, 2, 1)
            self.relu33 = nn.PReLU()
            self.conv33_bn = nn.GroupNorm(32,d*4)
            self.dropout33 = nn.Dropout(0.25)
            self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
            self.relu4 = nn.PReLU()
            self.conv44 = nn.Conv2d(d*8, d*8, 3, 2, 1)
            self.relu44 = nn.PReLU()
            self.conv44_bn = nn.GroupNorm(32,d*8)
            self.dropout44 = nn.Dropout(0.25)
            self.conv5 = nn.Conv2d(d*8, d*16, 3, 1, 1)
            self.relu5 = nn.PReLU()      
            self.conv6 = nn.Conv2d(d*16, 1, 1)
    def weight_init(self, mean=0, std=0.01):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)       
            
        
    # forward method
    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.relu11(self.conv11_bn(self.conv11(x)))
        x = self.relu2(self.conv2_bn(self.conv2(x)))
        x = self.relu22(self.conv22_bn(self.conv22(x)))
        x = self.relu3(self.conv3_bn(self.conv3(x)))
        x = self.relu33(self.conv33_bn(self.conv33(x)))
        x = self.relu4(self.conv4_bn(self.conv4(x)))
        x = self.relu44(self.conv44_bn(self.conv44(x)))
        x = self.relu5(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))      
        
        return x       

class Generator(nn.Module):
    # initializers
    def __init__(self, channels, no_of_dil_blocks):
        
        super(Generator, self).__init__()
        
        self.channels=channels
        self.block1 = nn.Sequential(
            nn.Conv2d(2, self.channels, kernel_size=9, padding=4),
            nn.PReLU()
        )


        model_sequence=[]
        kernel_size=3
        self.no_of_dil_blocks=no_of_dil_blocks
        exp=0

        for i in range(self.no_of_dil_blocks):
            model_sequence+=[ResidualBlock(self.channels,None,1,kernel_size)]           
            model_sequence+=[ResidualBlock(self.channels,None,2,kernel_size)]
            model_sequence+=[ResidualBlock(self.channels,None,4,kernel_size)]
        self.model=nn.Sequential(*model_sequence)
        
        self.blockSL = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.blockL = nn.Sequential(*self.blockSL)
        
    def weight_init(self, mean=0, std=0.01):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

        # forward method
    def forward(self, input):
        block1 = self.block1(input)
        x=self.model(block1)
        x=self.blockSL(x)

        blockL = self.blockL(block1 + x)
        x = torch.sigmoid(blockL)

        return x


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        image_loss = self.mse_loss(out_images, target_images)
        return image_loss + 0.001 * adversarial_loss


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs=2, help="Path to contact map input")
parser.add_argument("--G_path", help="Generator mdoel path")
parser.add_argument("--D_path", help="Discriminator mdoel path")
parser.add_argument("--G_res_blocks", type=int, help="No. of Generator resnet blocks")
parser.add_argument("--D_res_blocks", type=int, help="No. of Discriminator resnet blocks")
parser.add_argument("--GroupNorm", nargs='?', const=True, default=False, help="Use group norm in discriminator")

args = parser.parse_args()
pathG=args.G_path
pathD=args.D_path


lr1=args.input[0]
lr2=args.input[1]
dataset = getDataset(lr1,lr2)

batch_size = 1

dataset_size = len(dataset)
print(dataset_size)
indices = list(range(dataset_size))

test_sampler = SubsetRandomSampler(indices)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)



G = Generator(args.G_in_channels,args.G_res_blocks)
G.load_state_dict(torch.load(pathG))
G.eval()
D = Discriminator(args.D_in_channels,args.GroupNorm)
D.load_state_dict(torch.load(pathD))
D.eval()
G.cuda()
D.cuda()

G_criterionLoss = GeneratorLoss().cuda()
D_criterionloss = nn.BCEWithLogitsLoss()

G.eval()
D.eval()


with torch.no_grad():
    for ix,(lr1, lr2, name) in enumerate(test_loader):
        y_vl_dim=lr1.size()[-1]

        if(lr1.size()[-1]!=lr2.size()[-1]):
            continue    
       
        lr_batched=np.array([np.dstack((lr1.detach().tolist()[0],lr2.detach().tolist()[0]))])
        lr_batched=np.array(lr_batched)

        lr_batched=torch.from_numpy(lr_batched).float()
        val_z=Variable(lr_batched.permute(0,3,1,2))
        val_z=val_z.cuda()
        sr_test = G(val_z)

        output_map=sr_test.cpu().data.numpy()
        ccm_map=val_z.cpu().data.numpy()

        if not os.path.exists(os.path.join('./data/output_dir/')):
            os.mkdir('./data/output_dir/')
        np.save(os.path.join('./data/output_dir/Original_','{}.npy'.format(name[0])), ccm_map)
        np.save(os.path.join('./data/output_dir/ContactGAN_','{}.npy'.format(name[0])), output_map)