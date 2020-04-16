#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# from scipy.misc import imresize
# from torchvision.models.vgg import vgg16
from tqdm import tqdm
from PIL import Image
import math
import numpy as np

from statistics import mean 



def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100).view(-1, 100, 1, 1).cuda())
    return n

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1)).long()
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1)).long()
    return data

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
            # self.bn1 = nn.GroupNorm2d(channels,8)
            # self.relu = nn.PReLU()
            self.relu1 = nn.PReLU()
            

            self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size,padding=padding)
            self.relu2 = nn.PReLU()
            self.bn2 = nn.InstanceNorm2d(channels)
            self.dropout = nn.Dropout(0.25)
            # self.bn2 = nn.GroupNorm2d(channels,8)

        else:
            padding=calc_pad(kernel_size,dilation)
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation= dilation)
            self.bn1 = nn.InstanceNorm2d(channels)
            # self.relu = nn.PReLU()
            self.relu1 = nn.PReLU()
            
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation= dilation)
            self.relu2 = nn.PReLU()
            self.bn2 = nn.InstanceNorm2d(channels)
            self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # print("Res input:"+str(x.size()))
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.bn1(residual)
        # residual = self.prelu(residual)

        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.bn2(residual)
        residual = self.dropout(residual)

        # print("Res out:"+str(residual.size()))
        return x + residual
    

def getLRitem(lr_filename):
    lr_image = np.loadtxt(lr_filename)
    return lr_image

def update_exp(exp):
    return (exp+1) if (exp<=8) else 8

def getSpotItem(fil):
    ss_list=[]
    with open(fil,"r") as f:
        for l in f.readlines():
            if "#" in l:
                continue
            ss=l.strip().split()[2]
            if ss=="H":
                ss_list.append(2)
            elif ss=="E":
                ss_list.append(1)
            else:
                ss_list.append(0)

    return ss_list

class TrainDatasetFromFolder(Dataset):
    def __init__(self, lrd,hrd):
        super(TrainDatasetFromFolder, self).__init__()
        self.lr_imageFileNames = [os.path.join(lrd, x) for x in os.listdir(lrd)]
        # self.hr_imageFileNames = [os.path.join(hrd, x) for x in os.listdir(hrd)]+[os.path.join(hrd, x) for x in os.listdir(hrd)]
        self.hr_imageFileNames = [os.path.join(hrd, x) for x in os.listdir(hrd)]
        self.lr_imageFileNames.sort()
        self.hr_imageFileNames.sort()
        self.nameNoPath = [x for x in os.listdir(lrd)]
        self.nameNoPath.sort()
        print(self.hr_imageFileNames[5497:5507])
        print(self.lr_imageFileNames[5497:5507])

    def __getitem__(self,index):
        lr_filename = self.lr_imageFileNames[index]
        hr_filename = self.hr_imageFileNames[index]
        protein = self.nameNoPath[index].split(".")[0]
        spotFile = os.path.join("/net/kihara/home/ykagaya/work/20191005-SPOT1D/outputs",protein+".spot1d")
        
#         print(lr_filename)
#         print(hr_filename)
        lr_rowdata=getLRitem(lr_filename)       
        hr_rowdata=np.load(hr_filename)
        
        ss_list=getSpotItem(spotFile)
        #print(lr_rowdata)
        #lr_image = vectors_to_images(lr_rowdata)
        
        lr_image = lr_rowdata
        hr_image = hr_rowdata
     
    #         print(lr_image.shape)
#         print(hr_image.shape)
        return lr_image, hr_image, protein, ss_list
    
    def __len__(self):
        return len(self.lr_imageFileNames)

# root='/net/kihara/scratch/smaddhur/SuperReso/dataset_old'
hr='/net/kihara/home/smaddhur/tensorFlow/CPGAN/input_matrix/true_contacts'
# lr='/net/kihara/home/smaddhur/tensorFlow/CPGAN/input_matrix/ccm_contacts'
lr='/net/kihara/home/smaddhur/tensorFlow/CPGAN/input_matrix/deepcov_matrix'
# lr='/net/kihara/home/smaddhur/tensorFlow/CPGAN/input_matrix/deepcontact_contacts'
dataset = TrainDatasetFromFolder(lr,hr)


# In[5]:


# Load data
# Create loader with data
batch_size = 1
val_batch_size = 1
validation_per = 0.1
shuffle_dataset = True
random_seed= 42

# data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Creating data indices for training and validation splits:
# dataset_size = len(data_loader)*batch_size
dataset_size = len(dataset)
print(dataset_size)
indices = list(range(dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

split = 300


train_indices, val_indices = indices[split:], indices[:split]
print(len(train_indices))
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=val_batch_size,
                                                sampler=valid_sampler)


# Num batches
num_batches=len(train_loader)
print(len(train_loader))
print(len(validation_loader))

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=64, gn=False):
        super(Discriminator, self).__init__()

        if not gn:
            self.conv1 = nn.Conv2d(1, d, 3, 1, 1)
            self.relu1 = nn.PReLU()
            self.conv11 = nn.Conv2d(d, d, 3, 2, 1)
            # self.conv11_bn = nn.GroupNorm(16,d)
            self.relu11 = nn.PReLU()
            self.conv11_bn = nn.InstanceNorm2d(d)
            # self.dropout11 = nn.Dropout(0.25)

            self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1)
            self.relu2 = nn.PReLU()
            # self.conv2_bn = nn.GroupNorm(32,d*2)
            self.conv2_bn = nn.InstanceNorm2d(d*2)
            self.conv22 = nn.Conv2d(d*2, d*2, 3, 2, 1)
            self.relu22 = nn.PReLU()
            self.conv22_bn = nn.InstanceNorm2d(d*2)
            # self.dropout22 = nn.Dropout(0.25)
            # self.conv22_bn = nn.GroupNorm(32,d*2)
            self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
            self.relu3 = nn.PReLU()
            # self.conv3_bn = nn.GroupNorm(32,d*4)
            self.conv3_bn = nn.InstanceNorm2d(d*4)
            self.conv33 = nn.Conv2d(d*4, d*4, 3, 2, 1)
            self.relu33 = nn.PReLU()
            # self.conv33_bn = nn.GroupNorm(32,d*4)
            self.conv33_bn = nn.InstanceNorm2d(d*4)
            # self.dropout33 = nn.Dropout(0.25)

            self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
            self.relu4 = nn.PReLU()
            # self.conv4_bn = nn.GroupNorm(32,d*8)
            self.conv4_bn = nn.InstanceNorm2d(d*8)
            self.conv44 = nn.Conv2d(d*8, d*8, 3, 2, 1)
            self.relu44 = nn.PReLU()
            # self.conv44_bn = nn.GroupNorm(32,d*8)
            self.conv44_bn = nn.InstanceNorm2d(d*8)
            # self.dropout44 = nn.Dropout(0.25)
            #self.pool1=nn.AdaptiveAvgPool2d(1)
            self.conv5 = nn.Conv2d(d*8, d*16, 3, 1, 1)
            self.relu5 = nn.PReLU()
            self.dropout5 = nn.Dropout(0.25)
            #self.conv5 = nn.Conv2d(d*4, d*8, 3, 1, 1)
            
            
            self.conv6 = nn.Conv2d(d*16, 1, 1)
        #self.conv6 = nn.Conv2d(d*8, 1, 1)

        else:
            # print("GN")
            self.conv1 = nn.Conv2d(1, d, 3, 1, 1)
            self.relu1 = nn.PReLU()
            self.conv11 = nn.Conv2d(d, d, 3, 2, 1)
            self.relu11 = nn.PReLU()
            self.conv11_bn = nn.GroupNorm(16,d)
            # self.dropout11 = nn.Dropout(0.25)

            self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1)
            self.relu2 = nn.PReLU()
            self.conv2_bn = nn.GroupNorm(32,d*2)
            # self.conv2_bn = nn.InstanceNorm2d(d*2)
            self.conv22 = nn.Conv2d(d*2, d*2, 3, 2, 1)
            self.relu22 = nn.PReLU()
            # self.conv22_bn = nn.InstanceNorm2d(d*2)
            self.conv22_bn = nn.GroupNorm(32,d*2)
            # self.dropout22 = nn.Dropout(0.25)

            self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
            self.relu3 = nn.PReLU()
            self.conv3_bn = nn.GroupNorm(32,d*4)
            # self.conv3_bn = nn.InstanceNorm2d(d*4)
            self.conv33 = nn.Conv2d(d*4, d*4, 3, 2, 1)
            self.relu33 = nn.PReLU()
            self.conv33_bn = nn.GroupNorm(32,d*4)
            self.dropout33 = nn.Dropout(0.25)
            # self.conv33_bn = nn.InstanceNorm2d(d*4)
            self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
            self.relu4 = nn.PReLU()
            # self.conv4_bn = nn.GroupNorm(32,d*8)
            # self.conv4_bn = nn.InstanceNorm2d(d*8)
            self.conv44 = nn.Conv2d(d*8, d*8, 3, 2, 1)
            self.relu44 = nn.PReLU()
            self.conv44_bn = nn.GroupNorm(32,d*8)
            self.dropout44 = nn.Dropout(0.25)
            # self.conv44_bn = nn.InstanceNorm2d(d*8)
            #self.pool1=nn.AdaptiveAvgPool2d(1)
            self.conv5 = nn.Conv2d(d*8, d*16, 3, 1, 1)
            self.relu5 = nn.PReLU()
            # self.dropout5 = nn.Dropout(0.25)
            #self.conv5 = nn.Conv2d(d*4, d*8, 3, 1, 1)
        
        
            self.conv6 = nn.Conv2d(d*16, 1, 1)
        #self.conv6 = nn.Conv2d(d*8, 1, 1)   
    def weight_init(self, mean=0, std=0.01):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)       
            
        
    # forward method
    def forward(self, input):
#         print("Disc in:"+str(input.size()))

        x = self.relu1(self.conv1(input))
        x = self.relu11(self.conv11_bn(self.conv11(x)))
        # x = self.dropout11(x)
        x = self.relu2(self.conv2_bn(self.conv2(x)))
        x = self.relu22(self.conv22_bn(self.conv22(x)))
        # x = self.dropout22(x)
        x = self.relu3(self.conv3_bn(self.conv3(x)))
        x = self.relu33(self.conv33_bn(self.conv33(x)))
        # x = self.dropout33(x)
        x = self.relu4(self.conv4_bn(self.conv4(x)))
        x = self.relu44(self.conv44_bn(self.conv44(x)))
        # x = self.dropout44(x)
        x = self.relu5(self.conv5(x))
        # x = self.dropout5(x)
        

        # x = F.leaky_relu(self.conv1(input), 0.2)
        # x = F.leaky_relu(self.conv11_bn(self.conv11(x)), 0.2)
        # x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv22_bn(self.conv22(x)), 0.2)
        # x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv33_bn(self.conv33(x)), 0.2)
        # x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = F.leaky_relu(self.conv44_bn(self.conv44(x)), 0.2)
        # x = F.leaky_relu(self.conv5(x), 0.2)
#         print("Disc out:"+str(x.size()))
        x = torch.sigmoid(self.conv6(x))
        # x = (self.conv6(x))
        # x = F.leaky_relu(self.conv6(x))
        
        
        return x       

class Generator(nn.Module):
    # initializers
    def __init__(self, channels, no_of_dil_blocks):
        
        super(Generator, self).__init__()
        
        self.channels=channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=9, padding=4),
            nn.PReLU()
        )


        model_sequence=[]
        kernel_size=3
        self.no_of_dil_blocks=no_of_dil_blocks
        # exp=-1
        exp=0

        for i in range(self.no_of_dil_blocks):
            # if i<=0:
            #     kernel_size=5



            # exp=update_exp(exp)
            # model_sequence+=[ResidualBlock(self.channels,None,2**exp,kernel_size)]
            # exp=update_exp(exp)
            # model_sequence+=[ResidualBlock(self.channels,None,2**exp,kernel_size)]
            # exp=update_exp(exp)
            # model_sequence+=[ResidualBlock(self.channels,None,2**exp,kernel_size)]
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
#         print("G imnput:"+str(input.size()))
        block1 = self.block1(input)
        x=self.model(block1)
        x=self.blockSL(x)

        blockL = self.blockL(block1 + x)
        # x=blockL
        # x = (F.tanh(blockL) + 1) / 2
        x = torch.sigmoid(blockL)

        return x


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # image_loss = self.bce_loss(out_images, target_images)
        # TV Loss
        return image_loss + 0.001 * adversarial_loss
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss


def ss_evaluate(true_map, method_map, predicted_map, ss_list):
    true_map = true_map.squeeze().astype(float)
#     print(true_map.shape)
    method_map = method_map.squeeze().astype(float)
    predicted_map = predicted_map.squeeze().astype(float)
#     print(predicted_map.shape)
    assert(len(true_map.shape) == 2)
    assert(len(method_map.shape) == 2)
    assert(len(predicted_map.shape) == 2)

    protein_len = true_map.shape[0]
    assert(protein_len == len(ss_list))
    acount=0
    nacount=0
    bcount=0
    nbcount=0
    abcount=0
    nabcount=0    
    xs = ys = int(true_map.shape[0])
    alpha_prob_diff=[]
    beta_prob_diff=[]
    ab_prob_diff=[]
    for i in range(xs):
        for j in range(i,ys,1):
            if true_map[i, j] == 1.0:
                if ss_list[i]==2 and ss_list[j]==2:
                    if predicted_map[i,j] > method_map[i,j]:
                        acount+=1
                    elif predicted_map[i,j] < method_map[i,j]:
                        nacount+=1
                    alpha_prob_diff.append(predicted_map[i,j]-method_map[i,j])
                elif ss_list[i]==1 and ss_list[j]==1:
                    if predicted_map[i,j] > method_map[i,j]:
                        bcount+=1
                    elif predicted_map[i,j] < method_map[i,j]:
                        nbcount+=1
                    beta_prob_diff.append(predicted_map[i,j]-method_map[i,j])
                elif (ss_list[i]==2 and ss_list[j]==1) or (ss_list[i]==1 and ss_list[j]==2):
                    if predicted_map[i,j] > method_map[i,j]:
                        abcount+=1
                    elif predicted_map[i,j] < method_map[i,j]:
                        nabcount+=1
                    ab_prob_diff.append(predicted_map[i,j]-method_map[i,j])
    # print("good : {},bad : {}".format(str(acount),str(ncount)))
    amean=bmean=abmean=0
    if len(alpha_prob_diff)>0:
        amean = mean(alpha_prob_diff)
    if len(beta_prob_diff)>0:
        bmean = mean(beta_prob_diff)
    if len(ab_prob_diff)>0:
        abmean = mean(ab_prob_diff)
    
    return acount,nacount,bcount,nbcount,abcount,nabcount, amean , bmean, abmean




def evaluate(true_map, predicted_map,ss_list):
    # Squeeze the maps
    true_map = true_map.squeeze().astype(float)
#     print(true_map.shape)
    predicted_map = predicted_map.squeeze().astype(float)
#     print(predicted_map.shape)
    assert(len(true_map.shape) == 2)
    assert(len(predicted_map.shape) == 2)


    accuracy_info = {}
    protein_len = true_map.shape[0]
    assert(protein_len == len(ss_list))
    aacnt = 0
    bbcnt=0
    abcnt=0
    for r in [10, 5, 2, 1]:
        L_r = int(round(protein_len / r))
        accuracy_info[str('L/{}'.format(r))] = {}
        for contact_type in ['short', 'medium', 'long']:
            if contact_type == 'short':
                min_separation, max_separation = 6, 11
            if contact_type == 'medium':
                min_separation, max_separation = 12, 23
            if contact_type == 'long':
                min_separation, max_separation = 24, None

            # Get top pairs
            nb_founds, predicted_top_pairs = get_top_pairs(predicted_map, L_r, min_separation, max_separation)

            # Count number of true positives
            true_positives = 0
            for i in range(nb_founds):
                idx_x, idx_y = predicted_top_pairs[0][i], predicted_top_pairs[1][i]
                if true_map[idx_x, idx_y] == 1.0:
                    true_positives += 1
            
            if r==1 and contact_type=='long':
                # print(predicted_top_pairs)

                for i in range(nb_founds):
                    idx_x, idx_y = predicted_top_pairs[0][i], predicted_top_pairs[1][i]
                    if true_map[idx_x, idx_y] == 1.0 and ss_list[idx_x]==2 and ss_list[idx_y]==2:
                    # if true_map[idx_x, idx_y] == 1.0:
                        # print("x:{},y:{}".format(str(ss_list[idx_x]),str(ss_list[idx_y])))
                        aacnt+=1

                    elif true_map[idx_x, idx_y] == 1.0 and ss_list[idx_x]==1 and ss_list[idx_y]==1:
                    # if true_map[idx_x, idx_y] == 1.0:
                        # print("x:{},y:{}".format(str(ss_list[idx_x]),str(ss_list[idx_y])))
                        bbcnt+=1

                    elif true_map[idx_x, idx_y] == 1.0 and ((ss_list[idx_x]==1 and ss_list[idx_y]==2) or (ss_list[idx_x]==2 and ss_list[idx_y]==1)):
                    # if true_map[idx_x, idx_y] == 1.0:
                        # print("x:{},y:{}".format(str(ss_list[idx_x]),str(ss_list[idx_y])))
                        abcnt+=1

            # Update accuracy_info
            accuracy_info[str('L/{}'.format(r))][contact_type] = true_positives / L_r
    # print("DONE")
    return accuracy_info, aacnt, bbcnt, abcnt            



def get_top_pairs(mat, num_contacts, min_separation, max_separation = None):
    """Get the top-scoring contacts"""
    idx_delta = np.arange(mat.shape[1])[np.newaxis, :] - np.arange(mat.shape[0])[:, np.newaxis]

    if max_separation:
        mask = (idx_delta < min_separation) | (idx_delta > max_separation)
    else:
        mask = idx_delta < min_separation

    mat_masked = np.copy(mat)
    mat_masked[mask] = float("-inf")

    top = mat_masked.argsort(axis=None)[::-1][:(num_contacts)]
    top = (top % mat.shape[0]).astype(np.uint16), np.floor(top / mat.shape[0]).astype(np.uint16)

    # Post-filtering
    filtered_indices_x, filtered_indices_y, num_contacts_found = [], [], 0
    indices_x, indices_y = top
    for i in range(num_contacts):
        index_x, index_y = indices_x[i], indices_y[i]
        dist = abs(index_x.astype(float) - index_y.astype(float))
        if dist < min_separation or (max_separation != None and dist > max_separation): continue
        num_contacts_found += 1
        filtered_indices_x.append(index_x)
        filtered_indices_y.append(index_y)

    return num_contacts_found, (filtered_indices_x, filtered_indices_y)

from utils1 import Logger

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--G_path", help="Generator mdoel path")
parser.add_argument("--D_path", help="Discriminator mdoel path")
parser.add_argument("--G_res_blocks", type=int, help="No. of Generator resnet blocks")
parser.add_argument("--D_res_blocks", type=int, help="No. of Discriminator resnet blocks")
parser.add_argument("--G_in_channels", type=int, help="No. of Generator input channels")
parser.add_argument("--D_in_channels", type=int, help="No. of Discriminator input channels")
parser.add_argument("--GroupNorm", nargs='?', const=True, default=False, help="Use group norm in discriminator")

args = parser.parse_args()
pathG=args.G_path
pathD=args.D_path

lr = 0.001
train_epoch = 50
logger = Logger(model_name='CPGAN_trrosetta_in_dg', data_name='cmap')
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
#G_criterionLoss = GeneratorLoss()

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))

G.eval()
D.eval()
gan_accuracy = {}
ccm_accuracy = {}

gan_accuracies = {}
method_accuracies = {}

alpha_arr=[]
alpha_neg_arr=[]
beta_arr=[]
beta_neg_arr=[]
ab_arr=[]
ab_neg_arr=[]
alpha_arr_avg=[]
beta_arr_avg=[]
ab_arr_avg=[]

alpha_arr_l1=[]
alpha_neg_arr_l1=[]
beta_arr_l1=[]
beta_neg_arr_l1=[]
ab_arr_l1=[]
ab_neg_arr_l1=[]

method="deepcov"
with torch.no_grad():
    for ix,(lr, hr, name, ss_list) in enumerate(validation_loader):
        # if name[0] not in ['1BHUA', '2RNGA', '5GZTA', '1HP8A', '2CWYA', '2K49A', '2JMSA', '1VQOZ', '2KRXA', '3ID1A', '1UJ8A', '5YWRB', '5APGB', '2I9XA', '3BW6A', '5OHQA', '1IM3P', '3S8SA', '5M0WA', '3PESB', '2KSNA', '4XALA', '6FC0B', '1TM9A', '4WFTC', '6H9HB', '3BAMA', '1X9BA']:
        # if name[0] not in ['1UJ8A','4WFTC']:
        #     continue

        # if not os.path.exists(os.path.join("/net/kihara/home/ykagaya/work/20191005-SPOT1D/outputs",name[0]+".spot1d")):
        #     print(name[0])
        y_vl_dim=lr.size()[-1]
       
        if(y_vl_dim>700 or y_vl_dim==1):
#                         print("Val Omitted:"+str(lr.size()))
            continue
        
        lr_batched = [lr.detach().tolist()]
        lr_batched=np.array(lr_batched)
#                     print(lr_batched.shape)

        lr_batched=torch.from_numpy(lr_batched).float()
        val_z=Variable(lr_batched)
        val_z=val_z.cuda()
        sr_test = G(val_z)

        val_target_batched = [hr.detach().tolist()]
        val_target_batched=np.array(val_target_batched)
#                     print(val_target_batched.shape)
        val_target_batched=torch.from_numpy(val_target_batched).float()               
        val_target=Variable(val_target_batched)
        val_target=val_target.cuda()
        hr_test=D(val_target).mean()
        hr_fake=D(sr_test).mean()
        # hr_test=D(val_target)
        # hr_fake=D(sr_test)

        G_loss_valid=G_criterionLoss(hr_fake, sr_test, val_target)
        # G_loss_valid=G_criterionLoss(hr_fake.mean(), sr_test, val_target)
        D_loss_valid=1 - hr_test + hr_fake
        # D_loss_valid = D_criterionloss(hr_fake,hr_test)
#                 print(torch.cuda.memory_allocated())


        true_map=val_target.cpu().data.numpy()
        output_map=sr_test.cpu().data.numpy()
        ccm_map=val_z.cpu().data.numpy()
        gan_accuracy_info, aacnt, bbcnt, abcnt=evaluate(true_map[0],output_map[0],ss_list)
        ccm_accuracy_info, aacnt1, bbcnt1, abcnt1=evaluate(true_map[0],ccm_map[0],ss_list)
        gan_accuracy[name[0]] = gan_accuracy_info
        ccm_accuracy[name[0]] = ccm_accuracy_info

        alpha_arr_l1.append(aacnt)    
        alpha_neg_arr_l1.append(aacnt1)    

        beta_arr_l1.append(bbcnt)    
        beta_neg_arr_l1.append(bbcnt1)    

        ab_arr_l1.append(abcnt)    
        ab_neg_arr_l1.append(abcnt1)

        ax,nax,bx,nbx,abx,nabx,aavg,bavg,abavg = ss_evaluate(true_map[0],ccm_map[0],output_map[0],ss_list)
        alpha_arr.append(ax)
        alpha_neg_arr.append(nax)
        alpha_arr_avg.append(aavg)
        beta_arr.append(bx)
        beta_neg_arr.append(nbx)
        beta_arr_avg.append(bavg)
        ab_arr.append(abx)
        ab_neg_arr.append(nabx)
        ab_arr_avg.append(abavg)


        if not os.path.exists(os.path.join('./paper/output_npy/validation/validTrue_maps/')):
            os.mkdir('./paper/output_npy/validation/validTrue_maps/')
        np.save('./paper/output_npy/validation/validTrue_maps/{}.npy'.format(name[0]), true_map[0])
        if not os.path.exists(os.path.join('./paper/output_npy/validation/',method)):
            os.mkdir(os.path.join('./paper/output_npy/validation/',method))
        np.save(os.path.join('./paper/output_npy/validation/',method,'{}.npy'.format(name[0])), ccm_map)
        if not os.path.exists(os.path.join('./paper/output_npy/validation/validPred_'+method)):
            os.mkdir(os.path.join('./paper/output_npy/validation/validPred_'+method))
        np.save(os.path.join('./paper/output_npy/validation/validPred_'+method,'{}.npy'.format(name[0])), output_map)
       
    ac=0
    ag=0
    tot=0
    for i in range(len(alpha_arr)):
        if alpha_arr[i]<alpha_neg_arr[i]:
            ac+=1
        if alpha_arr[i]>alpha_neg_arr[i]:
            ag+=1
        if not (alpha_arr[i]==0 and alpha_neg_arr[i]==0):
            tot+=1
    print("Alpha-alpha Probability values - total : {}, good; {}, bad : {}".format(str(tot),str(ag),str(ac)))

    ac=0
    ag=0
    tot=0
    for i in range(len(beta_arr)):
        if beta_arr[i]<beta_neg_arr[i]:
            ac+=1
        if beta_arr[i]>beta_neg_arr[i]:
            ag+=1
        if not (beta_arr[i]==0 and beta_neg_arr[i]==0):
            tot+=1
    print("Beta-beta Probability values - total : {}, good; {}, bad : {}".format(str(tot),str(ag),str(ac)))

    ac=0
    ag=0
    tot=0
    for i in range(len(ab_arr)):
        if ab_arr[i]<ab_neg_arr[i]:
            ac+=1
        if ab_arr[i]>ab_neg_arr[i]:
            ag+=1
        if not (ab_arr[i]==0 and ab_neg_arr[i]==0):
            tot+=1
    print("Alpha-beta Probability values - total : {}, good; {}, bad : {}".format(str(tot),str(ag),str(ac)))

    aaavg=mean(alpha_arr_avg)
    bbavg=mean(beta_arr_avg)
    aabbavg=mean(ab_arr_avg)
    print("Avg Prob change - Alpha-alpha : {}, beta-beta : {}, alpha-beta : {}".format(str(aaavg),str(bbavg),str(aabbavg)))


    acl=0
    agl=0
    totl=0
    for i in range(len(alpha_arr_l1)):
        if alpha_arr_l1[i]<alpha_neg_arr_l1[i]:
            acl+=1
        if alpha_arr_l1[i]>alpha_neg_arr_l1[i]:
            agl+=1
        if not (alpha_arr_l1[i]==0 and alpha_neg_arr_l1[i]==0):
            totl+=1
    print("L/1 Long Alpha-alpha - total : {}, good; {}, bad : {}".format(str(totl),str(agl),str(acl)))


    acl=0
    agl=0
    totl=0
    for i in range(len(beta_arr_l1)):
        if beta_arr_l1[i]<beta_neg_arr_l1[i]:
            acl+=1
        if beta_arr_l1[i]>beta_neg_arr_l1[i]:
            agl+=1
        if not (beta_arr_l1[i]==0 and beta_neg_arr_l1[i]==0):
            totl+=1
    print("L/1 Long Beta-beta - total : {}, good; {}, bad : {}".format(str(totl),str(agl),str(acl)))

    acl=0
    agl=0
    totl=0
    for i in range(len(ab_arr_l1)):
        if ab_arr_l1[i]<ab_neg_arr_l1[i]:
            acl+=1
        if ab_arr_l1[i]>ab_neg_arr_l1[i]:
            agl+=1
        if not (ab_arr_l1[i]==0 and ab_neg_arr_l1[i]==0):
            totl+=1
    print("L/1 Long Alpha-beta - total : {}, good; {}, bad : {}".format(str(totl),str(agl),str(acl)))

    for r in [10, 5, 2, 1]:
        xA=[]
        yA=[]
        for type in ['short', 'medium', 'long']:
            top_l_r = str('L/{}'.format(r))
            gan_accuracy_list={}
            ccm_accuracy_list={}
            for k in gan_accuracy:
                gan_accuracy_list[k] = gan_accuracy[k][top_l_r][type]
                ccm_accuracy_list[k] = ccm_accuracy[k][top_l_r][type]
            score = np.average(list(gan_accuracy_list.values()))
            ccmScore=np.average(list(ccm_accuracy_list.values()))

            x=[]
            y=[]
            # for k in gan_accuracy_list:
            #     if type=='long':
            #         print('{},L/{},{},{},{}'.format(type,r,k,str(gan_accuracy_list[k]),str(ccm_accuracy_list[k])))
            strs=[]
            with open(os.path.join("./paper/results/valid_"+method,"validation_"+method+"_L"+str(r)+"_"+type),"w") as wf:
            # with open(os.path.join("tmp"),"w") as wf:
                for k in gan_accuracy_list:
                    # if type == 'long' and r==10 and (gan_accuracy_list[k]-ccm_accuracy_list[k]) <= -0.3:
                    #     # print('{},{},{}'.format(k,str(gan_accuracy_list[k]),str(ccm_accuracy_list[k])))
                    #     strs.append(k)
                                            
                    wf.write('{},{}\n'.format(str(gan_accuracy_list[k]),str(ccm_accuracy_list[k])))
                    # x.append(gan_accuracy_list[])
                    # y.append(ccm_accuracy_list)
            # print(strs)
            xA.append(list(gan_accuracy_list.values()))
            yA.append(list(ccm_accuracy_list.values()))

            print('CPGAN - For {} and {}-range contacts: {}'.format(top_l_r, type, score))
            print('CCMPRED - For {} and {}-range contacts: {}'.format(top_l_r, type, ccmScore))
        # plt.scatter(xA[0],yA[0],c='#00ffff', marker='^', s=12, label='L/{}'.format(r)+' short',linestyle='None')
        plt.scatter(xA[1],yA[1],c='#40E0D0', marker="o", s=12, label='L/{}'.format(r)+' medium',linestyle='None')
        plt.scatter(xA[2],yA[2],c='#ff00ff', marker="x", s=12, label='L/{}'.format(r)+' long',linestyle='None')
        plt.plot([0,0.1,0.3,0.5,0.7,0.9,1],[0,0.1,0.3,0.5,0.7,0.9,1])            
        plt.xlabel('CPGAN precision')
        plt.ylabel(method+' precision')
        plt.title(method+' Validation performance')
        legend=plt.legend()
        plt.savefig("validation_"+method+"_"+str(r)+".jpg",dpi=300)
        plt.close()
        


    del G_loss_valid
    del D_loss_valid          



