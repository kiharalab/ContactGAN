import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import sys
import os

from tqdm import tqdm
from PIL import Image
import math
import numpy as np



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

class TrainDatasetFromFolder(Dataset):
    def __init__(self, lrd,hrd):
        super(TrainDatasetFromFolder, self).__init__()
        self.lr_imageFileNames = [os.path.join(lrd, x) for x in os.listdir(lrd)]
        self.hr_imageFileNames = [os.path.join(hrd, x) for x in os.listdir(hrd)]
        self.lr_imageFileNames.sort()
        self.hr_imageFileNames.sort()
    def __getitem__(self,index):
        lr_filename = self.lr_imageFileNames[index]
        hr_filename = self.hr_imageFileNames[index]
        
        lr_rowdata=getLRitem(lr_filename)       
        hr_rowdata=np.load(hr_filename)
        
        
        lr_image = lr_rowdata
        hr_image = hr_rowdata
     
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.lr_imageFileNames)

###############################################################
###Modify this to your contact map directories to train##########
hr='TRUE_CONTACT_DIRECTORY'
lr='METHOD_CONTACT_DIRECTORY'
###############################################################
dataset = TrainDatasetFromFolder(lr,hr)

batch_size = 1
val_batch_size = 1
validation_per = 0.1
shuffle_dataset = True
random_seed= 42

dataset_size = len(dataset)
print(dataset_size,flush=True)
indices = list(range(dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

split = 300


train_indices, val_indices = indices[split:], indices[:split]
print(len(train_indices),flush=True)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=val_batch_size,
                                                sampler=valid_sampler)


# Num batches
num_batches=len(train_loader)
print(len(train_loader),flush=True)
print(len(validation_loader),flush=True)


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
            self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
            self.relu4 = nn.PReLU()
            self.conv4_bn = nn.GroupNorm(32,d*8)
            self.conv44 = nn.Conv2d(d*8, d*8, 3, 2, 1)
            self.relu44 = nn.PReLU()
            self.conv44_bn = nn.GroupNorm(32,d*8)

            self.conv5 = nn.Conv2d(d*8, d*16, 3, 1, 1)
            self.relu5 = nn.PReLU()
            self.dropout5 = nn.Dropout(0.25)     
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
            nn.Conv2d(1, self.channels, kernel_size=9, padding=4),
            nn.PReLU()
        )


        model_sequence=[]
        kernel_size=3
        self.no_of_dil_blocks=no_of_dil_blocks
        # exp=-1
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
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)

        return image_loss + 0.001 * adversarial_loss


def evaluate(true_map, predicted_map):
    # Squeeze the maps
    true_map = true_map.squeeze().astype(float)
#     print(true_map.shape)
    predicted_map = predicted_map.squeeze().astype(float)
#     print(predicted_map.shape)
    assert(len(true_map.shape) == 2)
    assert(len(predicted_map.shape) == 2)

    accuracy_info = {}
    protein_len = true_map.shape[0]
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

            accuracy_info[str('L/{}'.format(r))][contact_type] = true_positives / L_r
    return accuracy_info            


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
parser.add_argument("--G_res_blocks", type=int, help="No. of Generator resnet blocks")
parser.add_argument("--D_res_blocks", type=int, help="No. of Discriminator resnet blocks")
parser.add_argument("--G_in_channels", type=int, help="No. of Generator input channels")
parser.add_argument("--D_in_channels", type=int, help="No. of Discriminator input channels")
parser.add_argument("--model_dir", help="Directory to save your model")
parser.add_argument("--GroupNorm", nargs='?', const=True, default=False, help="Use group norm in discriminator")

args = parser.parse_args()

out_dir = args.model_dir

lr = 0.001
train_epoch = 50
logger = Logger(model_name='CPGAN_deepcov_new', data_name='cmap')
G = Generator(args.G_in_channels,args.G_res_blocks)
D=Discriminator(args.D_in_channels,args.GroupNorm)

G.cuda()
D.cuda()

G_criterionLoss = GeneratorLoss().cuda()
D_criterionloss = nn.BCEWithLogitsLoss()

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))

for epoch in range(train_epoch):
    dl=tqdm(train_loader)
    for n_batch,(x_, target) in enumerate(dl):
                
        G.train()
        D.train()      
        D.zero_grad()

        mini_batch = x_.size()[0]
        y_dim=x_.size()[-1]
        if(y_dim>700 or y_dim==1):
#             print("Omitted:"+str(x_.size()))
            continue
        x_batched = [x_.detach().tolist()]
        x_batched=np.array(x_batched)
#         print(x_batched.shape)
        x_batched=torch.from_numpy(x_batched).float()
#         print(x_batched)

        target_batched = [target.detach().tolist()]
        target_batched=np.array(target_batched)
#         print(target_batched.shape)
        target_batched=torch.from_numpy(target_batched).float()        

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)      
        
        real_img=Variable(target_batched)
        if torch.cuda.is_available():
            real_img=real_img.cuda()
#         fake_z=Variable(x_)
        fake_z=Variable(x_batched)

        if torch.cuda.is_available():
            fake_z=fake_z.cuda()
        
        fake_img=G(fake_z)
        realD=D(real_img).mean()
        fakeD=D(fake_img).mean()
        # print(realD)
        # print(fakeD)
        D_real_loss = 1 - realD + fakeD

        D_real_loss.backward(retain_graph=True)
        D_optimizer.step()

        
        G.zero_grad()
        G_train_loss = G_criterionLoss(fakeD,fake_img,real_img)       # Use if already calculaating mean for fakeD
        # G_train_loss = G_criterionLoss(fakeD.mean(),fake_img,real_img)
        G_train_loss.backward()
        G_optimizer.step() 

        

        if (n_batch) % 1000 == 0 and n_batch!=0:  
            print('TRAIN - ',flush=True)
            logger.display_status(
                (epoch+1), train_epoch, n_batch, num_batches,
                D_real_loss, G_train_loss,0,0) 

#         print(torch.cuda.memory_allocated())
        del D_real_loss
        del G_train_loss
        torch.cuda.empty_cache()
        
        #validation
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if (n_batch) % 1000 == 0 and n_batch!=0:             
            torch.save(G.state_dict(),
                os.path.join(out_dir,'G_epoch'+"_"+str(n_batch)+'_'+str(epoch+1)))
            torch.save(D.state_dict(),
                os.path.join(out_dir,'D_epoch'+"_"+str(n_batch)+'_'+str(epoch+1)))
            G.eval()
            D.eval()
            gan_accuracy = {}
            ccm_accuracy = {}
            with torch.no_grad():
                for ix,(lr, hr) in enumerate(validation_loader):

                    y_vl_dim=lr.size()[-1]
                   
                    if(y_vl_dim>700 or y_vl_dim==1):
                        continue
                    
                    lr_batched = [lr.detach().tolist()]
                    lr_batched=np.array(lr_batched)

                    lr_batched=torch.from_numpy(lr_batched).float()
                    val_z=Variable(lr_batched)
                    val_z=val_z.cuda()
                    sr_test = G(val_z)

                    val_target_batched = [hr.detach().tolist()]
                    val_target_batched=np.array(val_target_batched)
                    val_target_batched=torch.from_numpy(val_target_batched).float()               
                    val_target=Variable(val_target_batched)
                    val_target=val_target.cuda()
                    hr_test=D(val_target).mean()
                    hr_fake=D(sr_test).mean()

                    G_loss_valid=G_criterionLoss(hr_fake, sr_test, val_target)
                    D_loss_valid=1 - hr_test + hr_fake
                    # D_loss_valid = D_criterionloss(hr_fake,hr_test)
    #                 print(torch.cuda.memory_allocated())


                    true_map=val_target.cpu().data.numpy()
                    output_map=sr_test.cpu().data.numpy()
                    ccm_map=val_z.cpu().data.numpy()
                    gan_accuracy_info=evaluate(true_map[0],output_map[0])
                    ccm_accuracy_info=evaluate(true_map[0],ccm_map[0])
                    gan_accuracy[ix] = gan_accuracy_info
                    ccm_accuracy[ix] = ccm_accuracy_info

                for r in [10, 5, 2, 1]:
                    for type in ['short', 'medium', 'long']:
                        top_l_r = str('L/{}'.format(r))
                        score = np.average([info[top_l_r][type] for info in gan_accuracy.values()])
                        ccmScore=np.average([info[top_l_r][type] for info in ccm_accuracy.values()])

                        print('ContactGAN - For {} and {}-range contacts: {}'.format(top_l_r, type, score),flush=True)
                        print('METHOD - For {} and {}-range contacts: {}'.format(top_l_r, type, ccmScore),flush=True)


                print('VALIDATION - ',flush=True)
                logger.display_status(
                    (epoch+1), train_epoch, n_batch, num_batches,
                    D_loss_valid, G_loss_valid, 0, 0) 
                del G_loss_valid
                del D_loss_valid          
                torch.cuda.empty_cache()  



