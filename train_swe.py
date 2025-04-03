## https://github.com/vincent-leguen/PhyDNet/blob/master/data/moving_mnist.py

import numpy as np
import os
from PIL import Image
import random
import gc
import torch.utils.data as data
from torch.autograd import Variable

import time
#from skimage.measure import compare_ssim as ssim
#from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
import argparse

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
import torchvision
# import moviepy.editor as mp
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
import torchvision.transforms as T
from tqdm.auto import tqdm
import time

from adrNet import *  
from torchvision.transforms.functional import rgb_to_grayscale
from sweDataLoader import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#########################
batch_size = 256
test_ratio = 0.1
num_workers = 4
history = 10
prediction = 50
#lst = list_h5_datasets('./PDEBench/shallow-water/2D_rdb_NA_NA.h5')
#d = load_h5_to_tensor('./PDEBench/shallow-water/2D_rdb_NA_NA.h5', lst[0])

train_data = SWEDataset('/gladwell/ndj376/ADRnet/SWE/swe'+str(history)+'_'+str(prediction)+'_train_data.pt')
test_data = SWEDataset('/gladwell/ndj376/ADRnet/SWE/swe'+str(history)+'_'+str(prediction)+'_test_data.pt')

#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False) 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False) 

#xm = 0
#xmax = 0
#xstd = 0
#xmin = 0
#for x,y,t in test_loader:
#    xm = xm + x.mean()
#    xstd = xstd + x.std()
#    xmax = xmax + x.max()
#    xmin = xmin + x.min()
#xm = xm/len(test_loader)
#xstd = xstd/len(test_loader)
#xmax = xmax/len(test_loader)
#xmin = xmin/len(test_loader)
#print('mean:',xm)
#print('std:',xstd)
#print('min:',xmin)
#print('max:',xmax)

#mean: tensor(1.0338)
#std: tensor(0.1119)
#min: tensor(0.2729)
#max: tensor(2.)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ',device)

total_len = len(train_loader)  

in_c = history
SZ = 64
Mask = torch.ones(SZ, SZ)
model = resnet(in_c=history, hid_c = 128, out_c=prediction, nlayers=1, imsz=[SZ, SZ], integrator="FE")
model.to(device)

print('Number of model parameters = %3d'%(count_parameters(model)))

lr = 1e-4
optim = Adam(model.parameters(), lr)
epochs = 200
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999999)

f = open("detail.txt", "w")
f.write("Training: \n")
f.write('Number of model parameters = '+str(count_parameters(model))+'\n')

torch.autograd.set_detect_anomaly(True) 
tqdm_epoch = tqdm(range(epochs), desc=f"Training progress")
start = time.time()
loss_history = []
loss_test = []
best_loss = float('inf')
current_loss = float('inf')
best_epoch = 0

for k in tqdm_epoch:
    temp_loss = 0
    ct = 0
    for j,data in enumerate(train_loader):  # test fitting to a few samples on test_loader, trainloader always produce random generated set.
        optim.zero_grad()    
        xx, yy, tt = data
        xx, yy, tt = xx.to(device), yy.to(device), tt.to(device)
        xx = F.interpolate(xx, size=[SZ,SZ])
        yy = F.interpolate(yy, size=[SZ,SZ])
        xx = xx-1.0
        yy = yy-1.0

        qq = model(xx, tt)
        ycomp = qq #torch.relu(qq + xx[:,-1:,:,:])
           
        loss = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)     # misfit
        #breakpoint()
        loss.backward()
        optim.step()
        #scheduler.step()

        xx = xx + 1
        yy = yy + 1
        ycomp = ycomp + 1

        nMSE = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)
        nRMSE = (nMSE.item())**0.5

        tqdm_epoch.set_description('epochs = %3d.%3d   Loss(nMSE after translation) =  %3.2e  Training Avg nRMSE = %3.2e Best Epoch = %3d  Best Validation nRMSE = %3.2e'%(k,j,loss,current_loss,best_epoch, best_loss))

        f.write("Loss: "+str(nRMSE)+" s\n")
        temp_loss+=nRMSE
        ct = ct + 1

    loss_history.append(temp_loss/ct)
    current_loss = temp_loss/ct

    # test dataset for validation  # because testing misfit seemed higher than training misfit
    temp_loss1 = 0
    ct = 0 
    for j,data in enumerate(test_loader):
        if j>1:
            break
        xx, yy, tt = data
        xx, yy, tt = xx.to(device), yy.to(device), tt.to(device)
        xx = F.interpolate(xx, size=[SZ,SZ])
        yy = F.interpolate(yy, size=[SZ,SZ])
        xx = xx-1.0
        yy = yy-1.0
        with torch.no_grad():
            qq = model(xx, tt)

        ycomp = qq #torch.relu(qq + xx[:,-1:,:,:]) 
        ycomp = ycomp + 1
        xx = xx + 1
        yy = yy + 1
        loss1 = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)     # misfit or nMSE
        
        temp_loss1+=((loss1.item())**0.5)     # add nRMSE
        ct = ct + 1

    loss_test.append(temp_loss1/ct)
    if best_loss > (temp_loss1/ct):    # at least loss actually misfit for test dataset
        best_loss = (temp_loss1/ct)
        torch.save(model, "model-full.pth")
        best_epoch = k

    torch.save(model, "model-full-last.pth")

    
    plt.plot(np.array(loss_history), label='Training')
    plt.plot(np.array(loss_test),'r', label='Validation/Testing')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss (misfit)')
    plt.yscale("log")
    plt.title("Loss curve")
    plt.savefig("plots/Training_loss.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

stop = time.time()

print('Done Training\n')

f.write("Training Runtime: "+str(stop-start)+" s\n\n")


gc.collect()
torch.cuda.empty_cache()

#test_X = torch.load('/data/sid/KTH/X_test_1020.pt')
#test_Y = torch.load('/data/sid/KTH/Y_test_1020.pt')
#test = torch.utils.data.TensorDataset(test_X, test_Y)
#test_loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle = True)

f.write("\n\nTesting testing Single Shot: \n")
tqdm_epoch = tqdm((test_loader), desc=f"Direct Testing progress")
loss_history = []

MSE = 0
MAE = 0
SSIM = 0
RMSE = 0
nRMSE = 0
nMSE = 0
ssimT = ssim(data_range=2)
mae = nn.L1Loss()  # by default reduction = 'mean'

ct = 0
for j,data in enumerate(tqdm_epoch):
    xx, yy, tt = data
    xx, yy, tt = xx.to(device), yy.to(device), tt.to(device)
    xx = F.interpolate(xx, size=[SZ,SZ])
    yy = F.interpolate(yy, size=[SZ,SZ])
    xx = xx-1.0
    yy = yy-1.0
    with torch.no_grad():
        qq = model(xx, tt)

    ycomp = qq #torch.relu(qq + xx[:,-1:,:,:])

    xx = xx + 1.0
    yy = yy + 1.0
    ycomp = ycomp + 1.0

    loss = F.mse_loss(ycomp, yy)        # MSE
    mf_loss = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)     # misfit
    
    tqdm_epoch.set_description('step# = %3d   Loss(nMSE) =  %3.2e'%(j,mf_loss))
    
    f.write("Step:1"+str(j+1)+"\tmisfit: "+str(mf_loss.item())+"\tMSE: "+str(loss.item())+" \n")
        
    RMSE += (loss.item())**0.5
    nRMSE += (mf_loss.item())**0.5
    nMSE += mf_loss.item()
    MSE += loss.item()  # original image is scaled from 0-255
    MAE += mae(ycomp, yy).item()
    SSIM += ssimT(ycomp.cpu(), yy.cpu()).item()        #`preds` and `target` to have BxCxHxW or BxCx Depth xHxW
    # SSIM should give image wise averaged because tested on below code for SSIM(data_range=255)
    #S = 0
    #for b in range(16):   # 16 batches
    #   for t in range(10):  # 10 channels
    #       S += ssimT(j[b,t,:,:].unsqueeze(0).unsqueeze(0)*255,j[b,t,:,:].unsqueeze(0).unsqueeze(0)*255*0)
    # S/160 is same as ssimT(j,j*0)

    loss_history.append(mf_loss.item())
    
    ct = ct + 1

    # Testing plots
    if ct <= 10:
        s1 = int((torch.rand(1).item())*yy.shape[0])
        #s2 = int((torch.rand(1).item())*yy.shape[0])
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        c1 = axs[0, 0].imshow(yy[s1,0].cpu(), cmap='viridis')
        axs[0,0].set_title('First Frame Reference')
        axs[0, 0].axis("off")
        plt.colorbar(c1,ax = axs[0,0])
        c2 = axs[0,1].imshow(ycomp[s1,0].detach().cpu().numpy(), cmap='viridis')
        axs[0,1].set_title('First Frame Prediction')
        axs[0, 1].axis("off")
        plt.colorbar(c2,ax = axs[0,1])
        c3 = axs[0,2].imshow((yy.cpu()-ycomp.detach().cpu().numpy())[s1,0], cmap='viridis')
        axs[0,2].set_title('First Frame Error')
        axs[0, 2].axis("off")
        plt.colorbar(c3,ax = axs[0,2])  # colorbar and position
    
        c4 = axs[1, 0].imshow(yy[s1,-1].cpu(), cmap='viridis')
        axs[1, 0].set_title('Last Frame Reference')
        axs[1, 0].axis("off")
        plt.colorbar(c4,ax = axs[1, 0])
        c5 = axs[1, 1].imshow(ycomp[s1,-1].detach().cpu().numpy(), cmap='viridis')
        axs[1, 1].set_title('Last Frame Prediction')
        axs[1, 1].axis("off")
        plt.colorbar(c5,ax = axs[1, 1])
        c6 = axs[1, 2].imshow((yy.cpu()-ycomp.detach().cpu().numpy())[s1,-1], cmap='viridis')
        axs[1, 2].set_title('Last Frame Error')
        axs[1, 2].axis("off")
        plt.colorbar(c6,ax = axs[1, 2])  # colorbar and position
        plt.savefig("plots/SingleShotComparison"+str(ct)+".png")

# These are not the stadard calculations
PMSE = MSE/ct
RMSE = RMSE/ct
nRMSE = nRMSE/ct
nMSE = nMSE/ct
MSE = MSE*128*128/ct
MAE = MAE*128*128/ct
SSIM = SSIM/ct

print('Pixel level MSE: ',PMSE, '\tRMSE: ',RMSE, '\tnRMSE: ',nRMSE,'\tnMSE: ',nMSE)
print("Video Pred Setting: MSE: ",MSE,"\tMAE: ",MAE,"\tSSIM: ",SSIM)
f.write("\nDone\n")
    
f.write("Final Report\n")
f.write("PMSE: "+str(PMSE)+"RMSE: "+str(RMSE)+"\nnRMSE: "+str(nRMSE)+"\nnMSE: "+str(nMSE)+"\n")
f.write("Video PRediction Metrics: "+"MSE: "+str(MSE)+"\nMAE: "+str(MAE)+"\nSSIM: "+str(SSIM)+"\n")

plt.figure()
plt.plot(np.array(loss_history))
plt.xlabel('#forward steps| each step predicts '+str(prediction)+' time steps')
plt.ylabel('loss (misfit)')
plt.yscale("log")
plt.title("Testing loss curve")
plt.savefig("plots/Direct_Testing_loss.png")

print('Done Direct Testing')

f.close()
