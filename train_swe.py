## https://github.com/vincent-leguen/PhyDNet/blob/master/data/moving_mnist.py

import numpy as np
import os
import random
import gc
import yaml
import argparse

import hydra
from omegaconf import DictConfig
import logging


from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam

import sys
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt

from adrNet import *  
from sweDataLoader import *


@hydra.main(config_path="config", config_name="configs", version_base=None)
def main(cfg: DictConfig):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #########################

    # Determine the device
    if cfg.compute.accelerator == "gpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.compute.cuda_visible_devices}")
    else:
        device = torch.device("cpu")
    print('Device: ',device)
    

    plots_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)



    batch_size = 64
    test_ratio = 0.1
    num_workers = 4
    history = 10
    prediction = 1

    train_data = SWEDataset('/gladwell/ndj376/ADRnet/SWE/swe'+str(history)+'_'+str(prediction)+'_train_data.pt')
    test_data = SWEDataset('/gladwell/ndj376/ADRnet/SWE/swe'+str(history)+'_'+str(prediction)+'_test_data.pt')


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) 


    total_len = len(train_loader)  

    in_c = history
    SZ = 64
    Mask = torch.ones(SZ, SZ)
    model = resnet(cfg, in_c=history, hid_c = 2, out_c=prediction, nlayers=1, imsz=[SZ, SZ])
    model.to(device)

    print('Number of model parameters = %3d'%(count_parameters(model)))

    lr = cfg.trainer.learning_rate
    optim = Adam(model.parameters(), lr)
    epochs = cfg.trainer.num_epochs

    f = open(os.path.join(cfg.output_dir, "training_loss.txt"), "w")
    
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
        start_epoch = time.time()
        temp_loss = 0
        counter = 0
        for j,data in enumerate(train_loader):  # test fitting to a few samples on test_loader, trainloader always produce random generated set.
            optim.zero_grad()    
            xx, yy, tt = data
            xx, yy, tt = xx.to(device), yy.to(device), tt.to(device)
            xx = F.interpolate(xx, size=[SZ,SZ])
            yy = F.interpolate(yy, size=[SZ,SZ])
            xx = xx - 1.0
            yy = yy - 1.0

            qq = model(xx, tt)
            ycomp = qq #torch.relu(qq + xx[:,-1:,:,:])
            
            loss = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)
    
            loss.backward()
            optim.step()

            xx = xx + 1
            yy = yy + 1
            ycomp = ycomp + 1

            nMSE = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)
            nRMSE = (nMSE.item())**0.5

            # tqdm_epoch.set_description('epochs = %3d.%3d   Loss(nMSE after translation) =  %3.2e  Training Avg nRMSE = %3.2e Best Epoch = %3d  Best Validation nRMSE = %3.2e'%(k,j,loss,current_loss,best_epoch, best_loss))
        
            f.write(str(nRMSE)+"\n")
            temp_loss+=nRMSE
            counter = counter + 1


        loss_history.append(temp_loss/counter)
        current_loss = temp_loss/counter

        # validation 
        temp_loss1 = 0
        counter = 0 
        for j,data in enumerate(test_loader):
            if j>1:
                break
            xx, yy, tt = data
            xx, yy, tt = xx.to(device), yy.to(device), tt.to(device)
            xx = F.interpolate(xx, size=[SZ,SZ])
            yy = F.interpolate(yy, size=[SZ,SZ])
            xx = xx - 1.0
            yy = yy - 1.0

            with torch.no_grad():
                qq = model(xx, tt)

            ycomp = qq #torch.relu(qq + xx[:,-1:,:,:]) 
            ycomp = ycomp + 1
            xx = xx + 1
            yy = yy + 1
            loss1 = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)     # nMSE as loss
            
            temp_loss1+=((loss1.item())**0.5)     # add nRMSE
            counter = counter + 1

        loss_test.append(temp_loss1/counter)

        end_epoch = time.time()
        elapsed_time = end_epoch - start_epoch

        if best_loss > (temp_loss1/counter):    # at least loss actually misfit for test dataset
            best_loss = (temp_loss1/counter)
            torch.save(model, os.path.join(cfg.output_dir, "model-best.pth"))
            # torch.save(model, cfg.output_dir + "model-full.pth")
            best_epoch = k

        torch.save(model, os.path.join(cfg.output_dir, "model-last.pth"))
        print(
                    f"Epoch {k:4d} | "
                    f"Train Loss: {current_loss:.6f} | "
                    f"Val Loss: {temp_loss1/counter:.6f} | "
                    # f"LR: {current_lr:.2e} | "
                    f"Elapsed time: {elapsed_time:.4f}s",
                    flush=True,
                )

        
        plt.plot(np.array(loss_history), label='Training')
        plt.plot(np.array(loss_test),'r', label='Validation/Testing')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss (misfit)')
        plt.yscale("log")
        plt.title("Loss curve")
        plt.savefig(os.path.join(plots_dir, "Training_loss.png"))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

    stop = time.time()

    print('Done Training\n')

    f.write("Training Runtime: "+str(stop-start)+" s\n\n")


    gc.collect()
    torch.cuda.empty_cache()

    # f.write("\n\nTesting testing Single Shot: \n")
    tqdm_epoch = tqdm((test_loader), desc=f"Testing progress")
    loss_history = []

    MSE = 0
    MAE = 0
    SSIM = 0
    RMSE = 0
    nRMSE = 0
    nMSE = 0
    ssimT = ssim(data_range=2)
    mae = nn.L1Loss()  # by default reduction = 'mean'

    counter = 0
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
        mf_loss = F.mse_loss(ycomp, yy)/F.mse_loss(yy*0, yy)   
        
        # tqdm_epoch.set_description('step# = %3d   Loss(nMSE) =  %3.2e'%(j,mf_loss))
        
        # f.write("Step:1"+str(j+1)+"\tmisfit: "+str(mf_loss.item())+"\tMSE: "+str(loss.item())+" \n")
            
        RMSE += (loss.item())**0.5
        nRMSE += (mf_loss.item())**0.5
        nMSE += mf_loss.item()
        MSE += loss.item()  # original image is scaled from 0-255
        MAE += mae(ycomp, yy).item()
        SSIM += ssimT(ycomp.cpu(), yy.cpu()).item()        #`preds` and `target` to have BxCxHxW or BxCx Depth xHxW

        loss_history.append(mf_loss.item())
        
        counter = counter + 1

        # Testing plots
        if counter <= 10:
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
            c3 = axs[0,2].imshow((yy.detach().cpu().numpy()-ycomp.detach().cpu().numpy())[s1,0], cmap='viridis')
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
            c6 = axs[1, 2].imshow((yy.detach().cpu().numpy()-ycomp.detach().cpu().numpy())[s1,-1], cmap='viridis')
            axs[1, 2].set_title('Last Frame Error')
            axs[1, 2].axis("off")
            plt.colorbar(c6,ax = axs[1, 2])  # colorbar and position
            plt.savefig(os.path.join(plots_dir, "SingleShotComparison"+str(counter)+".png"))
            

    # These are not the stadard calculations
    PMSE = MSE/counter
    RMSE = RMSE/counter
    nRMSE = nRMSE/counter
    nMSE = nMSE/counter
    MSE = MSE*128*128/counter
    MAE = MAE*128*128/counter
    SSIM = SSIM/counter

    # Pixel level metrics
    print('PMSE: ',PMSE, '\nRMSE: ',RMSE, '\nnRMSE: ',nRMSE,'\nnMSE: ',nMSE)
    # print("\n")
    # Video prediction metrics
    print("MSE: ",MSE,"\nMAE: ", MAE, "\nSSIM: ",SSIM)

    # f.write("\nDone\n")
    # f.write("Final Report\n")
    # f.write("PMSE: "+str(PMSE)+"\nRMSE: "+str(RMSE)+"\nnRMSE: "+str(nRMSE)+"\nnMSE: "+str(nMSE)+"\n")
    # f.write("\n")
    # f.write("Video Prediction Metrics: " + "\nMSE: "+str(MSE)+"\nMAE: "+str(MAE)+"\nSSIM: "+str(SSIM)+"\n")

    plt.figure()
    plt.plot(np.array(loss_history))
    plt.xlabel('#forward steps| each step predicts '+str(prediction)+' time steps')
    plt.ylabel('loss')
    plt.yscale("log")
    plt.title("Testing loss curve")
    plt.savefig(os.path.join(plots_dir, "Direct_Testing_loss.png"))

    print('Done Testing')

    f.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

