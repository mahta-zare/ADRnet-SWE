# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np


class SWEDatasetOld(Dataset):
    def __init__(self,
                 initial_step=10,
                 saved_folder='/data/sid/PDEBench/shallow-water/',
                 if_test=False, test_ratio=0.1
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(saved_folder + "2D_rdb_NA_NA.h5")
        
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1-test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
        
        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
        
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)       
            
            history = hist
            prediction = pred
            X = torch.zeros(0, history, 128, 128)
            Y = torch.zeros(0, prediction, 128, 128)
            T = torch.zeros(0)
            tmax = data.shape[0]
            #data = data.permute(0, 3, 1,2)
            data = data[:,:,:,0]
            
            for t in range(history, tmax-prediction+1):
                xx = data[t-history:t, :, :]
                yy = data[t:t+prediction, :, :]
                tt = t*torch.ones(1, dtype=torch.float32)
                
                X = torch.cat((X, xx.unsqueeze(0)))
                Y = torch.cat((Y, yy.unsqueeze(0)))
                T = torch.cat((T, tt))
        
        return X, Y, T


class SWEDataset(Dataset):
    def __init__(self, data_loc='swe_train_data.pt'):
        """
        load the swe_test_data.pt or swe_train_data.pt
        """
        data = torch.load(data_loc)
        # Define path to files
        self.X = data['X']
        self.Y = data['Y']
        self.T = data['T']
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        
        X = self.X[idx]
        Y = self.Y[idx]
        T = self.T[idx]
            
        return X, Y, T



Generate = False
What = 'train' 
#What = 'test'
if Generate:
    # Convert the data to a .pt file for faster running
    hist = 10
    pred = 50
    if What == 'train':
        train_data = SWEDatasetOld(test_ratio=0.1)
        loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    elif What == 'test':
        test_data = SWEDatasetOld(if_test=True,test_ratio=0.1)
        loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False) 
    
    for data in loader:
        xx, yy, tt = data
        break
    
    Ll = len(loader)
    Nt= xx.shape[1]

    X = torch.zeros(Ll*Nt, hist, 128, 128)
    Y = torch.zeros(Ll*Nt, pred, 128, 128)
    T = torch.zeros(Ll*Nt)
                
    c = 0
    for j,data in enumerate(tqdm(loader)):  # test fitting to a few samples on test_loader, trainloader always produce random generated set.
        xx, yy, tt = data
        xx = xx.reshape(xx.shape[0]*xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4])
        yy = yy.reshape(yy.shape[0]*yy.shape[1], yy.shape[2], yy.shape[3], yy.shape[4])
        tt = tt.reshape(tt.shape[0]*tt.shape[1])

        X[c:c+Nt,:,:,:] = xx
        Y[c:c+Nt,:,:,:] = yy
        T[c:c+Nt] = tt
        c = c+Nt
        #print(j)

    file_path = '/gladwell/ndj376/ADRnet/SWE/swe'+str(hist)+'_'+str(pred)+'_'+What+'_data.pt'
    torch.save({'X': X, 'Y': Y, 'T': T}, file_path)
    print('done')
    print('Successfully generated ',What,' at ',file_path)
