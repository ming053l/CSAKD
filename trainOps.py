import numpy as np
from scipy.io import savemat, loadmat
import torch
import math

# normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;

def normColor(R):
#     import pdb
#     pdb.set_trace()
    R = R[[25,15,6],:,:]
    
    R = (R-np.mean(R)) / np.std(R)
    R = np.minimum(R, 2)
    R = np.maximum(R, -2)/3+0.5
    R = np.clip(R*255, 0,255)
    R= np.transpose(R, (1,2,0))
    return R.astype(np.uint8)


def sam2(x, y):
    num = np.sum(np.multiply(x, y), 0)
    den = np.sqrt(np.multiply(np.sum(x**2, 0), np.sum(y**2, 0)))
    sam = np.sum(np.degrees(np.arccos(num / den))) / (x.shape[2]*x.shape[1])
    return sam

def psnr(x,y):

    bands = x.shape[0]
    x = np.reshape(x, [bands,-1])
    y = np.reshape(y, [bands,-1])
    msr = np.mean((x-y)**2, 1)
    maxval = np.max(y, 1)**2
    return np.mean(10*np.log10(maxval/msr))

def ERGAS(x, y, Resize_fact=4):

    err = y-x
    ergas=0
    for i in range(y.shape[0]):
        ergas += np.mean(np.power(err[i],2)) / np.mean(y[i])**2
    ergas = (100.0/Resize_fact) * np.sqrt(1.0/y.shape[0] * ergas)
    return ergas

def lmat(fn):
    x=np.load(fn,allow_pickle=True)
    #x=loadmat(fn)
    #x=x[list(x.keys())[-1]]
   
    return x
        
def loadTxt(fn):
    a = []
    with open(fn, 'r') as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip('\n')
            a.append(fn)
    return a

def rmse(x, y, maxv=1, minv=0):

    rmse_total = np.sqrt(np.mean(np.power(x-y, 2)))
    rmse_total = rmse_total* (maxv-minv) + minv
    return rmse_total

def awgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/x.numel()
    npower = torch.sqrt(xpower / snr)
    return x + torch.randn(x.shape).cuda() * npower

def BandWiseMSE(x, y, sigma=1, reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    yp = torch.mean(y**2+0.1, (2,3))
    yp = torch.where(yp == 0, torch.ones_like(yp) * 1e-8, yp)  # replace zeros in yp
    diff = torch.abs(x - y)
    loss = 0.5 * diff ** 2 / beta + torch.relu(diff - beta) - 0.5 * beta
    loss = torch.mean(loss, (2,3)) / yp

    if reduce:
        return torch.mean(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer


class BandWiseMSEv1(torch.nn.Module):
    
    def forward(self, x,y, sigma=1, reduce=True, normalizer=1.0):
        yp = torch.sqrt(torch.sum(y**2, (2,3))) / (y.shape[2]*y.shape[3])+1e-9
        # print(yp)
        yp = torch.nn.functional.normalize(1/yp)
        # print(yp)
        loss = (x - y)**2
        loss = torch.sqrt(torch.mean(loss, (2,3))) * yp 
        if reduce:
            return torch.mean(loss) 

        return torch.mean(loss, dim=1)

def sam_lossv1(x, y):
    num = torch.sum(torch.multiply(x, y), 1)
    den = torch.sqrt(torch.multiply(torch.sum(x**2+1e-9, 1), torch.sum(y**2+1e-9, 1)))
    sam = torch.clip(torch.divide(num, den), -1, 1)
    sam = torch.mean(torch.arccos(sam))
    return sam
    
class SAMLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SAMLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        # Reshape the tensors to shape [B*H*W, C]
        x_flat = x.reshape(-1, x.shape[1])
        y_flat = y.reshape(-1, y.shape[1])

        # Compute the cosine similarity
        num = torch.sum(x_flat * y_flat, dim=1)
        den = torch.norm(x_flat, dim=1) * torch.norm(y_flat, dim=1) + self.epsilon

        # Compute SAM loss as 1 minus average cosine similarity
        sam_loss = 1 - torch.mean(num / den)

        return sam_loss

import torch
import torch.nn as nn

class BPLoss(nn.Module):
    def __init__(self):
        super(BPLoss, self).__init__()

    def forward(self, X, X_star):
        """
        Compute the BP-Loss between the prediction X and ground truth X_star using matrix operations.

        Parameters:
        X (torch.Tensor): Predicted hyperspectral image.
        X_star (torch.Tensor): Ground truth hyperspectral image.

        Returns:
        torch.Tensor: Calculated BP-Loss.
        """
        X, X_star = X/torch.max(X), X_star/torch.max(X_star)
        # Compute the norm of each channel in X
        norms = torch.norm(X.view(X.shape[0], X.shape[1], -1), dim=2, p=2)

        # Compute p_m for each channel
        p_m = norms 

        # Compute the squared L2 norm for each channel and divide by p_m
        loss = torch.mean(torch.mean((torch.norm((X - X_star).view(X.shape[0], X.shape[1], -1), dim=2, p=2) ) / p_m, dim=1))

        return loss




def sam_loss(x, y):
    num = torch.sum(torch.multiply(x, y), 1)
    den = torch.sqrt(torch.multiply(torch.sum(x**2, 1), torch.sum(y**2, 1)))
    mask = den != 0  # create a mask of non-zero elements
    num = num[mask]  # apply the mask to num
    den = den[mask]  # apply the mask to den
    cos_sim = torch.clip(torch.divide(num, den), -1, 1)
    approximate_sam = 1-torch.mean(cos_sim)
    return approximate_sam

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']