import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
import cv2
from PIL import Image
# from scipy.io import loadmat

def getAffine(src, y, pmax = 0.08, pmin=0.92, angle_max = 180, scale_min=0.9, scale_max=1.1):
   
        
        srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
        dstTri = np.array( [[0, src.shape[1]*rn.uniform(0, pmax)], 
                            [src.shape[1]*rn.uniform(pmin, 1), src.shape[0]*rn.uniform(0, pmax)], 
                            [src.shape[1]*rn.uniform(pmin, 1), src.shape[0]*rn.uniform(0, pmax)]] ).astype(np.float32)
        # Rotating the image after Warp
        angle = rn.randint(0, angle_max)
        scale = rn.uniform(scale_min, scale_max)
        
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
        center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)

        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
        
        warp_dst = cv2.warpAffine(y, warp_mat, (src.shape[1], src.shape[0]))
        warp_rotate_dst2 = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
        return warp_rotate_dst, warp_rotate_dst2

class dataset_h5(torch.utils.data.Dataset):
    
    def __init__(self, X, args, mode='train'):

        super(dataset_h5, self).__init__()

        self.root       = args.root
        self.fns        = X
        self.n_images   = len(self.fns)
        self.indices    = np.array(range(self.n_images))
        self.offset     = (args.image_size- args.crop_size)
        self.offsetB    = (args.bands- args.crop_size)
        self.mis_pix    = args.mis_pix
        self.msi        = args.msi_bands
        self.mixed      = args.mixed_align_opt==1

        self.mode       = mode.lower()
        self.crop_size  = args.crop_size
        self.img_size   = args.image_size
        
        self.offsets = []
        #print(f'Found {len(X)} image in {mode} mode!')
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])
        y = np.load(fn+'GT.npz.npy').astype(np.float32)
        y = y[:256,:256, :]
        
        x = np.load(fn+f'co{self.msi}{self.mis_pix+1}.npz.npy').astype(np.float32)
        assert x.shape==(self.img_size,self.img_size,172)     
        assert y.shape==(self.img_size,self.img_size,172)
        maxv = y.max()
        minv = y.min()
        y = (y-minv) / (maxv-minv)         
        y = (y-0.5) * 2
        x = (x-0.5) * 2
        
        if self.mode=='train':
            
            # Random crop
            xim, yim = rn.randint(0, self.offset), rn.randint(0, self.offset)
                
            h = yim + self.crop_size
            w = xim + self.crop_size
            
            x = x[yim:h, xim:w,:]
            y = y[yim:h, xim:w,:]
            
            if rn.random()>=0.5:
                x = np.flip(x, 1).copy()
                y = np.flip(y, 1).copy()
            if rn.random()>=0.5:
                x = np.flip(x, 0).copy()
                y = np.flip(y, 0).copy()
            
            if rn.random()>=0.5:
                times = rn.randint(1,3)
                x = np.rot90(x, times).copy()
                y = np.rot90(y, times).copy()
                
        
        x=np.transpose(x, (2,0,1))
        y=np.transpose(y, (2,0,1))
        
        return x, y, fn, maxv, minv

    def __len__(self):
        return self.n_images
    
class dataset_joint(torch.utils.data.Dataset):
    
    hrmsi = ['hrmsi4', 'hrmsi6']
    lrhsi = ['lrhsi', 'lrhsi1', 'lrhsi2', 'lrhsi3']
    
    def __init__(self, X, args, mode='train'):

        super(dataset_joint, self).__init__()

        self.root       = args.root
        self.fns        = X
        self.n_images   = len(self.fns)
        self.indices    = np.array(range(self.n_images))
        self.offset     = (args.image_size- args.crop_size)
        self.offsetB    = (args.bands- args.crop_size)
        self.mis_pix    = args.mis_pix
        self.msi        = args.msi_bands
        self.mixed      = args.mixed_align_opt==1

        self.mode       = mode.lower()
        self.crop_size  = args.crop_size
        self.img_size   = args.image_size
        
        self.offsets = []
        #print(f'Found {len(X)} image in {mode} mode!')
        for i in range(0,self.offset, 4):
            self.offsets.append(i)
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])
        
        x1key, x2key = f'hrmsi{self.msi}', 'lrhsi'
        if self.mixed and self.mode=='train':
            x2key = rn.choice(self.lrhsi)
        
        elif self.mis_pix > 0:
            x2key = eval(f"'lrhsi{self.mis_pix}'") if self.mis_pix>0 else "lrhsi"
            
        x = np.load(fn+'hrmsi.npz')[x1key].astype(np.float32)
        x2= np.load(fn+'lrhsi.npz')[x2key].astype(np.float32)
        y   = np.load(fn+'GT.npz.npy').astype(np.float32)
        y = y[:self.img_size,:self.img_size, :]
        assert y.shape==(self.img_size,self.img_size,172)
        maxv = y.max()
        minv = y.min()
        y = (y-minv) / (maxv-minv)         
        y = (y-0.5) * 2
        x = (x-0.5) * 2
        x2 = (x2-0.5) *2
        if self.mode=='train':
                        
            # Random crop
            xim, yim = rn.choice(self.offsets), rn.choice(self.offsets)
            
            h = yim + self.crop_size
            w = xim + self.crop_size
            h2 = yim//4 + self.crop_size//4
            w2 = xim//4 + self.crop_size//4
            
            
            x = x[yim:h, xim:w,:]
            x2 = x2[yim//4:h2, xim//4:w2,:]
            y = y[yim:h, xim:w,:]
            
            if rn.random()>=0.5:
                x = np.flip(x, 1).copy()
                x2 = np.flip(x2, 1).copy()
                y = np.flip(y, 1).copy()
            if rn.random()>=0.5:
                x = np.flip(x, 0).copy()
                x2 = np.flip(x2, 0).copy()
                y = np.flip(y, 0).copy()
                
            if rn.random()>=0.5:
                times = rn.randint(1,3)
                x =  np.rot90(x,  times).copy()
                x2 = np.rot90(x2, times).copy()
                y = np.rot90(y, times).copy()
                
        
        x=np.transpose(x, (2,0,1))
        x2=np.transpose(x2, (2,0,1))
        y=np.transpose(y, (2,0,1))
        return x, x2, y, fn, maxv, minv

    def __len__(self):
        return self.n_images
    
class dataset_joint2(torch.utils.data.Dataset):
    
    hrmsi = ['hrmsi4', 'hrmsi6']
    lrhsi = ['lrhsi', 'lrhsi1', 'lrhsi2', 'lrhsi3']
    # co4 = ['co41', 'co42', 'co43', 'co44']
    # co6 = ['co61', 'co62', 'co63', 'co64']
    
    def __init__(self, X, args, mode='train'):

        super(dataset_joint2, self).__init__()

        self.root       = args.root
        self.fns        = X
        self.n_images   = len(self.fns)
        self.indices    = np.array(range(self.n_images))
        self.offset     = (args.image_size- args.crop_size)
        self.offsetB    = (args.bands- args.crop_size)
        self.mis_pix    = args.mis_pix
        self.msi        = args.msi_bands
        self.mixed      = args.mixed_align_opt==1

        self.mode       = mode.lower()
        self.crop_size  = args.crop_size
        self.img_size   = args.image_size
        
        self.offsets = []
        #print(f'Found {len(X)} image in {mode} mode!')
        for i in range(0,self.offset, 4):
            self.offsets.append(i)
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])
        
        x1key, x2key = f'hrmsi{self.msi}', 'lrhsi'
        if self.mixed and self.mode=='train':
            rind = rn.randint(0,3)
            x2key = self.lrhsi[rind]
            #cokey = self.co4[rind] if self.msi==4 else self.co6[rind]
        
        elif self.mis_pix > 0:
            x2key = f'lrhsi{self.mis_pix}' if self.mis_pix>0 else "lrhsi"
            #cokey = f'co{self.msi}{self.mis_pix+1}'
            
        #x = np.load(fn+f'{cokey}.npz.npy').astype(np.float32)
        x2 = np.load(fn+'lrhsi.npz')[x2key].astype(np.float32)
        x3 = np.load(fn+'hrmsi.npz')[x1key].astype(np.float32)
        
        y   = np.load(fn+'GT.npz.npy').astype(np.float32)
        y = y[:self.img_size,:self.img_size, :]
        assert y.shape==(self.img_size,self.img_size,172)
        maxv = y.max()
        minv = y.min()
        y = (y-minv) / (maxv-minv)         
        y =  (y -0.5) * 2
        #x =  (x -0.5) * 2
        x2 = (x2-0.5) * 2
        x3 = (x3-0.5) * 2
        
#         if self.mode=='train':
                        
#             # Random crop
#             xim, yim = rn.choice(self.offsets), rn.choice(self.offsets)
            
#             h = yim + self.crop_size
#             w = xim + self.crop_size
#             h2 = yim//4 + self.crop_size//4
#             w2 = xim//4 + self.crop_size//4
            
            
#             #x = x[yim:h, xim:w,:]
#             x3 = x3[yim:h, xim:w,:]
#             x2 = x2[yim//4:h2, xim//4:w2,:]
#             y = y[yim:h, xim:w,:]
            
#             if rn.random()>=0.5:
#                 #x = np.flip(x, 1).copy()
#                 x2 = np.flip(x2, 1).copy()
#                 x3 = np.flip(x3, 1).copy()
#                 y = np.flip(y, 1).copy()
#             if rn.random()>=0.5:
#                 #x = np.flip(x, 0).copy()
#                 x2 = np.flip(x2, 0).copy()
#                 x3 = np.flip(x3, 0).copy()
#                 y = np.flip(y, 0).copy()
                
#             if rn.random()>=0.5:
#                 times = rn.randint(1,3)
#                 #x =  np.rot90(x,  times).copy()
#                 x2 = np.rot90(x2, times).copy()
#                 x3 = np.rot90(x3, times).copy()
#                 y = np.rot90(y, times).copy()
                
        
        #x=np.transpose(x, (2,0,1))
        x2=np.transpose(x2, (2,0,1))
        x3=np.transpose(x3, (2,0,1))
        y=np.transpose(y, (2,0,1))
        return  x2, x3, y, fn, maxv, minv

    def __len__(self):
        return self.n_images