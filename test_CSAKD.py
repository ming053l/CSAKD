import os
import time
import numpy as np
# from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
# from sklearn.metrics import log_loss
import random as rn
from utils import *
import torch
import numpy as np
from dataset import *
from scipy.io import savemat, loadmat
from trainOps import *
import tqdm
import time
import argparse

from models.KQD import Teacher , Student

# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 1
device = 'cuda'
MAX_EP = 16000
BANDS = 172
SIGMA = 0.0    ## Noise free -> SIGMA = 0.0

prefix='DCSN_joint2'
DEBUG=False


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--resume_ind', type=int, default=999)
    #parser.add_argument('--resume_ckpt', type=str, default="")
    parser.add_argument('--snr', type=int, default=0)
    
    
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=2)
    parser.add_argument('--finetuning_step', type=int, default=300, help='Works only if the mixed_align_opt is on')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate, 0 means training without weight decay')
    parser.add_argument('--joint_loss', type=int, default=1)
    parser.add_argument('--adaptive_fuse', type=int, default=3)
    parser.add_argument('--student_layers', type=list, default=[1,4,4,1])

    ## Data generator configuration
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bands', type=int, default=172)
    parser.add_argument('--msi_bands', type=int, default=4)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=86, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    # parser.add_argument("--root", type=str, default="/work/u1657859/DCSNv2/Data", help='data root folder')
    #parser.add_argument("--root", type=str, default="/work/u5088463/Data", help='data root folder')   
    parser.add_argument("--root", type=str, default="../../Fusion_data", help='data root folder')
    parser.add_argument("--test_file", type=str, default="./test.txt")   
    parser.add_argument("--prefix", type=str, default="KD_QRCODE_band4")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--test_mode", type=int, default=2 , help='1 : test teacher , 2 : test student , 3 : check FLOPS and Params' ) 
    
    
    args = parser.parse_args()

    return args

def awgn(x):
        snr = 10**(args.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower

def testing_teacher(args):
    ## Reading files #
    
    ## Load test files from specified text with SOURCE/TARGET name (you can replace it with other path u want)
    valfn = loadTxt(args.test_file)
    rn.shuffle(valfn)
    index = np.array(range(len(valfn)))
    rn.shuffle(index)
    print(f'#Testing samples is {len(valfn)}')

    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')

    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, drop_last = True, pin_memory=True, num_workers=args.workers)

    model = Teacher(args).to(args.device)
    
    if args.resume_ind>0:
        args.resume_ckpt = os.path.join('teacher_checkpoint', args.prefix, 'best.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict = torch.load(args.resume_ckpt)  
        
        for key in list(state_dict.keys()):
          if '_orig_mod.' in key:
              state_dict[key.replace('_orig_mod.', '')] = state_dict[key]
              del state_dict[key]
              
              
        model.load_state_dict(state_dict)
        print(f'Loading the pretrained teacher model from {args.resume_ckpt}')
        ## finetuneen(sams)

    model.eval().cuda()


    with torch.no_grad():
        
        for X in val_loader:
            if args.network_mode==2:
                x,x2,x3,vy,vfn, maxv, minv = X
                x3 = x3.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

            elif args.network_mode==1:
                x3,x2,vy,vfn, maxv, minv= X
                x2 = x2.to(device, non_blocking=True)
                x3 = x3.to(device, non_blocking=True)

            elif args.network_mode==0:
                x,vy,vfn, maxv, minv = X

            # x2 = awgn(x2) # LRHSI
            # x3 = awgn(x3) # HRMSI    
            # x = x.to(device, non_blocking=True)
            
            if args.network_mode==2:
                val_dec = model(x, LRHSI=x2, HRMSI=x3, mode=1)

            elif args.network_mode==1:
                val_dec, _ = model(LRHSI=x2, HRMSI=x3, mode=1)

            elif args.network_mode==0:
                val_dec = model(x, LRHSI=None, HRMSI=None, mode=1)
            
            break
            
            
        rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
        ep = 0.0

        for batch_idx, (X) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            
            args.network_mode = 1
            if args.network_mode==2:
                x,x2,x3,vy,vfn, maxv, minv = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                x3,x2,vy,vfn, maxv, minv= X
                x2 = x2.cuda()
                x3 = x3.cuda()
            elif args.network_mode==0:
                x,vy,vfn, maxv, minv = X
            #x = x.cuda()
            
            # x2 = awgn(x2) # LRHSI
            # x3 = awgn(x3) # HRMSI  

            start_time = time.time()
            if args.network_mode==2:
                val_dec = model(x, LRHSI=x2, HRMSI=x3, mode=1)
            elif args.network_mode==1:
                val_dec, _ = model(LRHSI=x2, HRMSI=x3, mode=1)
            elif args.network_mode==0:
                val_dec = model(x, LRHSI=None, HRMSI=None, mode=1)
            ep = ep + (time.time()-float(start_time))

            val_dec = val_dec.cpu().numpy()
            vy = vy.cpu().numpy()

            
            for predimg, gtimg,f, v1, v2 in zip(val_dec, vy, vfn, maxv, minv):
                predimg = (predimg/2+0.5) 
                gtimg = (gtimg/2+0.5) 

                sams.append(sam2(predimg, gtimg))
                psnrs.append(psnr(predimg, gtimg))
                ergas.append(ERGAS(predimg, gtimg))
                savemat('test_image/teacher/'+os.path.basename(f)+'.mat', {'pred':np.transpose(predimg,(1,2,0))})
#                 predimg = predimg * (v1-v2) + v2
#                 gtimg = gtimg * (v1-v2) + v2
                rmses.append(rmse(predimg, gtimg, maxv=v1, minv=v2))
        
        ep = ep / len(sams)

        print('T : val-PSNR: %.3f, val-SAM: %.3f, val-ERGAS: %.3f, val-rmse: %.3f, AVG-Time: %f ms on %d misaligned pix' %
              (np.mean(psnrs), np.mean(sams), np.mean(ergas), np.mean(rmses), ep*1000.0, args.mis_pix))
        
        # print('T : AVG-Time: %f ms on %d misaligned pix' %
        #       (ep*1000.0, args.mis_pix))

def testing_student(args):
    ## Reading files #
    
    ## Load test files from specified text with SOURCE/TARGET name (you can replace it with other path u want)
    valfn = loadTxt(args.test_file)
    rn.shuffle(valfn)
    index = np.array(range(len(valfn)))
    rn.shuffle(index)
    print(f'#Testing samples is {len(valfn)}')

    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')

    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, drop_last = True, pin_memory=True, num_workers=args.workers)
    
    model_s = Student(args).to(args.device)
    
    if args.resume_ind>0:
        args.resume_ckpt = os.path.join('student_checkpoint', args.prefix, 'best.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict_s = torch.load(args.resume_ckpt)  
        
        for key in list(state_dict_s.keys()):
          if '_orig_mod.' in key:
              state_dict_s[key.replace('_orig_mod.', '')] = state_dict_s[key]
              del state_dict_s[key]
        
        model_s.load_state_dict(state_dict_s)
        print(f'Loading the pretrained student model from {args.resume_ckpt}')
        ## finetune

    model_s.eval().cuda()

    with torch.no_grad():
        
        for X in val_loader:
            if args.network_mode==2:
                x,x2,x3,vy,vfn, maxv, minv = X
                x3 = x3.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

            elif args.network_mode==1:
                x3,x2,vy,vfn, maxv, minv= X
                x2 = x2.to(device, non_blocking=True)
                x3 = x3.to(device, non_blocking=True)

            elif args.network_mode==0:
                x,vy,vfn, maxv, minv = X

            # x2 = awgn(x2) # LRHSI
            # x3 = awgn(x3) # HRMSI    
            # x = x.to(device, non_blocking=True)
            
        rmses_s, sams_s, fnames_s, psnrs_s, ergas_s = [], [], [], [], []
        ep_s = 0.0

        for batch_idx, (X) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            
            if args.network_mode==2:
                x,x2,x3,vy,vfn, maxv, minv = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                x3,x2,vy,vfn, maxv, minv= X
                x2 = x2.cuda()
                x3 = x3.cuda()
            elif args.network_mode==0:
                x,vy,vfn, maxv, minv = X
            #x = x.cuda()
            
            # x2 = awgn(x2) # LRHSI
            # x3 = awgn(x3) # HRMSI  

            start_time = time.time()
            if args.network_mode==2:
                val_dec_s = model_s(x, LRHSI=x2, HRMSI=x3, mode=1)
            elif args.network_mode==1:
                val_dec_s, _= model_s(LRHSI=x2, HRMSI=x3, mode=1)
            elif args.network_mode==0:
                val_dec_s = model_s(x, LRHSI=None, HRMSI=None, mode=1)
            ep_s = ep_s + (time.time()-float(start_time))
            
            val_dec_s = val_dec_s.cpu().numpy()
            vy = vy.cpu().numpy()

            for predimg_s, gtimg_s,f, v1, v2 in zip(val_dec_s, vy, vfn, maxv, minv):
                predimg_s = (predimg_s/2+0.5) 
                gtimg_s = (gtimg_s/2+0.5) 

                sams_s.append(sam2(predimg_s, gtimg_s))
                psnrs_s.append(psnr(predimg_s, gtimg_s))
                ergas_s.append(ERGAS(predimg_s, gtimg_s))
                savemat('test_image/student/'+os.path.basename(f)+'.mat', {'pred':np.transpose(predimg_s,(1,2,0))})
#                 predimg = predimg * (v1-v2) + v2
#                 gtimg = gtimg * (v1-v2) + v2
                rmses_s.append(rmse(predimg_s, gtimg_s, maxv=v1, minv=v2))

        ep_s = ep_s / len(sams_s)

        print('S : val-PSNR: %.3f, val-SAM: %.3f, val-ERGAS: %.3f, val-rmse: %.3f, AVG-Time: %f ms on %d misaligned pix' %
              (np.mean(psnrs_s), np.mean(sams_s), np.mean(ergas_s), np.mean(rmses_s), ep_s*1000.0, args.mis_pix))
        # print('S : AVG-Time: %f ms on %d misaligned pix' %
        #       (ep_s*1000.0, args.mis_pix))


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.SEED)
    rn.seed(args.SEED)
    np.random.seed(args.SEED)

    print("#"*80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#"*80)
    print('start testing')
    print("#"*80)

    if args.test_mode == 1:
      print('testing teacher model')
      print("#"*80)
      testing_teacher(args)
    if args.test_mode == 2:
      print('testing student model')
      print("#"*80)
      testing_student(args)

    print("#"*80)
    
    if args.test_mode == 3:
      from thop import profile
      
      args = parse_args()
      m1=Student(args)
      m2=Teacher(args)
      
      input1 = torch.randn(1, 4, 256, 256) 
      input2 = torch.randn(1, 172, 64,64)           
      flops, params = profile(m1, inputs=(input2,input1, ))
      print('FLOPs = ' + str(flops/1000**3) + 'G')
      print('Params = ' + str(params/1000**2) + 'M')
      
      flops, params = profile(m2, inputs=(input2,input1, ))
      print('FLOPs = ' + str(flops/1000**3) + 'G')
      print('Params = ' + str(params/1000**2) + 'M')
      
    print('finish')
