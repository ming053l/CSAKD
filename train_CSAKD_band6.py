import os, argparse, glob, time, cv2, pdb, torch, tqdm, models
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random as rn
from scipy.io import savemat, loadmat
from utils import *
from dataset import *
from math import acos, degrees
from trainOps import *
from tensorboardX import SummaryWriter 


# from models.dcsn3d import DCSNMain as DCSN
from models.KQD import Teacher , Student

torch.backends.cudnn.benchmark=False

def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--resume_ind', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default="")
    parser.add_argument('--snr', type=int, default=35)
    
    parser.add_argument('--lr', type=float, default=1e-4)
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
    parser.add_argument('--msi_bands', type=int, default=6)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=86, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    parser.add_argument("--root", type=str, default="../../Fusion_data", help='data root folder')
    # parser.add_argument("--root", type=str, default="/work/u5088463/Data", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="./val.txt")   
    parser.add_argument("--train_file", type=str, default="./train.txt")   
    parser.add_argument("--prefix", type=str, default="KD_4bn_Band6_retrain")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args

def trainer(args):

    # ===============Prepare Data===============

    flist = loadTxt(args.train_file)
    valfn = loadTxt(args.val_file)
    print(f'#training samples is {len(flist)} and validation samples is {len(valfn)}')

    writer = SummaryWriter(f'runs/exp-{args.prefix}')

    dataset = dataset_joint
        #print('Use pairwise (LRHSI+HRMSI) dataset')


    train_loader = torch.utils.data.DataLoader(dataset(flist, args), batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.workers)

    model_t = Teacher(args).to(args.device)
    model = Student(args).to(args.device)

    if args.resume_ind>0 or os.path.isfile(args.resume_ckpt):
        if not os.path.isfile(args.resume_ckpt):
            args.resume_ckpt = os.path.join('teacher_checkpoint', args.prefix, 'best.pth')
            resume_ckpt = os.path.join('student_checkpoint', args.prefix, 'last.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict = torch.load(args.resume_ckpt)  
        model_t.load_state_dict(state_dict)
        state_dict = torch.load(resume_ckpt)  
        model.load_state_dict(state_dict)
        print(f'Loading the pretrained model from {args.resume_ckpt}')
    
    model_t.train()
    model.train()

    optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    if args.lr_scheduler=='cosine':
        scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, args.epochs-args.resume_ind, eta_min=1e-10, last_epoch=-1, verbose=False)
    else:
        scheduler_t = optim.lr_scheduler.StepLR(optimizer_t, step_size=args.step_size, gamma=0.92)
        # scheduler_t = optim.lr_scheduler.ExponentialLR(optimizer_t, gamma=0.9)

    

    optimizer_s = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    scheduler = optim.lr_scheduler.StepLR(optimizer_s, step_size=args.step_size, gamma=0.9) # 0.0001*0.9^11 = 0.000031(epoch = 30*11=330)
        
    L1Loss = torch.nn.SmoothL1Loss()
    bceloss = torch.nn.BCEWithLogitsLoss()

    if not os.path.isdir('teacher_checkpoint'):
        os.mkdir('teacher_checkpoint')
    if not os.path.isdir(f'teacher_checkpoint/{args.prefix}'):
        os.mkdir(f'teacher_checkpoint/{args.prefix}')
    if not os.path.isdir('teacher_Rec'):
        os.mkdir('teacher_Rec')    
    if not os.path.isdir(f'teacher_Rec/{args.prefix}'):
        os.mkdir(f'teacher_Rec/{args.prefix}')
    
    if not os.path.isdir('student_checkpoint'):
        os.mkdir('student_checkpoint')
    if not os.path.isdir(f'student_checkpoint/{args.prefix}'):
        os.mkdir(f'student_checkpoint/{args.prefix}')
    if not os.path.isdir('student_Rec'):
        os.mkdir('student_Rec')    
    if not os.path.isdir(f'student_Rec/{args.prefix}'):
        os.mkdir(f'student_Rec/{args.prefix}')
    
    
    resume_ind = args.resume_ind if args.resume_ind>0 else 0
    step = resume_ind
    best_sam = 99
    best_sam_s = 99
    iteration = 0
    for epoch in range(resume_ind, args.epochs): 

        model_t.train()
        model.train()

        ep_loss = 0.
        ep_loss_s = 0.

        running_loss_T, running_sam_T, running_bws_T=[],[],[]
        running_loss, running_sam, running_bws=[],[],[]

        # optim_time = time.perf_counter()
        # optim_time_s = time.perf_counter()
        total_optim_time = 0
        total_optim_time_s = 0

        for batch_idx, (X) in enumerate(train_loader):
            iteration +=1
            #===================================Teacher Training===================================
            t1 = time.perf_counter()
            optim_time = time.perf_counter()

            t2 = time.perf_counter()-t1
            if args.DEBUG:
                print(f'Sampling time: {t2} seconds')
            
            t1 = time.perf_counter()
            
            # x2 = lrhsi , x3 = hrmsi , y  = GT
            x3,x2,y,_,_,_ = X
            x2 = x2.to(args.device)
            x3 = x3.to(args.device)

                
            optimizer_t.zero_grad()
            #x = x.to(args.device)
            y = y.to(args.device)
            
            if args.DEBUG:
                print(f'To cuda tensor time {time.perf_counter()-t1} seconds')
            
            t1 = time.perf_counter()

            co_decoded, cofeat = model_t(LRHSI=x2, HRMSI=x3)
           
                
            if args.DEBUG:
                print(f'model inference time{time.perf_counter()-t1} seconds')
            
            loss_t = L1Loss(co_decoded, y)
            loss2_t = sam_loss(co_decoded, y)
            loss3_t = BandWiseMSE(co_decoded, y)
            
            if torch.isnan(loss2_t):
                print('It is unnecessary to optimize teacher anymore...abort the process')
                break
                          
            if args.joint_loss==1:
                teacher_loss = loss_t + 0.1*loss2_t + 0.1*loss3_t
                writer.add_scalar('train/loss_t', loss_t.item(), iteration)
                writer.add_scalar('train/loss2_t', loss2_t.item(), iteration)
                writer.add_scalar('train/loss3_t', loss3_t.item(), iteration)
            else:
                teacher_loss = loss_t
           
            t1 = time.perf_counter()
            teacher_loss.backward(retain_graph=True)
        
            if args.DEBUG:
                print(f'Backward time {time.perf_counter()-t1} seconds')
            
            # if teacher_loss != teacher_loss :
            #     raise Exception('Nan in loss , crack!')

            optimizer_t.step()
            running_loss_T.append(loss_t.item())
            running_sam_T.append(loss2_t.item())
            running_bws_T.append(loss3_t.item())

            optim_time = time.perf_counter()-optim_time
            total_optim_time = total_optim_time + optim_time
            #===================================Student Training====================================
            t1 = time.perf_counter()
            optim_time_s = time.perf_counter()

            x = co_decoded.detach()
            optimizer_s.zero_grad() #將所有參數的梯度緩衝區（buffer）歸零
            x = x.to(args.device)

            t1 = time.perf_counter()
            
            decoded, cofeat_s = model(LRHSI=x2, HRMSI=x3)
                
            if args.DEBUG:
                print(f'model inference time{time.perf_counter()-t1} seconds')
            
            loss_s = L1Loss(decoded, y) # L1(Student Output , GT) 
            lossST = L1Loss(decoded, x) # L1(Student Output , Teacher Output) 

            loss2_s = sam_loss(decoded, y)
            loss3_s = BandWiseMSE(decoded,y)
            
            loss_cofeat = bceloss(cofeat_s, torch.sigmoid( cofeat.detach()))

           
                
            if torch.isnan(loss2_s):
                print('It is unnecessary to optimize student anymore...abort the process')
                break
                      
            if args.joint_loss==1:
                student_loss = loss_s + 0.1*lossST + 0.1*loss_cofeat + 0.1*loss3_s + 0.1*loss2_s
                #loss_s + 0.1*lossST + 0.1*loss_cofeat #0.1*loss2_s + 0.1*loss3_s + 0.1*lossST + 0.1*loss_cofeat
                writer.add_scalar('train/loss_s', loss_s.item(), iteration)
                writer.add_scalar('train/loss2_s', loss2_s.item(), iteration)
                writer.add_scalar('train/loss3_s', loss3_s.item(), iteration)
                writer.add_scalar('train/lossST',lossST.item(), iteration)
                writer.add_scalar('train/loss_cofeat', loss_cofeat.item(), iteration)
            else:
                student_loss = loss_s 
                # student_loss = (1-a) * loss_s + a * lossST
#           
            t1 = time.perf_counter()
            student_loss.backward() #開始進行反向傳播
        
            if args.DEBUG:
                print(f'Backward time {time.perf_counter()-t1} seconds')
            
            optimizer_s.step() #更新權重
            running_loss.append(loss_s.item())
            running_sam.append(loss2_s.item())
            running_bws.append(loss3_s.item())

            optim_time_s = time.perf_counter()-optim_time_s
            total_optim_time_s = total_optim_time_s + optim_time_s

        if torch.isnan(loss2_t):
            break
        
        if torch.isnan(loss2_s):
            break


        scheduler_t.step()
        
        scheduler.step()
        
#             
        if epoch% args.eval_step ==0:

            model_t.eval()
            model.eval()

            with torch.no_grad():

                rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
                rmses_s, sams_s, fnames_s, psnrs_s, ergas_s = [], [], [], [], []
                
                ep = 0
                ep_s = 0

                for ind2, X in enumerate(val_loader):
                    
                    if ind2 % 5 !=0:
                        continue
                    
                    #===================================Teacher Val===================================
                    (vx3, vx2, vy, vfn, maxv, minv) = X
                    vx2=vx2.to(args.device)
                    vx3=vx3.to(args.device)
                        
                    vy = vy.cpu().numpy()
                    vy = vy[:,:,:args.image_size,:args.image_size]
                    #vx=vx.to(args.device)
                    
                    maxv, minv = maxv.cpu().numpy(), minv.cpu().numpy()
                   
                    start_time = time.time()
                    val_dec, _ = model_t(LRHSI=vx2, HRMSI=vx3, mode=1)
                    ep = ep+(time.time()-start_time)
                    
                    vx = val_dec.detach()
                    val_dec = val_dec.cpu().numpy()
                    
                    for predimg, gtimg,f, v1, v2 in zip(val_dec, vy, vfn, maxv, minv):
                        predimg = (predimg/2+0.5) 
                        gtimg = (gtimg/2+0.5) 
                        
                        sams.append(sam2(predimg, gtimg))
                        psnrs.append(psnr(predimg, gtimg))
                        ergas.append(ERGAS(predimg, gtimg))
                        predimg = predimg * (v1-v2) + v2
                        gtimg = gtimg * (v1-v2) + v2
                        rmses.append(rmse(predimg, gtimg))
#                         savemat(f'teacher_Rec/{args.prefix}/{os.path.basename(f)}.mat', {'pred':np.transpose(predimg,(1,2,0))})

                    #===================================Student Val===================================

                    vx = vx.to(args.device)

                    start_time = time.time()
                    val_dec_s, _ = model(LRHSI=vx2, HRMSI=vx3, mode=1)
                    ep_s = ep_s + (time.time()-start_time)
                    
                    val_dec_s = val_dec_s.cpu().numpy()
                    
                    for predimg_s, gtimg_s,f, v1, v2 in zip(val_dec_s, vy, vfn, maxv, minv):
                        predimg_s = (predimg_s/2+0.5) 
                        gtimg_s = (gtimg_s/2+0.5) 
                        
                        sams_s.append(sam2(predimg_s, gtimg_s))
                        psnrs_s.append(psnr(predimg_s, gtimg_s))
                        ergas_s.append(ERGAS(predimg_s, gtimg_s))
                        predimg_s = predimg_s * (v1-v2) + v2
                        gtimg_s = gtimg_s * (v1-v2) + v2
                        rmses_s.append(rmse(predimg_s, gtimg_s))
#                         savemat(f'student_Rec/{args.prefix}/{os.path.basename(f)}.mat', {'pred':np.transpose(predimg_s,(1,2,0))})
                                        
                ep = ep / len(sams)
                ep_s = ep_s / len(sams_s)

                print('T : [epoch: %d] Loss: %.3f, Loss-SAM: %.3f, Loss-BWS: %.3f, val-rmse: %.3f, val-SAM: %.3f, val-PSNR: %.3f, val-ERGAS: %.3f, Inference time: %f ms, Optim time: %f, lr: %f' % (epoch, 100*np.mean(running_loss_T), np.mean(running_sam_T), np.mean(running_bws_T), np.mean(rmses), np.mean(sams), np.mean(psnrs), np.mean(ergas), ep*1000, total_optim_time, scheduler_t.get_last_lr()[0]))
                print('S : [epoch: %d] Loss: %.3f, Loss-SAM: %.3f, Loss-BWS: %.3f, val-rmse: %.3f, val-SAM: %.3f, val-PSNR: %.3f, val-ERGAS: %.3f, Inference time: %f ms, Optim time: %f, lr: %f' % (epoch, 100*np.mean(running_loss), np.mean(running_sam), np.mean(running_bws), np.mean(rmses_s), np.mean(sams_s), np.mean(psnrs_s), np.mean(ergas_s), ep_s*1000, total_optim_time_s, scheduler.get_last_lr()[0]))
                print(' ')
                
                writer.add_scalar('val/RMSE_T', np.mean(rmses), iteration)
                writer.add_scalar('val/SAM_T', np.mean(sams), iteration)
                writer.add_scalar('val/PSNR_T', np.mean(psnrs), iteration)
                writer.add_scalar('val/ERGAS_T', np.mean(ergas), iteration)
                writer.add_scalar('val/RMSE_S', np.mean(rmses_s), iteration)
                writer.add_scalar('val/SAM_S', np.mean(sams_s), iteration)
                writer.add_scalar('val/PSNR_S', np.mean(psnrs_s), iteration)
                writer.add_scalar('val/ERGAS_S', np.mean(ergas_s), iteration)
                

            model_t.train() 
            model.train() 

            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model_t.state_dict(),  f'teacher_checkpoint/{args.prefix}/best.pth')
                
            ep_loss = ep_loss + np.mean(running_loss_T)

                
            running_loss_T, running_sam_T, running_bws_T=[],[], [] 
            
            if best_sam_s > np.mean(sams_s):
                best_sam_s = np.mean(sams_s)
                torch.save(model.state_dict(),  f'student_checkpoint/{args.prefix}/best.pth')
                
            ep_loss_s = ep_loss_s + np.mean(running_loss)

                
            running_loss, running_sam, running_bws=[],[], []
            
            model_t.train() 
            model.train() 
                
        torch.save(model_t.state_dict(), f'teacher_checkpoint/{args.prefix}/last.pth')
        torch.save(model.state_dict(), f'student_checkpoint/{args.prefix}/last.pth')

        step = step + 1  

if __name__ == '__main__':
    
    args = parse_args()
    torch.manual_seed(args.SEED)
    rn.seed(args.SEED)
    np.random.seed(args.SEED)

    ## Reading files
    print("#"*80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#"*80)
    print('start training')
    print("#"*80)

    trainer(args)

    print("#"*80)
    print('finish')
