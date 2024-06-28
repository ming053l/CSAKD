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
from tensorboardX import SummaryWriter 
from trainOps import *
from torchvision import transforms


# from models.dcsn3d import DCSNMain as DCSN
from models.dcsn2d_light import DCSN_Teacher , DCSN_Student

torch.backends.cudnn.benchmark=True

def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=600) #
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--resume_ind', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default="")
    parser.add_argument('--snr', type=int, default=35)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=200) #
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=2)
    parser.add_argument('--finetuning_step', type=int, default=300, help='Works only if the mixed_align_opt is on') #
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate, 0 means training without weight decay')
    
    
    ## Data generator configuration
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bands', type=int, default=172)
    parser.add_argument('--msi_bands', type=int, default=4)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    parser.add_argument('--joint_loss', type=int, default=1)
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=172, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    parser.add_argument("--root", type=str, default="/work/u1657859/DCSNv2/Data", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="./val.txt")   
    parser.add_argument("--train_file", type=str, default="./train.txt")   
    parser.add_argument("--prefix", type=str, default="KD_QRCODE")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args


def Teacher_trainer(args):
    
    flist = loadTxt(args.train_file)
    valfn = loadTxt(args.val_file)
    tlen = len(flist)
    print(f'#training samples is {tlen} and validation samples is {len(valfn)}')

    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')
    
    train_loader = torch.utils.data.DataLoader(dataset(flist, args), batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.workers)

    model = DCSN_Teacher(args).to(args.device)
    if args.gpus>1:
        model = torch.nn.DataParallel(model).to(args.device)
    
    if args.resume_ind>0 or os.path.isfile(args.resume_ckpt):
        if not os.path.isfile(args.resume_ckpt):
            # args.resume_ckpt = os.path.join('pretrain_teacher', args.prefix, 'best.pth')
            args.resume_ckpt = os.path.join('pretrain_teacher', args.prefix, 'last.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict = torch.load(args.resume_ckpt)  
        model.load_state_dict(state_dict)
        print(f'Loading the pretrained model from {args.resume_ckpt}')
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    if args.lr_scheduler=='cosine':
        #設置的最初始化及最小學習率，依照cosine週期衰減，直到T-max週期到了，又開始往上增加學習率。
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.resume_ind, eta_min=1e-10, last_epoch=-1, verbose=False)
    else:
        #會依照你設定的step_size來調整學習率。
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
        
    L1Loss = torch.nn.SmoothL1Loss()

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(f'checkpoint/{args.prefix}'):
        os.mkdir(f'checkpoint/{args.prefix}')
    if not os.path.isdir('Rec'):
        os.mkdir('Rec')    
    if not os.path.isdir(f'Rec/{args.prefix}'):
        os.mkdir(f'Rec/{args.prefix}')
    
    writer = SummaryWriter('log/%s_exp2' % (args.prefix))
    
    resume_ind = args.resume_ind if args.resume_ind>0 else 0
    step = resume_ind
    best_sam = 99
    for epoch in range(resume_ind, args.epochs): 
            
        ep_loss = 0.
        running_loss, running_sam, running_bws=[],[],[]
        t1 = time.perf_counter()
        optim_time = time.perf_counter()
        for batch_idx, (X) in enumerate(train_loader):
            t2 = time.perf_counter()-t1
            if args.DEBUG:
                print(f'Sampling time: {t2} seconds')
            
            t1 = time.perf_counter()
            if args.network_mode==2:
                # x  = cokey , x2 = lrhsi , x3 = hrmsi , y  = GT
                x,x2,x3,y,_,_,_ = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                # x2 = lrhsi , x = hrmsi , y  = GT
                x,x2,y,_,_,_ = X
                x2 = x2.cuda()
            elif args.network_mode==0:
                # x  = co , y  = GT
                x,y,_,_,_ = X
                
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            if args.DEBUG:
                print(f'To cuda tensor time {time.perf_counter()-t1} seconds')
            
            t1 = time.perf_counter()
            
            if args.network_mode==2:
                # x  = cokey , x2 = lrhsi , x3 = hrmsi , y  = GT
                co_decoded = model(x, LRHSI=x2, HRMSI=x3)
            elif args.network_mode==1:
                # x2 = lrhsi , x = hrmsi , y  = GT
                co_decoded = model(None, LRHSI=x2, HRMSI=x)
            elif args.network_mode==0:
                # x  = co , y  = GT
                co_decoded = model(x, LRHSI=None, HRMSI=None)
                
            if args.DEBUG:
                print(f'model inference time{time.perf_counter()-t1} seconds')
            loss_t = L1Loss(co_decoded, y)
            loss2_t = sam_loss(co_decoded, y)
            loss3_t = BandWiseMSE(co_decoded, y)
            
            while torch.isnan(loss2_t) and scheduler.get_last_lr()[0]>1e-12:
                print('Force learning rate decay to', scheduler.get_last_lr()[0]/5)
                
                args.resume_ckpt = os.path.join('teacher_checkpoint', args.prefix, 'last.pth')
                state_dict = torch.load(args.resume_ckpt)  
                model.load_state_dict(state_dict)
                
                optimizer = optim.Adam(model.parameters(), lr=scheduler.get_last_lr()[0]/5)  
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
                args.joint_loss = 0
                
                continue
                
            if torch.isnan(loss2_t):
                print('It is unnecessary to optimize anymore...abort the process')
                break
            
                         
            reg = torch.std(co_decoded)
            if args.joint_loss==1:
                teacher_loss = loss_t + 0.1*loss2_t + 0.1*loss3_t
            else:
                teacher_loss = loss_t

#           
            t1 = time.perf_counter()
            teacher_loss.backward(retain_graph=True)
        
            if args.DEBUG:
                print(f'Backward time {time.perf_counter()-t1} seconds')
            
            # if teacher_loss != teacher_loss :
            #     raise Exception('Nan in loss , crack!')

            optimizer.step()
            running_loss.append(loss_t.item())
            running_sam.append(loss2_t.item())
            running_bws.append(loss3_t.item())

        if torch.isnan(loss2_t):
            break

        scheduler.step()
        optim_time = time.perf_counter()-optim_time
#             
        if epoch% args.eval_step ==0:
            model.eval()
            with torch.no_grad():
                rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
                
                ep = 0
                for ind2, X in enumerate(val_loader):
                    
                    if args.network_mode==2:
                        (vx,vx2, vx3, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                        vx3=vx3.cuda()
                    elif args.network_mode==1:
                        (vx, vx2, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                    elif args.network_mode==0:
                        (vx, vy, vfn, maxv, minv) = X
                        
                    vy = vy.cpu().numpy()
                    vy = vy[:,:,:args.image_size,:args.image_size]
                    vx = vx.cuda()
                    
                    maxv, minv = maxv.cpu().numpy(), minv.cpu().numpy()
                   
                    start_time = time.time()
                    if args.network_mode==2:
                        val_dec = model(vx, LRHSI=vx2, HRMSI=vx3, mode=1)
                    elif args.network_mode==1:
                        val_dec = model(None, LRHSI=vx2, HRMSI=vx, mode=1)
                    elif args.network_mode==0:
                        val_dec = model(vx, LRHSI=None, HRMSI=None, mode=1)
                    ep = ep+(time.time()-start_time)
                    
                    val_dec_out = val_dec
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
                        savemat(f'Rec/{args.prefix}/{os.path.basename(f)}.mat', {'pred':np.transpose(predimg,(1,2,0))})
                                        
                ep = ep / len(sams)
                print('[epoch: %d] Loss: %.3f, Loss-SAM: %.3f, Loss-BWS: %.3f, val-rmse: %.3f, val-SAM: %.3f, val-PSNR: %.3f, val-ERGAS: %.3f, Inference time: %f ms, Optim time: %f, lr: %f' % (epoch, 100*np.mean(running_loss), np.mean(running_sam), np.mean(running_bws), np.mean(rmses), np.mean(sams), np.mean(psnrs), np.mean(ergas), ep*1000, optim_time,scheduler.get_last_lr()[0]))
                ## Dump the SAM/RMSE
                writer.add_scalar('Validation RMSE', np.mean(rmses), step)
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                writer.add_scalar('Validation PSNR', np.mean(psnrs), step)
                writer.add_scalar('Validation ERGAS', np.mean(ergas), step)
                writer.add_scalar('Validation RMSE/std', np.std(rmses), step)
                writer.add_scalar('Validation SAM/std', np.std(sams), step)
                writer.add_scalar('Validation PSNR/std', np.std(psnrs), step)
                writer.add_scalar('Validation ERGAS/std', np.std(ergas), step)

            model.train() 
            
            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(),  f'checkpoint/{args.prefix}/best.pth')
                
            ep_loss = ep_loss + np.mean(running_loss)
            writer.add_scalar('Loss/Running loss', np.mean(running_loss), step)
            writer.add_scalar('Loss/Running SAM-loss', np.mean(running_sam), step)
            writer.add_scalar('Loss/Running Weighted-MSE', np.mean(running_bws), step)
                
            running_loss, running_sam, running_bws=[],[], []
            model.train() 
        
                
        torch.save(model.state_dict(), f'checkpoint/{args.prefix}/last.pth')
        
        output_t = co_decoded
        output_val = val_dec_out

        step = step + 1
    
    return output_t , output_val

def Student_trainer(args , co_decoded, val):

    flist = loadTxt(args.train_file)
    valfn = loadTxt(args.val_file)
    tlen = len(flist)
    print(f'#training samples is {tlen} and validation samples is {len(valfn)}')

    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')
    
    train_loader = torch.utils.data.DataLoader(dataset(flist, args), batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.workers)

    model = DCSN_Student(args).to(args.device)
    # if args.gpus>1:
    #     model = torch.nn.DataParallel(model).to(args.device)
    
    # if args.resume_ind>0 or os.path.isfile(args.resume_ckpt):
    #     if not os.path.isfile(args.resume_ckpt):
    #         args.resume_ckpt = os.path.join('student_checkpoint', args.prefix, 'best.pth')
    #     if not os.path.isfile(args.resume_ckpt):
    #         print(f"checkpoint is not found at {args.resume_ckpt}")
    #         raise 
    #     state_dict = torch.load(args.resume_ckpt)  
    #     model.load_state_dict(state_dict)
    #     print(f'Loading the pretrained model from {args.resume_ckpt}')
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    if args.lr_scheduler=='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.resume_ind, eta_min=1e-10, last_epoch=-1, verbose=False)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
        
    L1Loss = torch.nn.SmoothL1Loss()

    if not os.path.isdir('student_checkpoint'):
        os.mkdir('student_checkpoint')
    if not os.path.isdir(f'student_checkpoint/{args.prefix}'):
        os.mkdir(f'student_checkpoint/{args.prefix}')
    if not os.path.isdir('student_Rec'):
        os.mkdir('student_Rec')    
    if not os.path.isdir(f'student_Rec/{args.prefix}'):
        os.mkdir(f'student_Rec/{args.prefix}')
    
    writer = SummaryWriter('log/student/%s_exp2' % (args.prefix))
    
    resume_ind = args.resume_ind if args.resume_ind>0 else 0
    step = resume_ind
    best_sam = 99
    for epoch in range(resume_ind, args.epochs): 
            
        ep_loss = 0.
        running_loss, running_sam, running_bws=[],[], []
        t1 = time.perf_counter()
        optim_time = time.perf_counter()
        for batch_idx, (X) in enumerate(train_loader):
            t2 = time.perf_counter()-t1
            if args.DEBUG:
                print(f'Sampling time: {t2} seconds')
            
            t1 = time.perf_counter()
            if args.network_mode==2:
                # x  = cokey , x2 = lrhsi , x3 = hrmsi , y  = GT
                x2,x3,y,_,_,_ = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                # x2 = lrhsi , x3 = hrmsi , y  = GT
                x3,x2,y,_,_,_ = X
                x2 = x2.cuda()
                x3 = x3.cuda()
            elif args.network_mode==0:
                x,y,_,_,_ = X

            optimizer.zero_grad() #將所有參數的梯度緩衝區（buffer）歸零
            x = co_decoded
            x = x.cuda()
            y = y.cuda()
            if args.DEBUG:
                print(f'To cuda tensor time {time.perf_counter()-t1} seconds')
            
            t1 = time.perf_counter()
            
            if args.network_mode==2:
                decoded = model(x, LRHSI=x2, HRMSI=x3)
            elif args.network_mode==1:
                decoded = model(None, LRHSI=x2, HRMSI=x3)
            elif args.network_mode==0:
                decoded = model(x, LRHSI=None, HRMSI=None)
                
            if args.DEBUG:
                print(f'model inference time{time.perf_counter()-t1} seconds')
            
            # a = 0.5
            # loss_s = a * L1Loss(decoded, y) + (1-a) * L1Loss(decoded, x) # 0.5 * L1(Student Output , GT) + 0.5 * L1(Student Output , Teacher Output)
            loss_s = L1Loss(decoded, y) # L1(Student Output , GT) 
            lossST = L1Loss(decoded, x) # L2(Student Output , Teacher Output) 
            loss2_s = sam_loss(decoded, y)
            loss3_s = BandWiseMSE(decoded, y)
            
            while torch.isnan(loss2_s) and scheduler.get_last_lr()[0]>1e-12:
                print('Force learning rate decay to', scheduler.get_last_lr()[0]/5)
                
                args.resume_ckpt = os.path.join('student_checkpoint', args.prefix, 'last.pth')
                state_dict = torch.load(args.resume_ckpt)  
                model.load_state_dict(state_dict)
                
                optimizer = optim.Adam(model.parameters(), lr=scheduler.get_last_lr()[0]/5)  
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
                args.joint_loss = 0
                
                continue
                
            if torch.isnan(loss2_s):
                print('It is unnecessary to optimize anymore...abort the process')
                break
            
                      
            reg = torch.std(decoded)
            if args.joint_loss==1:
                student_loss = loss_s + 0.1*loss2_s + 0.1*loss3_s + 0.1*lossST
            else:
                student_loss = loss_s 

#           
            t1 = time.perf_counter()
            student_loss.backward() #開始進行反向傳播
        
            if args.DEBUG:
                print(f'Backward time {time.perf_counter()-t1} seconds')

            # if student_loss != student_loss :
            #     raise Exception('Nan in loss , crack!')
            
            optimizer.step() #更新權重
            running_loss.append(loss_s.item())
            running_sam.append(loss2_s.item())
            running_bws.append(loss3_s.item())

        if torch.isnan(loss2_s):
            break 

        scheduler.step()
        optim_time = time.perf_counter()-optim_time
#            
        if epoch% args.eval_step ==0:
            model.eval()
            with torch.no_grad():
                rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
                
                ep = 0
                for ind2, X in enumerate(val_loader):
                    
                    if args.network_mode==2:
                        (vx2, vx3, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                        vx3=vx3.cuda()
                    elif args.network_mode==1:
                        (vx3, vx2, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                        vx3=vx3.cuda()
                    elif args.network_mode==0:
                        (vx, vy, vfn, maxv, minv) = X
                      
                    vy = vy.cpu().numpy()
                    vy = vy[:,:,:args.image_size,:args.image_size]

                    # vx = co_decoded.detach() #torch.Size([1, 172, 128, 128])
                    # resize = transforms.Resize([256,256])
                    # vx = resize(vx)

                    vx = val
                    vx = vx.cuda()

                    maxv, minv = maxv.cpu().numpy(), minv.cpu().numpy()
                   
                    start_time = time.time()
                    if args.network_mode==2:
                        val_dec = model(vx, LRHSI=vx2, HRMSI=vx3, mode=1)
                    elif args.network_mode==1:
                        val_dec = model(None, LRHSI=vx2, HRMSI=vx3, mode=1)
                    elif args.network_mode==0:
                        val_dec = model(vx, LRHSI=None, HRMSI=None, mode=1)
                    ep = ep+(time.time()-start_time)
                    
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
                        savemat(f'student_Rec/{args.prefix}/{os.path.basename(f)}.mat', {'pred':np.transpose(predimg,(1,2,0))})
                                        
                ep = ep / len(sams)
                print('[epoch: %d] Loss: %.3f, Loss-SAM: %.3f, Loss-BWS: %.3f, val-rmse: %.3f, val-SAM: %.3f, val-PSNR: %.3f, val-ERGAS: %.3f, Inference time: %f ms, Optim time: %f, lr: %f' % (epoch, 100*np.mean(running_loss), np.mean(running_sam), np.mean(running_bws), np.mean(rmses), np.mean(sams), np.mean(psnrs), np.mean(ergas), ep*1000, optim_time,scheduler.get_last_lr()[0]))
                ## Dump the SAM/RMSE
                writer.add_scalar('Validation RMSE', np.mean(rmses), step)
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                writer.add_scalar('Validation PSNR', np.mean(psnrs), step)
                writer.add_scalar('Validation ERGAS', np.mean(ergas), step)
                writer.add_scalar('Validation RMSE/std', np.std(rmses), step)
                writer.add_scalar('Validation SAM/std', np.std(sams), step)
                writer.add_scalar('Validation PSNR/std', np.std(psnrs), step)
                writer.add_scalar('Validation ERGAS/std', np.std(ergas), step)

            model.train() 
            
            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(),  f'student_checkpoint/{args.prefix}/best.pth')
                
            ep_loss = ep_loss + np.mean(running_loss)
            writer.add_scalar('Loss/Running loss', np.mean(running_loss), step)
            writer.add_scalar('Loss/Running SAM-loss', np.mean(running_sam), step)
            writer.add_scalar('Loss/Running Weighted-MSE', np.mean(running_bws), step)
                
            running_loss, running_sam, running_bws=[],[], []
            model.train() 
        
                
        torch.save(model.state_dict(), f'student_checkpoint/{args.prefix}/last.pth')
        
        step = step + 1

if __name__ == '__main__':
    
    args = parse_args()
    torch.manual_seed(args.SEED)
    rn.seed(args.SEED)
    np.random.seed(args.SEED)

    ## Reading files #
    print("#"*80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#"*80)

    print('start training network')
    print("#"*80)
    
    co_decoded , val= Teacher_trainer(args)

    # print("#"*80)

    # #args.network_mode = 2 # 2 : COCNN (LRHSI+HRMSI+CO)
    # args.resume_ind = 0
    # #args.epochs = 1000
    
    # print('start training student network')
    # print("#"*80)
    
    # # 使用detach操作，將requires_grad停止
    # # detach()就是返回一個新的tensor，並且這個tensor是從當前的計算圖中分離出來的。但是返回的tensor和原來的tensor是共享內存空間
    # # 如果A網絡的輸出給B網絡作為輸入， 如果我們希望在梯度反傳的時候只更新B中參數的值，而不更新A中的參數值，這時候就可以使用detach()
    # Student_trainer(args , co_decoded.detach() , val.detach())

    print("#"*80)
    print('finish')
