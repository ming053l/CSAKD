import torch
from torch import nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
import functools
from module_util import *
from ptflops import get_model_complexity_info
import pdb
# Attention-based fusion module with Sigmoid
class SigmoidAttentionFusion(nn.Module):
    def __init__(self, num_features):
        super(SigmoidAttentionFusion, self).__init__()
        self.weights = nn.Parameter(torch.randn(4, num_features, 1,1))

    def forward(self, x1, x2, x3, x4):
        # Apply sigmoid to the weights to get independent attention scores
        attn_scores = torch.sigmoid(self.weights)

        # Independent weighted sum of the features
        fused = attn_scores[0] * x1 + attn_scores[1] * x2 + attn_scores[2] * x3 + attn_scores[3] * x4
        return fused


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channels, reduced_dim, heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.heads = heads
        self.reduced_dim = reduced_dim // heads
        self.query_conv = nn.Conv2d(channels, reduced_dim, 1)
        self.key_conv = nn.Conv2d(channels, reduced_dim, 1)
        self.value_conv = nn.Conv2d(channels, reduced_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv2d(reduced_dim, channels, 1)

    def forward(self, x1, x2, x3, x4):
        batch_size, channels, height, width = x1.size()
        
        # Step 1: Dimensionality reduction and reshaping for multi-head attention
        q = self.query_conv(x1).view(batch_size, self.heads, self.reduced_dim, -1)
        k = self.key_conv(x2).view(batch_size, self.heads, self.reduced_dim, -1)
        v = self.value_conv(x3).view(batch_size, self.heads, self.reduced_dim, -1)
        
        # Step 2: Scaled dot-product attention
        qk = torch.einsum('bhqd,bhkd->bhqk', q, k) / self.reduced_dim ** 0.5
        attn = self.softmax(qk)
        
        # Step 3: Apply attention to values
        v = torch.einsum('bhqk,bhvd->bhqd', attn, v)
        
        # Step 4: Recombine heads and apply final transformation
        v = v.contiguous().view(batch_size, -1, height, width)
        x4 = self.output_conv(v) + x4  # Skip connection from input x4
        
        return x4
    
class ConcatAttentionFusion(nn.Module):
    def __init__(self, num_features):
        super(ConcatAttentionFusion, self).__init__()
        self.attention_conv = nn.Conv2d(num_features * 4, 4, 1)  # Generate attention weights from concatenated inputs

    def forward(self, x1, x2, x3, x4):
        # Concatenate the inputs along the channel dimension
        concatenated = torch.cat([x1, x2, x3, x4], dim=1)

        # Generate attention weights using a convolution
        # Apply sigmoid to get attention scores between 0 and 1
        attn_scores = torch.sigmoid(self.attention_conv(concatenated))

        # Apply attention weights to each input
        weighted_sum = attn_scores[:, 0:1] * x1 + attn_scores[:, 1:2] * x2 + attn_scores[:, 2:3] * x3 + attn_scores[:, 3:4] * x4

        return weighted_sum


class CrossSigmoidAttention(nn.Module):
    def __init__(self, channels, reduced_dim, heads):
        super(CrossSigmoidAttention, self).__init__()
        self.heads = heads
        self.reduced_dim = reduced_dim // heads
        self.query_conv = nn.Conv2d(channels, reduced_dim, 1)
        self.key_conv = nn.Conv2d(channels, reduced_dim, 1)
        self.value_conv = nn.Conv2d(channels, reduced_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv2d(reduced_dim, channels, 1)
        self.attention_weights = nn.Conv2d(channels * 4, 4, 1)  # For generating attention weights

    def forward(self, x1, x2, x3, x4):
        batch_size, channels, height, width = x1.size()

        # Cross Attention
        q = self.query_conv(x1).view(batch_size, self.heads, self.reduced_dim, -1)
        k = self.key_conv(x2).view(batch_size, self.heads, self.reduced_dim, -1)
        v = self.value_conv(x3).view(batch_size, self.heads, self.reduced_dim, -1)

        qk = torch.einsum('bhqd,bhkd->bhqk', q, k) / self.reduced_dim ** 0.5
        attn = self.softmax(qk)
        v = torch.einsum('bhqk,bhvd->bhqd', attn, v).contiguous().view(batch_size, -1, height, width)
        attended_output = self.output_conv(v) + x4  # Skip connection from input x4

        # Concatenate all inputs and attended output
        concatenated = torch.cat([x1, x2, x3, attended_output], dim=1)

        # Generate sigmoid attention weights
        attn_weights = torch.sigmoid(self.attention_weights(concatenated))

        # Apply sigmoid attention weights
        weighted_sum = (attn_weights[:, 0:1] * x1 +
                        attn_weights[:, 1:2] * x2 +
                        attn_weights[:, 2:3] * x3 +
                        attn_weights[:, 3:4] * attended_output)

        return weighted_sum


    
##===============Student + MFB + MFA ===============================================
class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, _group=None):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch , kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class MultiScaleFeatFusionBlock(nn.Module):   ## MFB
    def __init__(self, nf, gc, bias=True, groups=4, depthwise=False):
        super(MultiScaleFeatFusionBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        block = nn.Conv2d if depthwise == False else DeepWise_PointWise_Conv
        
        self.conv1 = block(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = block(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = block(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = block(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = block(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class MultiScaleFeatAggregation(nn.Module):   ## MFA

    def __init__(self, nf, gc, groups=4):
        super(MultiScaleFeatAggregation, self).__init__()
        self.MFB1 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
        self.MFB2 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
        self.MFB3 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)

    def forward(self, x):
        out = self.MFB1(x)
        out = self.MFB2(out)
        out = self.MFB3(out)
        return out * 0.2 + x


class FNN_Teacher(nn.Module):
    
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def __init__(self, in_nc, out_nc, nf, nb, in_msi, gc, groups=4, scale=4, adaptive_fuse=0):
        super(FNN_Teacher, self).__init__()

        self.msi_num = in_msi
        self.hsi_num = out_nc
        self.adaptive_fuse = adaptive_fuse
        self.scale=scale
        # for LRHSI
        in_nc_group = groups
        if in_nc % groups != 0:
            in_nc_group = 1
        
        fuser_set = [SigmoidAttentionFusion(nf), 
                     MultiHeadCrossAttention(nf, 64, 4),
                     CrossSigmoidAttention(nf, 64, 4),
                    ConcatAttentionFusion(nf)]

        if adaptive_fuse!=0:
            fuser = fuser_set[adaptive_fuse-1]
            del fuser_set
            
        
        self.hmconv1 = nn.Conv2d(self.msi_num+self.hsi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=1)
        self.hmfeat = make_layer(block1, 6)
        
        self.mhconv1 = nn.Conv2d(self.msi_num+self.hsi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=1)
        self.mhfeat = make_layer(block1, 6)
        
        self.hconv1 = nn.Conv2d(self.hsi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=1)
        self.hfeat = make_layer(block1, 4)
        
        self.mconv1 = nn.Conv2d(self.msi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=1)
        self.mfeat = make_layer(block1, 4)
        
        self.fuse = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True, groups=groups)
        self.last = nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=False)
        
        self.up = torch.nn.Upsample(scale_factor = scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.down = torch.nn.Upsample(scale_factor = a)
        if adaptive_fuse>0: self.fuse_att = fuser
        
       
        
    def forward(self, lrhsi, hrmsi):
        
        lrhsi_up = F.interpolate(lrhsi, scale_factor=self.scale, mode='bilinear')
        hrmsi_down = F.interpolate(hrmsi, scale_factor=1/self.scale, mode='bicubic')
        
        lrhsi_up = self.lrelu(self.hmconv1(torch.cat((lrhsi_up, hrmsi), 1) ))
        lrhsi_up = self.hmfeat(lrhsi_up)
        
        hrmsi_down = self.lrelu(self.mhconv1(torch.cat((lrhsi, hrmsi_down), 1) ))
        hrmsi_down = self.mhfeat(hrmsi_down)
        hrmsi_down = self.up(hrmsi_down)
        
        hrmsi      = self.lrelu(self.mconv1(hrmsi))
        hrmsi      = self.mfeat(hrmsi)
        
        lrhsi      = self.lrelu(self.hconv1(lrhsi))
        lrhsi      = self.hfeat(lrhsi)
        
        if self.adaptive_fuse==0:
            fuse = self.up(lrhsi) + hrmsi + hrmsi_down+lrhsi_up
        else:
            fuse = self.fuse_att(self.up(lrhsi), hrmsi, hrmsi_down, lrhsi_up)
        co_teacher = self.lrelu(self.fuse(fuse))
        co_teacher = self.last(co_teacher)
        
        
        return co_teacher, fuse





class FNN_Student(nn.Module):
    
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def __init__(self, in_nc,out_nc, nf, nb, in_msi, gc, groups=4, scale=4, adaptive_fuse=False, s_layers=None):
        super(FNN_Student, self).__init__()
        
        self.msi_num = in_msi
        self.hsi_num = out_nc
        self.adaptive_fuse = adaptive_fuse
        self.scale=scale
        # for LRHSI
        in_nc_group = groups
        if in_nc % groups != 0:
            in_nc_group = 1
            
        if s_layers is None:
            s_layers = [1,4,4,1]
            
        # Example usage:
        num_channels = 256  # Number of input channels for each feature map
        reduced_dim = 64  # Reduced number of dimensions after transformation
        heads = 4  # Number of attention heads

#         # Create the cross-attention module
#         cross_attention_module = MultiHeadCrossAttention(num_channels, reduced_dim, heads)


#         # Apply the cross-attention module
#         fused_output = cross_attention_module(x1, x2, x3, x4)

        fuser_set = [SigmoidAttentionFusion(nf), 
                     MultiHeadCrossAttention(nf, 64, 4),
                     CrossSigmoidAttention(nf, 64, 4),
                    ConcatAttentionFusion(nf)]

        if adaptive_fuse!=0:
            fuser = fuser_set[adaptive_fuse-1]
            del fuser_set    
        
        self.hmconv1 = nn.Conv2d(self.msi_num+self.hsi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=in_nc_group)
        self.hmfeat = make_layer(block1, s_layers[0])
        
        self.mhconv1 = nn.Conv2d(self.msi_num+self.hsi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=1)
        self.mhfeat = make_layer(block1, s_layers[1])
        
        self.hconv1 = nn.Conv2d(self.hsi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=nf//2)
        self.hfeat = make_layer(block1, s_layers[2])
        
        self.mconv1 = nn.Conv2d(self.msi_num, nf, 3, 1, 1, bias=True, groups=1)
        block1 = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc, groups=in_nc_group)
        self.mfeat = make_layer(block1, s_layers[3])
        
        if adaptive_fuse>0: self.fuse_att = fuser
        self.fuse = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True, groups=groups)
        self.last = nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=False)
        
        self.up = torch.nn.Upsample(scale_factor = scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.down = torch.nn.Upsample(scale_factor = a)
        
       
        
    def forward(self, lrhsi, hrmsi):
        
        lrhsi_up = F.interpolate(lrhsi, scale_factor=self.scale, mode='bilinear')
        hrmsi_down = F.interpolate(hrmsi, scale_factor=1/self.scale, mode='bicubic')
        
        
        lrhsi_up = self.lrelu(self.hmconv1(torch.cat((lrhsi_up, hrmsi), 1) ))
#         lrhsi_up = self.hmfeat(lrhsi_up)
        
        hrmsi_down = self.lrelu(self.mhconv1(torch.cat((lrhsi, hrmsi_down), 1) ))
        hrmsi_down = self.mhfeat(hrmsi_down)
        hrmsi_down = self.up(hrmsi_down)
        
        hrmsi      = self.lrelu(self.mconv1(hrmsi))
        hrmsi      = self.mfeat(hrmsi)
        
#         import pdb
#         pdb.set_trace()
        lrhsi      = self.lrelu(self.hconv1(lrhsi))
        lrhsi      = self.hfeat(lrhsi)
        
        if self.adaptive_fuse==0:
            fuse = self.up(lrhsi) + hrmsi + hrmsi_down+ lrhsi_up
        else:
            fuse = self.fuse_att(self.up(lrhsi), hrmsi, hrmsi_down, lrhsi_up)
        co = self.lrelu(self.fuse(fuse))
        co = self.last(co)
        
        
        return co, fuse

##=========DCSN_Teacher full module==========
class Teacher(nn.Module): 
    def __init__(self, args):
        super(Teacher, self).__init__()
        self.snr = args.snr
        self.device=args.device
        self.joint = args.network_mode
        # down = torch.nn.Upsample(scale_factor=0.5)
        
        self.decoder = FNN_Teacher(in_nc=args.bands, out_nc=args.bands, nf=args.num_base_chs, 
                                            nb=args.num_blocks, gc=args.num_agg_feat, in_msi=args.msi_bands, 
                                            groups=args.groups, adaptive_fuse=args.adaptive_fuse)
       
    def awgn(self, x):
        snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).to(self.device) * npower


    def forward(self, LRHSI=None, HRMSI=None, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        if self.snr>0 and mode==0 and self.joint==1:
            LRHSI = self.awgn(LRHSI)
            # HRMSI = self.awgn(HRMSI)

        return self.decoder(LRHSI, HRMSI)

##=========DCSN_Student full module==========
class Student(nn.Module): 
    def __init__(self, args):
        super(Student, self).__init__()
        self.snr = args.snr
        self.device = args.device
        self.joint = args.network_mode
        
        #print(args.adaptive_fuse)
        self.decoder = FNN_Student(in_nc=args.bands, out_nc=args.bands, nf=args.num_base_chs, 
                                            nb=args.num_blocks, gc=args.num_agg_feat, in_msi=args.msi_bands, 
                                            groups=args.groups, adaptive_fuse=args.adaptive_fuse, s_layers=args.student_layers)
       
    def awgn(self, x):
        snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).to(self.device) * npower


    def forward(self, LRHSI=None, HRMSI=None, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        if self.snr>0 and mode==0 and self.joint==1:
            LRHSI = self.awgn(LRHSI)
            #HRMSI = self.awgn(HRMSI)
            
        return self.decoder(LRHSI, HRMSI)

if __name__ == '__main__':
    
    #Flops & Parameter
    # Create a network and a corresponding input
    device = 'cuda:0'
    print('==============================================================================')
    model = Teacher()
    flops, params = get_model_complexity_info(model, input_res=(1, 224, 224), 
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)

    print('==============================================================================')
    model = Student()
    flops, params = get_model_complexity_info(model, input_res=(1, 224, 224), 
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)