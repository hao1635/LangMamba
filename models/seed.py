import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
from torch.nn import init
import os
import util.util as util
import ipdb


class Attention2d(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2d, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out
    

class SpatialChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x*out

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class Attention_Block(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads=8,**kwargs):
        super(Attention_Block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        base_d_state=4
        d_state=int(base_d_state * num_heads)

        compress_ratio=4
        num_feat=input_channel
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor=16)
        )
    
        self.spa=SpatialChannelAttention(num_feat, reduction=16)
        

    def forward(self, inputs):

        attn_s=self.attention_s(inputs.permute(0,2,3,1)).permute(0,3,1,2)

        inputs_attn=inputs+attn_s +self.spa(self.cab(inputs))

        return inputs_attn


class Conv_FFN(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel,res=True):
        super(Conv_FFN,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.conv_1=nn.Conv2d(input_channel,middle_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_2=nn.Conv2d(middle_channel,output_channel,kernel_size=3,stride=1,padding=1,bias=False)
        if self.input_channel != self.output_channel:
            self.shortcut=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)
        self.res=res
        self.act=nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        conv_S=self.act(self.conv_1(inputs))
        conv_S=self.act(self.conv_2(conv_S))

        if self.input_channel == self.output_channel:
            identity_out=inputs
        else:
            identity_out=self.shortcut(inputs)

        if self.res:
            output=conv_S+identity_out
        else:
            output=conv_S

        return output


class EMA_Block(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=8,res=True):
        super(EMA_Block,self).__init__()
        self.emablock=nn.Sequential(
            Attention_Block(in_channels,in_channels,num_heads=num_heads),
            Conv_FFN(in_channels,in_channels,out_channels,res=res),
        )
    def forward(self,x):
        return self.emablock(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(Down,self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d((2,2), (2,2)),
            EMA_Block(in_channels,out_channels,num_heads=num_heads,res=res)
        )
            
    def forward(self, x):
        return self.encoder(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True,trilinear=True, num_heads=8,res=False):
        super(Up,self).__init__()

        self.res_unet=res
        if trilinear:
            self.up = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels , kernel_size=2, stride=2)
        
        self.conv = EMA_Block(in_channels, out_channels, num_heads=num_heads,res=res)

    def forward(self, x1, x2):
        ##ipdb.set_trace()

        x1 = self.up(x1)
        
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,decouple=None,bn=True,res=True,activation=False):
        super(SingleConv,self).__init__()
        self.act=activation
        self.conv =nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x=self.conv(x)
        if self.act==True:
            x=self.activation(x)
        return x
        

class SEED(nn.Module):
    def __init__(self,opt,encoder,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=False):
        super(SEED,self).__init__()
        #ipdb.set_trace()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels

        self.encoder=encoder
        for params in self.encoder.parameters():
            params.requires_grad = False
        
        self.bottleneck=EMA_Block(2 * n_channels,2 * n_channels,num_heads=num_heads[3])
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads=num_heads[3],res=res)
        
        self.dec2 = Up(3 * n_channels, 1 * n_channels,num_heads=num_heads[2],res=res)
        
        self.dec3 = Up(2 * n_channels, n_channels,num_heads=num_heads[1],res=res)

        self.dec4 = Up(2 * n_channels, n_channels,num_heads=num_heads[1],res=res)
        
        self.out = SingleConv(n_channels,out_channels,res=res,activation=False)

    def forward(self, x):
        b, c, h, w = x.size()

        _,enout=self.encoder(x)  # 128,32,32

        output= self.bottleneck(enout[-1])

        #ipdb.set_trace()
        output = self.dec1(output, enout[-2]) # 
        output = self.dec2(output, enout[-3])
        output = self.dec3(output, enout[-4])
        output = self.dec4(output, enout[-5])
        
        output = self.out(output) #+x

        return output


