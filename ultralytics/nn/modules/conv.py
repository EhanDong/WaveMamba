# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from copy import copy
from pathlib import Path
import clip
import numpy as np
import pandas as pd
import requests
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from typing import Tuple
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import matplotlib.pyplot as plt
#from scipy.misc import imread, imsave
from PIL import Image
from torch.nn import init, Sequential
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
#from .block import C3,C2f
from .wavelet import create_wavelet_filter, wavelet_transform, inverse_wavelet_transform
__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "frequencyfuse",
    "fusehigh",
    "fuselow",
    "fusemiddle",
    "Cross_modal_Vmaba_module2",
    "ChannelAttention2",
    "SpatialAttention2",
    "CBAM3",
    "Add2",
    "Add",
    "myTransformerBlock",
    "fusehd",
    "getreturnir",
    "getreturnrgb",
    "getreturnfeature",
    "getreturnfeature2",
    "getreturnfuse",
    "getreturnfhigh",
    "chnnel_swapping_Vmamba",
    "CrossLayer",
    "fusehdxiaorong",
    "fusehdxiaorong2",
    "fusehdxiaorong3",
    "fusehdxiaorong4",
    "fusehdquanzhong",
    "fusehd23",
    "fusehd33",
    "fusehdwithoutlow",
    "fusehdwithoutlh",
    "fusehdwithouthl",
    "fusehdwithouthh",
)




def l1_norm(feature_map):  
    """  
    Compute the L1-normalized version of the input feature map.  
    """  
    l1_norm = torch.sum(torch.abs(feature_map), dim=-1, keepdim=True)  
    normalized_feature_map = feature_map / (l1_norm + 1e-8)  # Add small value to prevent division by zero  
    return normalized_feature_map  

def softmax_fusion(feature_map_a, feature_map_b):  
    """  
    Fuse two L1-normalized feature maps using the Softmax function.  
    """  
    exp_a = torch.exp(feature_map_a)  
    exp_b = torch.exp(feature_map_b)  
    sum_exp = exp_a + exp_b  
    softmax_a = exp_a / (sum_exp + 1e-8)  # Add small value to prevent division by zero  
    softmax_b = exp_b / (sum_exp + 1e-8)  
    
    fused_feature_map = softmax_a * feature_map_a + softmax_b * feature_map_b  
    return fused_feature_map  
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None, 
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            #torch.arange(1, d_state + 1, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    #得到可以依据输入变化的参数A、B、C、D并且计算最终的输出
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    


    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        #assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = y.to(x.dtype)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        d=self.ln_1(input)
        #d=d.to(torch.float32)
        c=self.self_attention(d)
        #c=c.to(torch.float32)
        y=self.drop_path(c)
        x = input+y
        return x
class chnnel_swapping_Vmamba(nn.Module):
    """
    Build chnnel swapping Vmamba module
    input:x1,x2 with the shape of "B, C, H, W"
    output:x1,x2 with the shape of "B, C, H, W"
    """ 
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        self.blocks2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample1 = downsample(dim=dim, norm_layer=norm_layer)
            self.downsample2 = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample1 = None
            self.downsample2 = None
        
    def forward(self, x):
        x1,x2 = x[0], x[1]
        #print(x[0].shape,x[1].shape)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        assert x1.shape == x2.shape, "Inputs must have the same shape."
        chnnels = x1.shape[-1]

        split_size = chnnels // 4
        x1_part1, x1_part2, x1_part3, x1_part4 = torch.split(x1, split_size, dim=-1)
        x2_part1, x2_part2, x2_part3, x2_part4 = torch.split(x2, split_size, dim=-1)

        x1 = torch.cat([x1_part1, x2_part2, x1_part3, x2_part4],dim=-1)
        x2 = torch.cat([x2_part1, x1_part2, x2_part3, x1_part4],dim=-1)
        
        for blk in self.blocks1:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1)
            else:
                x1 = x1.cuda()
                x1 = blk(x1)
        
        for blk in self.blocks2:
            if self.use_checkpoint:
                x2 = checkpoint.checkpoint(blk, x2)
            else:
                x2 = x2.cuda()
                x2 = blk(x2)
        
        if self.downsample1 is not None:
            x1 = self.downsample1(x1)
        if self.downsample2 is not None:
            x2 = self.downsample2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        #print(x1.shape)
        return x1, x2
####################################################################### End ###############################################################################
    

########################################################### Cross modal Vmamba module #####################################################################
class Cross_modal_Vmaba_module(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None, 
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        self.x_proj_weight2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj
        
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_weight2 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        self.dt_projs_bias2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.A_logs2 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds2 = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.selective_scan = selective_scan_fn
        self.forward_core1 = self.forward_corev0
        self.forward_core2 = self.forward_corev0

        self.out_norm1 = nn.LayerNorm(self.d_inner)
        self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else None
        self.dropout2 = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            #torch.arange(1, d_state + 1, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    #得到可以依据输入变化的参数A、B、C、D并且计算最终的输出
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward_corev3(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        for i in range(0,B):
            for j in range(0,4):
               for k in range(0,C):
                  for m in range(0,W):
                      if m % 2 ==0:
                         xs[i][j][k][W*m:(m+1)*W]=xs[i][j][k][W*m:(m+1)*W].flip(0)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight2)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight2)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds2.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs2.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias2.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward(self, xs: Tuple [torch.Tensor, torch.Tensor], **kwargs):
        x1, x2 = xs
        assert x1.shape == x2.shape, "Inputs must have the same shape."
        B, H, W, C = x1.shape

        xz1 = self.in_proj1(x1)
        xz2 = self.in_proj2(x2)
        x1, z1 = xz1.chunk(2, dim=-1) # (b, h, w, d)
        x2, z2 = xz2.chunk(2, dim=-1) # (b, h, w, d)
        
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x2 = x2.permute(0, 3, 1, 2).contiguous()

        x1 = self.act1(self.conv2d1(x1)) # (b, d, h, w)
        x2 = self.act1(self.conv2d2(x2)) # (b, d, h, w)

        y1_1, y2_1, y3_1, y4_1 = self.forward_core1(x1)
        y1_2, y2_2, y3_2, y4_2 = self.forward_core2(x2)
        #assert y1.dtype == torch.float32
        y1 = y1_1 + y2_1 + y3_1 + y4_1
        y2 = y1_2 + y2_2 + y3_2 + y4_2
        y1 = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y2 = torch.transpose(y2, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y1 = y1.to(x1.dtype)
        y2 = y2.to(x2.dtype)
        y1 = self.out_norm1(y1)
        y2 = self.out_norm2(y2)
        
        y11 = y1 * F.silu(z1)
        y21 = y2 * F.silu(z1)
        y1 = y11 + y21
        out1 = self.out_proj1(y1)
        if self.dropout1 is not None:
            out1 = self.dropout1(out1)
        
        y12 = y1 * F.silu(z2)
        y22 = y2 * F.silu(z2)
        y2 = y22 + y12
        out2 = self.out_proj2(y2)
        if self.dropout2 is not None:
            out2 = self.dropout2(out2)
        return out1, out2

class Cross_Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = Cross_modal_Vmaba_module(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, inputs: Tuple [torch.Tensor, torch.Tensor]):
        input1, input2 = inputs
        d1 = self.ln_1(input1)
        d2 = self.ln_2(input2)
        c1, c2 = self.self_attention((d1, d2))
        y1 = self.drop_path1(c1)
        y2 = self.drop_path2(c2)
        x1 = input1 + y1
        x2 = input2 + y2
        return x1, x2

class CrossLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Cross_Block(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        self.conv = nn.Conv2d(2, 1, kernel_size=7,
                              padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(2, 1, kernel_size=7,
                              padding=7 // 2, bias=False)
        self.sigmoid2 = nn.Sigmoid()
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample1 = downsample(dim=dim, norm_layer=norm_layer)
            self.downsample2 = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample1 = None
            self.downsample2 = None


    def forward(self, x):
        x1, x2 = x[0], x[1]
        #chnnels = x.shape[-1]
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x = (x1, x2)
        for blk in self.blocks:
            if self.use_checkpoint:
                x1, x2 = checkpoint.checkpoint(blk, x)
            else:
                x1, x2 = blk(x)
        
        if self.downsample1 is not None:
            x1 = self.downsample1(x1)
        if self.downsample2 is not None:
            x2 = self.downsample2(x2)
        x1=x1.permute(0, 3, 1, 2)
        x2=x2.permute(0, 3, 1, 2)
        '''max_out, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out = torch.mean(x1, dim=1, keepdim=True)
        spatial_out = x1*self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x1 = spatial_out * x1
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        avg_out = torch.mean(x2, dim=1, keepdim=True)
        spatial_out = x2*self.sigmoid2(self.conv2(torch.cat([max_out, avg_out], dim=1)))
        x2 = spatial_out * x2'''
        #print(type(x1),type(x2))
        # print(x1.shape)
        # print(x2.shape)
        return x1, x2
class fusehdxiaorong(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xfuse=(xir+xrgb)*0.5
        xirreturn=xir+xfuse
        xrgbreturn=xrgb+xfuse
        #xfuse2=xfuse
        return [xirreturn,xrgbreturn]
class fusehdxiaorong2(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        xrgbf = self.wt_function(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5
        xfuselow=(xirflow+xrgbflow)*0.5
        xirreturn=xfuselow+xirlowcancha
        xrgbreturn=xfuselow+xrgblowcancha
        xirlow=torch.unsqueeze(xirreturn,2)
        xrgblow=torch.unsqueeze(xrgbreturn,2)
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xfuse2=xfuse
        return [xirreturn,xrgbreturn,xirfuse,xrgbfuse]
class fusehdxiaorong3(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        xrgbf = self.wt_function(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5
        xirlow,xrgblow=self.swap([xirflow,xrgbflow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        xirreturn=xirlow+xirlowcancha
        xrgbreturn=xrgblow+xrgblowcancha
        xirlow=torch.unsqueeze(xirreturn,2)
        xrgblow=torch.unsqueeze(xrgbreturn,2)
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xfuse2=xfuse
        return [xirreturn,xrgbreturn,xirfuse,xrgbfuse]
class fusehdxiaorong4(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        #self.swap=chnnel_swapping_Vmamba(c1,1)
        #self.cross=CrossLayer(c1,1)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        xrgbf = self.wt_function(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        xfuselow=(xirflow+xrgbflow)*0.5
        xirreturn=xfuselow+xirlowcancha
        xrgbreturn=xfuselow+xrgblowcancha
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        xirlow=torch.unsqueeze(xirreturn,2)
        xrgblow=torch.unsqueeze(xrgbreturn,2)
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xfuse2=xfuse
        return [xirreturn,xrgbreturn,xirfuse,xrgbfuse]
class fusehdquanzhong(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        self.conv1=nn.Conv2d(2*c1, c1, 1, 1, bias=False)
        self.bn1=nn.BatchNorm2d(c1)
        self.act1=nn.SiLU() 
        self.conv2=nn.Conv2d(2*c1, c1, 1, 1, bias=False)
        self.bn2=nn.BatchNorm2d(c1)
        self.act2=nn.SiLU()
        self.conv3=nn.Conv2d(2*c1, c1, 1, 1, bias=False)
        self.bn3=nn.BatchNorm2d(c1)
        self.act3=nn.SiLU() 
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        xrgbf = self.wt_function(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        xirlow,xrgblow=self.swap([xirflow,xrgbflow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        xirreturn=xirlow+xirlowcancha
        xrgbreturn=xrgblow+xrgblowcancha
        high1=torch.cat([xirfhigh1,xrgbfhigh1],dim=1)
        high2=torch.cat([xirfhigh2,xrgbfhigh2],dim=1)
        high3=torch.cat([xirfhigh3,xrgbfhigh3],dim=1)
        high1=self.conv1(high1)
        high1=self.bn1(high1)
        xfusehigh1=self.act1(high1)
        high2=self.conv2(high2)
        high2=self.bn2(high2)
        xfusehigh2=self.act2(high2)
        high3=self.conv3(high3)
        high3=self.bn3(high3)
        xfusehigh3=self.act3(high3)
        xirlow=torch.unsqueeze(xirreturn,2)
        xrgblow=torch.unsqueeze(xrgbreturn,2)
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xfuse2=xfuse
        return [xirreturn,xrgbreturn,xirfuse,xrgbfuse]
class fusehd(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        '''self.cross2=CrossLayer(c1,1)
        self.cross3=CrossLayer(c1,1)
        self.cross4=CrossLayer(c1,1)
        self.cross5=CrossLayer(c1,1)
        self.cross6=CrossLayer(c1,1)
        self.cross7=CrossLayer(c1,1)
        self.cross8=CrossLayer(c1,1)'''
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(x[0].shape,x[1].shape)
        #print("77")
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        #print(xirfhigh2.shape)
        xrgbf = self.wt_function(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        '''xirfhigh1 = l1_norm(xirfhigh1)  
        xrgbfhigh1 = l1_norm(xrgbfhigh1)
        xfusehigh1=softmax_fusion(xirfhigh1,xrgbfhigh1)
        xirfhigh2 = l1_norm(xirfhigh2)  
        xrgbfhigh2 = l1_norm(xrgbfhigh2)
        xfusehigh2=softmax_fusion(xirfhigh2,xrgbfhigh2)
        xirfhigh3 = l1_norm(xirfhigh3)  
        xrgbfhigh3 = l1_norm(xrgbfhigh3)
        xfusehigh3=softmax_fusion(xirfhigh3,xrgbfhigh3)'''
        '''xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5'''
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        '''xfusehigh1=torch.max(xirfhigh1,xrgbfhigh1)
        xfusehigh2=torch.max(xirfhigh2,xrgbfhigh2)
        xfusehigh3=torch.max(xirfhigh3,xrgbfhigh3)'''
        '''if xfusehigh3.shape[3]==80:
           for i in range(0,32):
              #print(xirfhigh1[0][i].shape)
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir3.jpg'
              plt.imshow(xirfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb3.jpg'
              plt.imshow(xrgbfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse3.jpg'
              plt.imshow(xfusehigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()'''
        #print(xfusehigh3.shape)
        '''xfusehigh1=xrgbfhigh1
        xfusehigh2=xrgbfhigh2
        xfusehigh3=xrgbfhigh3'''
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        #print(xirlow.shape)
        #print("2")
        xirlow,xrgblow=self.swap([xirlow,xrgblow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross2([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross3([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross4([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross5([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross6([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross7([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross8([xirlow,xrgblow])
        #print(xirlow.shape)
        #xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        '''if xirlow.shape[3]==80:
             files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(0)+'fuse3.jpg'
             plt.imshow(xirlow[0][4].cpu())
             plt.axis("off")
             plt.savefig(files3,dpi=300, bbox_inches="tight")
             plt.close()'''   
        #print(xirfhigh3.shape)
        #print("1")
        #print(xfusehigh1.shape)
        #print(xirlow.shape)
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class fusehdwithouthh(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        '''self.cross2=CrossLayer(c1,1)
        self.cross3=CrossLayer(c1,1)
        self.cross4=CrossLayer(c1,1)
        self.cross5=CrossLayer(c1,1)
        self.cross6=CrossLayer(c1,1)
        self.cross7=CrossLayer(c1,1)
        self.cross8=CrossLayer(c1,1)'''
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(x[0].shape,x[1].shape)
        #print("77")
        xirf = self.wt_function(xir)
        xirf[:,:,3,:,:]=0.0001
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        #print(xirfhigh2.shape)
        xrgbf = self.wt_function(xrgb)
        xrgbf[:,:,3,:,:]=0.0001
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        '''xirfhigh1 = l1_norm(xirfhigh1)  
        xrgbfhigh1 = l1_norm(xrgbfhigh1)
        xfusehigh1=softmax_fusion(xirfhigh1,xrgbfhigh1)
        xirfhigh2 = l1_norm(xirfhigh2)  
        xrgbfhigh2 = l1_norm(xrgbfhigh2)
        xfusehigh2=softmax_fusion(xirfhigh2,xrgbfhigh2)
        xirfhigh3 = l1_norm(xirfhigh3)  
        xrgbfhigh3 = l1_norm(xrgbfhigh3)
        xfusehigh3=softmax_fusion(xirfhigh3,xrgbfhigh3)'''
        '''xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5'''
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        '''xfusehigh1=torch.max(xirfhigh1,xrgbfhigh1)
        xfusehigh2=torch.max(xirfhigh2,xrgbfhigh2)
        xfusehigh3=torch.max(xirfhigh3,xrgbfhigh3)'''
        '''if xfusehigh3.shape[3]==80:
           for i in range(0,32):
              #print(xirfhigh1[0][i].shape)
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir3.jpg'
              plt.imshow(xirfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb3.jpg'
              plt.imshow(xrgbfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse3.jpg'
              plt.imshow(xfusehigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()'''
        #print(xfusehigh3.shape)
        '''xfusehigh1=xrgbfhigh1
        xfusehigh2=xrgbfhigh2
        xfusehigh3=xrgbfhigh3'''
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        #print(xirlow.shape)
        #print("2")
        xirlow,xrgblow=self.swap([xirlow,xrgblow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross2([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross3([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross4([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross5([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross6([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross7([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross8([xirlow,xrgblow])
        #print(xirlow.shape)
        #xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        '''if xirlow.shape[3]==80:
             files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(0)+'fuse3.jpg'
             plt.imshow(xirlow[0][4].cpu())
             plt.axis("off")
             plt.savefig(files3,dpi=300, bbox_inches="tight")
             plt.close()'''   
        #print(xirfhigh3.shape)
        #print("1")
        #print(xfusehigh1.shape)
        #print(xirlow.shape)
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class fusehdwithouthl(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        '''self.cross2=CrossLayer(c1,1)
        self.cross3=CrossLayer(c1,1)
        self.cross4=CrossLayer(c1,1)
        self.cross5=CrossLayer(c1,1)
        self.cross6=CrossLayer(c1,1)
        self.cross7=CrossLayer(c1,1)
        self.cross8=CrossLayer(c1,1)'''
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(x[0].shape,x[1].shape)
        #print("77")
        xirf = self.wt_function(xir)
        xirf[:,:,2,:,:]=xirf[:,:,2,:,:]*0.0001
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        #print(xirfhigh2.shape)
        xrgbf = self.wt_function(xrgb)
        xrgbf[:,:,2,:,:]=xrgbf[:,:,2,:,:]*0.0001
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        '''xirfhigh1 = l1_norm(xirfhigh1)  
        xrgbfhigh1 = l1_norm(xrgbfhigh1)
        xfusehigh1=softmax_fusion(xirfhigh1,xrgbfhigh1)
        xirfhigh2 = l1_norm(xirfhigh2)  
        xrgbfhigh2 = l1_norm(xrgbfhigh2)
        xfusehigh2=softmax_fusion(xirfhigh2,xrgbfhigh2)
        xirfhigh3 = l1_norm(xirfhigh3)  
        xrgbfhigh3 = l1_norm(xrgbfhigh3)
        xfusehigh3=softmax_fusion(xirfhigh3,xrgbfhigh3)'''
        '''xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5'''
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        '''xfusehigh1=torch.max(xirfhigh1,xrgbfhigh1)
        xfusehigh2=torch.max(xirfhigh2,xrgbfhigh2)
        xfusehigh3=torch.max(xirfhigh3,xrgbfhigh3)'''
        '''if xfusehigh3.shape[3]==80:
           for i in range(0,32):
              #print(xirfhigh1[0][i].shape)
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir3.jpg'
              plt.imshow(xirfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb3.jpg'
              plt.imshow(xrgbfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse3.jpg'
              plt.imshow(xfusehigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()'''
        #print(xfusehigh3.shape)
        '''xfusehigh1=xrgbfhigh1
        xfusehigh2=xrgbfhigh2
        xfusehigh3=xrgbfhigh3'''
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        #print(xirlow.shape)
        #print("2")
        xirlow,xrgblow=self.swap([xirlow,xrgblow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross2([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross3([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross4([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross5([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross6([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross7([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross8([xirlow,xrgblow])
        #print(xirlow.shape)
        #xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        '''if xirlow.shape[3]==80:
             files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(0)+'fuse3.jpg'
             plt.imshow(xirlow[0][4].cpu())
             plt.axis("off")
             plt.savefig(files3,dpi=300, bbox_inches="tight")
             plt.close()'''   
        #print(xirfhigh3.shape)
        #print("1")
        #print(xfusehigh1.shape)
        #print(xirlow.shape)
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class fusehd23(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter1, self.iwt_filter1 = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter1 = nn.Parameter(self.wt_filter1, requires_grad=False)
        #self.iwt_filter1 = nn.Parameter(self.iwt_filter1, requires_grad=False)
        self.wt_function1 = partial(wavelet_transform, filters = self.wt_filter1)
        #self.iwt_function1 = partial(inverse_wavelet_transform, filters = self.iwt_filter1)
        self.wt_filter2, self.iwt_filter2 = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter2 = nn.Parameter(self.wt_filter2, requires_grad=False)
        self.iwt_filter2 = nn.Parameter(self.iwt_filter2, requires_grad=False)
        self.wt_function2 = partial(wavelet_transform, filters = self.wt_filter2)
        self.iwt_function2 = partial(inverse_wavelet_transform, filters = self.iwt_filter2)
        self.wt_filter3, self.iwt_filter3 = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter3 = nn.Parameter(self.wt_filter3, requires_grad=False)
        self.iwt_filter3 = nn.Parameter(self.iwt_filter3, requires_grad=False)
        self.wt_function3 = partial(wavelet_transform, filters = self.wt_filter3)
        self.iwt_function3 = partial(inverse_wavelet_transform, filters = self.iwt_filter3)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        self.swap2=chnnel_swapping_Vmamba(c1,1)
        self.cross2=CrossLayer(c1,1)
        self.swap3=chnnel_swapping_Vmamba(c1,1)
        self.cross3=CrossLayer(c1,1)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xirf = self.wt_function1(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        
        xirf2=self.wt_function2(xirflow)
        xirflow2=xirf2[:,:,0,:,:]
        xirlowcancha2=xirf2[:,:,0,:,:]
        xirfhigh23=xirf2[:,:,3,:,:]
        xirfhigh21=xirf2[:,:,1,:,:]
        xirfhigh22=xirf2[:,:,2,:,:]
        
        xirf3=self.wt_function3(xirflow2)
        xirflow3=xirf3[:,:,0,:,:]
        xirlowcancha3=xirf3[:,:,0,:,:]
        xirfhigh33=xirf3[:,:,3,:,:]
        xirfhigh31=xirf3[:,:,1,:,:]
        xirfhigh32=xirf3[:,:,2,:,:]
        
        xrgbf = self.wt_function1(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        
        xrgbf2 = self.wt_function2(xrgbflow)
        xrgblowcancha2=xrgbf2[:,:,0,:,:]
        xrgbflow2=xrgbf2[:,:,0,:,:]
        xrgbfhigh23=xrgbf2[:,:,3,:,:]
        xrgbfhigh21=xrgbf2[:,:,1,:,:]
        xrgbfhigh22=xrgbf2[:,:,2,:,:]
        
        xrgbf3 = self.wt_function3(xrgbflow2)
        xrgblowcancha3=xrgbf3[:,:,0,:,:]
        xrgbflow3=xrgbf3[:,:,0,:,:]
        xrgbfhigh33=xrgbf3[:,:,3,:,:]
        xrgbfhigh31=xrgbf3[:,:,1,:,:]
        xrgbfhigh32=xrgbf3[:,:,2,:,:]
        
        temprgb1=torch.abs(xrgbfhigh31)
        temprgb2=torch.abs(xrgbfhigh32)
        temprgb3=torch.abs(xrgbfhigh33)
        tempir1=torch.abs(xirfhigh31)
        tempir2=torch.abs(xirfhigh32)
        tempir3=torch.abs(xirfhigh33)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh31=max1*xrgbfhigh31+max2*xirfhigh31
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh32=max1*xrgbfhigh32+max2*xirfhigh32
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh33=max1*xrgbfhigh33+max2*xirfhigh33
        xirlow3=torch.squeeze(xirflow3,2)
        xrgblow3=torch.squeeze(xrgbflow3,2)
        xirlow3,xrgblow3=self.swap3([xirlow3,xrgblow3])
        xirlow3,xrgblow3=self.cross3([xirlow3,xrgblow3])
        xirlow3=xirlow3+xirlowcancha3
        xrgblow3=xrgblow3+xrgblowcancha3
        xirlow3=torch.unsqueeze(xirlow3,2)
        xfusehigh31=torch.unsqueeze(xfusehigh31,2)
        xfusehigh32=torch.unsqueeze(xfusehigh32,2)
        xfusehigh33=torch.unsqueeze(xfusehigh33,2)
        xrgblow3=torch.unsqueeze(xrgblow3,2)
        xirfuse3=torch.cat([xirlow3,xfusehigh31,xfusehigh32,xfusehigh33],dim=2)
        xrgbfuse3=torch.cat([xrgblow3,xfusehigh31,xfusehigh32,xfusehigh33],dim=2)
        xirreturn3 = self.iwt_function3(xirfuse3)
        xrgbreturn3=self.iwt_function3(xrgbfuse3)
        
        xrgbflow2=xrgbflow2+xrgbreturn3
        xirflow2= xirflow2+xirreturn3
        
        temprgb1=torch.abs(xrgbfhigh21)
        temprgb2=torch.abs(xrgbfhigh22)
        temprgb3=torch.abs(xrgbfhigh23)
        tempir1=torch.abs(xirfhigh21)
        tempir2=torch.abs(xirfhigh22)
        tempir3=torch.abs(xirfhigh23)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh21=max1*xrgbfhigh21+max2*xirfhigh21
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh22=max1*xrgbfhigh22+max2*xirfhigh22
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh23=max1*xrgbfhigh23+max2*xirfhigh23
        xirlow2=torch.squeeze(xirflow2,2)
        xrgblow2=torch.squeeze(xrgbflow2,2)
        xirlow2,xrgblow2=self.swap2([xirlow2,xrgblow2])
        xirlow2,xrgblow2=self.cross2([xirlow2,xrgblow2])
        xirlow2=xirlow2+xirlowcancha2
        xrgblow2=xrgblow2+xrgblowcancha2
        xirlow2=torch.unsqueeze(xirlow2,2)
        xfusehigh21=torch.unsqueeze(xfusehigh21,2)
        xfusehigh22=torch.unsqueeze(xfusehigh22,2)
        xfusehigh23=torch.unsqueeze(xfusehigh23,2)
        xrgblow2=torch.unsqueeze(xrgblow2,2)
        xirfuse2=torch.cat([xirlow2,xfusehigh21,xfusehigh22,xfusehigh23],dim=2)
        xrgbfuse2=torch.cat([xrgblow2,xfusehigh21,xfusehigh22,xfusehigh23],dim=2)
        xirreturn2 = self.iwt_function2(xirfuse2)
        xrgbreturn2=self.iwt_function2(xrgbfuse2)
        
        xrgbflow=xrgbflow+xrgbreturn2
        xirflow= xirflow+xirreturn2
        
        
        
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        xirlow,xrgblow=self.swap([xirlow,xrgblow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        #print(xirlow.shape,xrgblow.shape)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class fusehdwithoutlh(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        '''self.cross2=CrossLayer(c1,1)
        self.cross3=CrossLayer(c1,1)
        self.cross4=CrossLayer(c1,1)
        self.cross5=CrossLayer(c1,1)
        self.cross6=CrossLayer(c1,1)
        self.cross7=CrossLayer(c1,1)
        self.cross8=CrossLayer(c1,1)'''
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(x[0].shape,x[1].shape)
        #print("77")
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirf[:,:,1,:,:]=xirf[:,:,1,:,:]*0.0001
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        #print(xirfhigh2.shape)
        xrgbf = self.wt_function(xrgb)
        xrgbf[:,:,1,:,:]=xrgbf[:,:,1,:,:]*0.0001
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        '''xirfhigh1 = l1_norm(xirfhigh1)  
        xrgbfhigh1 = l1_norm(xrgbfhigh1)
        xfusehigh1=softmax_fusion(xirfhigh1,xrgbfhigh1)
        xirfhigh2 = l1_norm(xirfhigh2)  
        xrgbfhigh2 = l1_norm(xrgbfhigh2)
        xfusehigh2=softmax_fusion(xirfhigh2,xrgbfhigh2)
        xirfhigh3 = l1_norm(xirfhigh3)  
        xrgbfhigh3 = l1_norm(xrgbfhigh3)
        xfusehigh3=softmax_fusion(xirfhigh3,xrgbfhigh3)'''
        '''xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5'''
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        '''xfusehigh1=torch.max(xirfhigh1,xrgbfhigh1)
        xfusehigh2=torch.max(xirfhigh2,xrgbfhigh2)
        xfusehigh3=torch.max(xirfhigh3,xrgbfhigh3)'''
        '''if xfusehigh3.shape[3]==80:
           for i in range(0,32):
              #print(xirfhigh1[0][i].shape)
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir3.jpg'
              plt.imshow(xirfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb3.jpg'
              plt.imshow(xrgbfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse3.jpg'
              plt.imshow(xfusehigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()'''
        #print(xfusehigh3.shape)
        '''xfusehigh1=xrgbfhigh1
        xfusehigh2=xrgbfhigh2
        xfusehigh3=xrgbfhigh3'''
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        #print(xirlow.shape)
        #print("2")
        xirlow,xrgblow=self.swap([xirlow,xrgblow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross2([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross3([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross4([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross5([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross6([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross7([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross8([xirlow,xrgblow])
        #print(xirlow.shape)
        #xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        '''if xirlow.shape[3]==80:
             files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(0)+'fuse3.jpg'
             plt.imshow(xirlow[0][4].cpu())
             plt.axis("off")
             plt.savefig(files3,dpi=300, bbox_inches="tight")
             plt.close()'''   
        #print(xirfhigh3.shape)
        #print("1")
        #print(xfusehigh1.shape)
        #print(xirlow.shape)
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class fusehdwithoutlow(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        #self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        #self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        #self.swap=chnnel_swapping_Vmamba(c1,1)
        #self.cross=CrossLayer(c1,1)
        '''self.cross2=CrossLayer(c1,1)
        self.cross3=CrossLayer(c1,1)
        self.cross4=CrossLayer(c1,1)
        self.cross5=CrossLayer(c1,1)
        self.cross6=CrossLayer(c1,1)
        self.cross7=CrossLayer(c1,1)
        self.cross8=CrossLayer(c1,1)'''
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(x[0].shape,x[1].shape)
        #print("77")
        xirf = self.wt_function(xir)
        xirf[:,:,0,:,:]=0.0001
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        #print(xirfhigh2.shape)
        xrgbf = self.wt_function(xrgb)
        xrgbf[:,:,0,:,:]=0.0001
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        '''xirfhigh1 = l1_norm(xirfhigh1)  
        xrgbfhigh1 = l1_norm(xrgbfhigh1)
        xfusehigh1=softmax_fusion(xirfhigh1,xrgbfhigh1)
        xirfhigh2 = l1_norm(xirfhigh2)  
        xrgbfhigh2 = l1_norm(xrgbfhigh2)
        xfusehigh2=softmax_fusion(xirfhigh2,xrgbfhigh2)
        xirfhigh3 = l1_norm(xirfhigh3)  
        xrgbfhigh3 = l1_norm(xrgbfhigh3)
        xfusehigh3=softmax_fusion(xirfhigh3,xrgbfhigh3)'''
        '''xfusehigh1=(xirfhigh1+xrgbfhigh1)*0.5
        xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xfusehigh3=(xirfhigh3+xrgbfhigh3)*0.5'''
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        '''xfusehigh1=torch.max(xirfhigh1,xrgbfhigh1)
        xfusehigh2=torch.max(xirfhigh2,xrgbfhigh2)
        xfusehigh3=torch.max(xirfhigh3,xrgbfhigh3)'''
        '''if xfusehigh3.shape[3]==80:
           for i in range(0,32):
              #print(xirfhigh1[0][i].shape)
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'ir3.jpg'
              plt.imshow(xirfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xirfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'rgb3.jpg'
              plt.imshow(xrgbfhigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xrgbfhigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()
              files="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse1.jpg'
              files2="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse2.jpg'
              files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(i)+'fuse3.jpg'
              plt.imshow(xfusehigh1[0][i].cpu())
              plt.axis("off")
              plt.savefig(files,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh2[0][i].cpu())
              plt.axis("off")
              plt.savefig(files2,dpi=300, bbox_inches="tight")
              plt.close()
              plt.imshow(xfusehigh3[0][i].cpu())
              plt.axis("off")
              plt.savefig(files3,dpi=300, bbox_inches="tight")
              plt.close()'''
        #print(xfusehigh3.shape)
        '''xfusehigh1=xrgbfhigh1
        xfusehigh2=xrgbfhigh2
        xfusehigh3=xrgbfhigh3'''
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        #print(xirlow.shape)
        #print("2")
        #xirlow,xrgblow=self.swap([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross2([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross3([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross4([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross5([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross6([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross7([xirlow,xrgblow])
        #xirlow,xrgblow=self.cross8([xirlow,xrgblow])
        #print(xirlow.shape)
        #xfusehigh2=(xirfhigh2+xrgbfhigh2)*0.5
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        '''if xirlow.shape[3]==80:
             files3="/data1/code/dwh2/TwoStream_Yolov8-main/ultralytics/results/"+str(0)+'fuse3.jpg'
             plt.imshow(xirlow[0][4].cpu())
             plt.axis("off")
             plt.savefig(files3,dpi=300, bbox_inches="tight")
             plt.close()'''   
        #print(xirfhigh3.shape)
        #print("1")
        #print(xfusehigh1.shape)
        #print(xirlow.shape)
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class fusehd33(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter1, self.iwt_filter1 = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter1 = nn.Parameter(self.wt_filter1, requires_grad=False)
        #self.iwt_filter1 = nn.Parameter(self.iwt_filter1, requires_grad=False)
        self.wt_function1 = partial(wavelet_transform, filters = self.wt_filter1)
        #self.iwt_function1 = partial(inverse_wavelet_transform, filters = self.iwt_filter1)
        self.wt_filter2, self.iwt_filter2 = create_wavelet_filter("haar", c1, c1, torch.float)
        self.wt_filter2 = nn.Parameter(self.wt_filter2, requires_grad=False)
        self.iwt_filter2 = nn.Parameter(self.iwt_filter2, requires_grad=False)
        self.wt_function2 = partial(wavelet_transform, filters = self.wt_filter2)
        self.iwt_function2 = partial(inverse_wavelet_transform, filters = self.iwt_filter2)
        self.swap=chnnel_swapping_Vmamba(c1,1)
        self.cross=CrossLayer(c1,1)
        self.swap2=chnnel_swapping_Vmamba(c1,1)
        self.cross2=CrossLayer(c1,1)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        xirf = self.wt_function1(xir)
        xirflow=xirf[:,:,0,:,:]
        xirlowcancha=xirf[:,:,0,:,:]
        xirfhigh3=xirf[:,:,3,:,:]
        xirfhigh1=xirf[:,:,1,:,:]
        xirfhigh2=xirf[:,:,2,:,:]
        
        xirf2=self.wt_function2(xirflow)
        xirflow2=xirf2[:,:,0,:,:]
        xirlowcancha2=xirf2[:,:,0,:,:]
        xirfhigh23=xirf2[:,:,3,:,:]
        xirfhigh21=xirf2[:,:,1,:,:]
        xirfhigh22=xirf2[:,:,2,:,:]
        
        
        xrgbf = self.wt_function1(xrgb)
        xrgblowcancha=xrgbf[:,:,0,:,:]
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh3=xrgbf[:,:,3,:,:]
        xrgbfhigh1=xrgbf[:,:,1,:,:]
        xrgbfhigh2=xrgbf[:,:,2,:,:]
        
        xrgbf2 = self.wt_function2(xrgbflow)
        xrgblowcancha2=xrgbf2[:,:,0,:,:]
        xrgbflow2=xrgbf2[:,:,0,:,:]
        xrgbfhigh23=xrgbf2[:,:,3,:,:]
        xrgbfhigh21=xrgbf2[:,:,1,:,:]
        xrgbfhigh22=xrgbf2[:,:,2,:,:]
        
        
        
        
        temprgb1=torch.abs(xrgbfhigh21)
        temprgb2=torch.abs(xrgbfhigh22)
        temprgb3=torch.abs(xrgbfhigh23)
        tempir1=torch.abs(xirfhigh21)
        tempir2=torch.abs(xirfhigh22)
        tempir3=torch.abs(xirfhigh23)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh21=max1*xrgbfhigh21+max2*xirfhigh21
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh22=max1*xrgbfhigh22+max2*xirfhigh22
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh23=max1*xrgbfhigh23+max2*xirfhigh23
        xirlow2=torch.squeeze(xirflow2,2)
        xrgblow2=torch.squeeze(xrgbflow2,2)
        xirlow2,xrgblow2=self.swap2([xirlow2,xrgblow2])
        xirlow2,xrgblow2=self.cross2([xirlow2,xrgblow2])
        xirlow2=xirlow2+xirlowcancha2
        xrgblow2=xrgblow2+xrgblowcancha2
        xirlow2=torch.unsqueeze(xirlow2,2)
        xfusehigh21=torch.unsqueeze(xfusehigh21,2)
        xfusehigh22=torch.unsqueeze(xfusehigh22,2)
        xfusehigh23=torch.unsqueeze(xfusehigh23,2)
        xrgblow2=torch.unsqueeze(xrgblow2,2)
        xirfuse2=torch.cat([xirlow2,xfusehigh21,xfusehigh22,xfusehigh23],dim=2)
        xrgbfuse2=torch.cat([xrgblow2,xfusehigh21,xfusehigh22,xfusehigh23],dim=2)
        xirreturn2 = self.iwt_function2(xirfuse2)
        xrgbreturn2=self.iwt_function2(xrgbfuse2)
        
        xrgbflow=xrgbflow+xrgbreturn2
        xirflow= xirflow+xirreturn2
        
        
        
        temprgb1=torch.abs(xrgbfhigh1)
        temprgb2=torch.abs(xrgbfhigh2)
        temprgb3=torch.abs(xrgbfhigh3)
        tempir1=torch.abs(xirfhigh1)
        tempir2=torch.abs(xirfhigh2)
        tempir3=torch.abs(xirfhigh3)
        max1=torch.where(temprgb1>=tempir1, 1, 0)
        max2=torch.where(tempir1>=temprgb1, 1, 0)
        xfusehigh1=max1*xrgbfhigh1+max2*xirfhigh1
        max1=torch.where(temprgb2>=tempir2, 1, 0)
        max2=torch.where(tempir2>=temprgb2, 1, 0)
        xfusehigh2=max1*xrgbfhigh2+max2*xirfhigh2
        max1=torch.where(temprgb3>=tempir3, 1, 0)
        max2=torch.where(tempir3>=temprgb3, 1, 0)
        xfusehigh3=max1*xrgbfhigh3+max2*xirfhigh3
        xirlow=torch.squeeze(xirflow,2)
        xrgblow=torch.squeeze(xrgbflow,2)
        xirlow,xrgblow=self.swap([xirlow,xrgblow])
        xirlow,xrgblow=self.cross([xirlow,xrgblow])
        xirlowreturn=xirlow
        xirlow=torch.unsqueeze(xirlow,2)
        xrgblowreturn=xrgblow
        xfusehigh1=torch.unsqueeze(xfusehigh1,2)
        xfusehigh2=torch.unsqueeze(xfusehigh2,2)
        xfusehigh3=torch.unsqueeze(xfusehigh3,2)
        xrgblow=torch.unsqueeze(xrgblow,2)
        xirfuse=torch.cat([xirlow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        xrgbfuse=torch.cat([xrgblow,xfusehigh1,xfusehigh2,xfusehigh3],dim=2)
        #xirreturn = self.iwt_function(xirfuse)
        #xrgbreturn=self.iwt_function(xrgbfuse)
        xirlow=torch.squeeze(xirflow,2)
        xirlow=xirlow+xirlowcancha
        xrgblow=torch.squeeze(xrgbflow,2)
        xrgblow=xrgblow+xrgblowcancha
        xirfhigh3=torch.squeeze(xirfhigh3,2)
        xirfhigh2=torch.squeeze(xirfhigh2,2)
        xirfhigh1=torch.squeeze(xirfhigh1,2)
        xrgbfhigh3=torch.squeeze(xrgbfhigh3,2)
        xrgbfhigh2=torch.squeeze(xrgbfhigh2,2)
        xrgbfhigh1=torch.squeeze(xrgbfhigh1,2)
        xfusehigh1=torch.squeeze(xfusehigh1,2)
        xfusehigh2=torch.squeeze(xfusehigh2,2)
        xfusehigh3=torch.squeeze(xfusehigh3,2)
        #print(xirlow.shape,xrgblow.shape)
        return [xirlow,xrgblow,xirfuse,xrgbfuse,xirfhigh1,xirfhigh2,xirfhigh3,xrgbfhigh1,xrgbfhigh2,xrgbfhigh3,xfusehigh1,xfusehigh2,xfusehigh3,xirlowreturn,xrgblowreturn]
class getreturnir(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
      def forward(self,x):
        #print(x[0].shape)
        #print("3")
        return x[0]
class getreturnrgb(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
      def forward(self,x):
        return x[1]
class getreturnfuse(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
        #self.upnear = nn.Upsample(scale_factor=2)
        #self.conv1=Conv(c1,c2,1,1)
      def forward(self,x):
        #y=torch.cat([x[0],x[1]], 1)
        #y=self.conv1(y)
        #print(y.shape)
        #print(self.upnear(y).shape)
        y=x[0]+x[1]
        return y
class getreturnfhigh(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
        #self.upnear = nn.Upsample(scale_factor=2)
      def forward(self,x):
        x1=torch.squeeze(x[2][:,:,1,:,:],2)
        x2=torch.squeeze(x[2][:,:,2,:,:],2)
        x3=torch.squeeze(x[2][:,:,3,:,:],2)
        x4=torch.squeeze(x[3][:,:,1,:,:],2)
        x5=torch.squeeze(x[3][:,:,2,:,:],2)
        x6=torch.squeeze(x[3][:,:,3,:,:],2)
        #print(self.upnear(y).shape)
        return [x1,x2,x3,x4,x5,x6]
class getreturnfeature(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
      def forward(self,x):
        #print("1")
        #print((x[2]+x[3]).shape)
        xirhigh=x[2][:,:,1:4,:,:]
        xrgbhigh=x[3][:,:,1:4,:,:]
        xhighfuse=xirhigh+xrgbhigh
        xlowfuse=x[0]+x[1]
        xlowfuse=torch.unsqueeze(xlowfuse,2)
        xirfuse=torch.cat([xlowfuse,xhighfuse],dim=2)
        #print(xirfuse.shape)
        #print("1")
        xirreturn = self.iwt_function(xirfuse)
        #print(xirreturn.shape)
        return xirreturn
class getreturnfeature2(nn.Module):
      def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("haar", c1, c1, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
      def forward(self,x):
        #print("1")
        #print((x[2]+x[3]).shape)
        xlow=x[0]
        xirhigh=x[1][2][:,:,1:4,:,:]
        xrgbhigh=x[1][3][:,:,1:4,:,:]
        xhighfuse=xirhigh+xrgbhigh
        #xlowfuse=x[0]+x[1]
        xlow=torch.unsqueeze(xlow,2)
        #print(xlow.shape,xhighfuse.shape)
        xirfuse=torch.cat([xlow,xhighfuse],dim=2)
        #print(xirfuse.shape)
        #print("1")
        xirreturn = self.iwt_function(xirfuse)
        #print(xirreturn.shape)
        return xirreturn
class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(2*d_model)
        self.ln_output = nn.LayerNorm(2*d_model)
        self.sa = SelfAttention(2*d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model, block_exp * d_model*2),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model*2, d_model*2),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x

class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        #print(x[0].shape,x[1].shape)
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
class frequencyfuse(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter("db3", c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)
        
        self.fusehigh=fusehigh(c1,c2)
        self.fuselow=fuselow(c1,c2)
        self.fusemiddle=fusemiddle(c1,c2)
    def forward(self,x):
        xir,xrgb=x[0],x[1]
        #print(xir.shape,xrgb.shape)
        xirf = self.wt_function(xir)
        xirflow=xirf[:,:,0,:,:]
        xirfhigh=xirf[:,:,3,:,:]
        xirfmiddle=xirf[:,:,1:3,:,:]
        xrgbf = self.wt_function(xrgb)
        xrgbflow=xrgbf[:,:,0,:,:]
        xrgbfhigh=xrgbf[:,:,3,:,:]
        xrgbfmiddle=xrgbf[:,:,1:3,:,:]
        fusehighrgb=self.fusehigh(xirfhigh,xrgbfhigh)
        fuselowir=self.fuselow(xirflow,xrgbflow)
        fusemiddlecom=self.fusemiddle(xirfmiddle,xrgbfmiddle)
        #print(fusehighrgb.shape)
        #print(fuselowir.shape)
        #print(fusemiddlecom.shape)
        xirflow=torch.unsqueeze(xirflow,2)
        xrgbflow=torch.unsqueeze(xrgbflow,2)
        xirfhigh=torch.unsqueeze(xirfhigh,2)
        xrgbhigh=torch.unsqueeze(xrgbfhigh,2)
        xirfuse=torch.cat([fuselowir,fusemiddlecom,xirfhigh],dim=2)
        xrgbfuse=torch.cat([xrgbflow,fusemiddlecom,fusehighrgb],dim=2)
        xirreturn = self.iwt_function(xirfuse)
        xrgbreturn=self.iwt_function(xrgbfuse)
        return [xirreturn,xrgbreturn]
        
class fusehigh(nn.Module):
    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        #print(c1)
        #print(c2)
        self.convrgb=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.normrgb=nn.BatchNorm2d(c2)
        self.CBAM=CBAM3(c2)
        self.convir=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.normir=nn.BatchNorm2d(c2)
        #self.normcom=nn.BatchNorm2d(c2)
    def forward(self,xir,xrgb):
        xshape=xir.shape
        #print(xshape)
        #print(xrgb.shape)
        xiruse=torch.squeeze(xir,2)
        xrgbuse=torch.squeeze(xrgb,2)
        #xrgborl=
        xircancha=xrgbuse
        #print(xircancha.shape)
        xrgbuse=self.convrgb(xiruse)
        #if len(xrgbuse.shape)==3:
           #xrgbuse=
        xrgbuse=self.normrgb(xrgbuse)
        xrgbuse=self.CBAM(xrgbuse)
        xiruse=self.convir(xircancha)
        xiruse=self.normir(xiruse)
        xiruse=F.silu(xiruse)
        xfuse=xiruse*xrgbuse
        #print(xfuse.shape)
        #xfuse=self.normcom(xfuse)
        xreturn=xfuse+xircancha
        xreturn=torch.unsqueeze(xreturn,2)
        return xreturn
class fuselow(nn.Module): 
    def __init__(self,c1,c2,**kwargs):
        super().__init__()
        self.normir=nn.BatchNorm2d(c2)
        self.convir=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        #self.in_proj1 = nn.Linear(c2, 64, bias=False)
        self.attentionir=Cross_modal_Vmaba_module2(d_model=c2, dropout=0, d_state=16, **kwargs)
        self.convrgb=nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.normrgb=nn.BatchNorm2d(c2)
    def forward(self,xir,xrgb):
        xiruse=torch.squeeze(xir,2)
        xrgbuse=torch.squeeze(xrgb,2)
        xrgbcancha=xiruse
        xiruse=self.convir(xrgbuse)
        xiruse=self.normir(xiruse)
        xiruse=self.attentionir(xiruse)
        xrgbuse=self.convrgb(xrgbcancha)
        xrgbuse=self.normrgb(xrgbuse)
        xrgbuse=F.silu(xrgbuse)
        #print(xiruse.shape)
        #print(xrgbuse.shape)
        xfuse=xiruse*xrgbuse
        xreturn=xfuse+xrgbcancha
        xreturn=torch.unsqueeze(xreturn,2)
        return xreturn
class fusemiddle(nn.Module):
     def __init__(self,c1,c2,h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = c1
        self.vert_anchors = 2*vert_anchors
        self.horz_anchors = 2*horz_anchors
        d_k = c1
        d_v = c1
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.vert_anchors * self.horz_anchors, 2*self.n_embd))

        # transformer
        self.trans_blocks = myTransformerBlock(c1, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
        self.ln_f = nn.LayerNorm(2*self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)
        self.conv1=nn.Conv2d(2*c1, c1, 1, 1, 0, bias=True)
        self.conv2=nn.Conv2d(c1, 2*c2, 1, 1, 0, bias=True)
        self.norm1=nn.BatchNorm2d(2*c2)
        self.apply(self._init_weights)
        
     @staticmethod
     def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
     def forward(self,xir,xrgb):
         #xfuse=torch.cat([xir,xrgb],dim=2)
         xfuse=xir+xrgb
         shape=xfuse.shape
         #print(shape)
         xfuse=xfuse.reshape(shape[0], shape[1] * 2, shape[3], shape[4])
         xcancha=xfuse
         bs, c, h, w = xcancha.shape
         xfuse = self.avgpool(xfuse)
         xfuse = xfuse.view(bs, c, -1)
         token_embeddings = xfuse.permute(0, 2, 1).contiguous()
         #print(token_embeddings.shape)
         
         x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
         x = self.trans_blocks(x)
         x = self.ln_f(x)
         x = x.view(bs, self.vert_anchors, self.horz_anchors, 2*self.n_embd)
         x = x.permute(0, 3, 1, 2)
         xfuse = x.contiguous().view(bs, 2*self.n_embd, self.vert_anchors, self.horz_anchors)
         xfuse = F.interpolate(xfuse, size=([h, w]), mode='bilinear')
         xfuse=self.conv1(xfuse)
         xfuse=self.conv2(xfuse)
         xfuse=self.norm1(xfuse)
         xfuse=F.silu(xfuse)
         xfuse=xfuse+xcancha
         xfuse=xfuse.reshape(shape[0],shape[1], 2, shape[3], shape[4])
         return xfuse
         
         
         

class Cross_modal_Vmaba_module2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None, 
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        #self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        '''self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )'''

        self.act1 = nn.SiLU()
        #self.act2 = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.forward_core1 = self.forward_corev0
        #self.forward_core2 = self.forward_corev0

        self.out_norm1 = nn.LayerNorm(self.d_inner)
        #self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        #self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else None
        #self.dropout2 = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            #torch.arange(1, d_state + 1, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    #得到可以依据输入变化的参数A、B、C、D并且计算最终的输出
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, xs, **kwargs):
        x1= xs
        x1=x1.permute(0, 2, 3, 1).contiguous()
        #assert x1.shape == x2.shape, "Inputs must have the same shape."
        B, H, W, C = x1.shape
        
        xz1 = self.in_proj1(x1)
        #xz2 = self.in_proj2(x2)
        x1, z1 = xz1.chunk(2, dim=-1) # (b, h, w, d)
        #x2, z2 = xz2.chunk(2, dim=-1) # (b, h, w, d)
        
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        #x2 = x2.permute(0, 3, 1, 2).contiguous()

        x1 = self.act1(self.conv2d1(x1)) # (b, d, h, w)
        #x2 = self.act1(self.conv2d2(x2)) # (b, d, h, w)

        y1_1, y2_1, y3_1, y4_1 = self.forward_core1(x1)
        #y1_2, y2_2, y3_2, y4_2 = self.forward_core2(x2)
        #assert y1.dtype == torch.float32
        y1 = y1_1 + y2_1 + y3_1 + y4_1
        #y2 = y1_2 + y2_2 + y3_2 + y4_2
        y1 = torch.transpose(y1, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        #y2 = torch.transpose(y2, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y1 = y1.to(x1.dtype)
        #y2 = y2.to(x2.dtype)
        y1 = self.out_norm1(y1)
        #y2 = self.out_norm2(y2)
        out1 = self.out_proj1(y1)
        if self.dropout1 is not None:
            out1 = self.dropout1(out1)
        out1 = out1.permute(0, 3, 1, 2).contiguous()
        return out1     
class ChannelAttention2(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention2(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM3(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention2(c1)
        self.spatial_attention = SpatialAttention2(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        #print(x[0].shape, x[1].shape)
        #print(torch.cat(x, self.d).shape)
        return torch.cat(x, self.d)
