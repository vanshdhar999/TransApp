#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia (Modified for TST integration)
# @description : TransApp with TST (Time Series Transformer) architecture
# @component: src/TransAppModel/
# @file : TransApp_TST.py
#
#################################################################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Optional

from src.TransAppModel.PositionalEncoding import PositionalEncoding1D, LearnablePositionalEncoding1D
from src.TransAppModel.AttentionMask import DiagonalMask, TriangularCausalMask

# Import the original embedding components from TransApp
from src.TransAppModel.TransApp import (
    Conv1dSamePadding, conv1d_same_padding, Transpose, ResUnit, DilatedBlock
)

# ============================================================================
# TST COMPONENTS IMPLEMENTATION
# ============================================================================

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, temperature: float = 1.0): 
        super().__init__()
        self.d_k = d_k
        self.temperature = temperature
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        # q: [bs, n_heads, q_len, d_k]
        # k: [bs, n_heads, d_k, k_len] 
        # v: [bs, n_heads, k_len, d_v]
        scores = torch.matmul(q, k) / (self.d_k ** 0.5 * self.temperature)
        
        if mask is not None: 
            scores.masked_fill_(mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context, attn

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, res_attention: bool = True, 
                 attn_dropout: float = 0., proj_dropout: float = 0., 
                 qkv_bias: bool = True, lsa: bool = False):
        super().__init__()
        
        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads
        
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        # Learnable temperature for scaled attention
        if lsa:
            self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))
        else:
            self.temperature = 1.0
            
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_k, temperature=self.temperature)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, 
                prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, 
                attn_mask: Optional[Tensor] = None):
        
        if K is None: K = Q
        if V is None: V = Q
        
        bs = Q.size(0)
        
        # Linear transformation and split into heads
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Attention
        if self.res_attention:
            context, attn = self.sdp_attn(q_s, k_s, v_s, mask=attn_mask)
        else:
            context, attn = self.sdp_attn(q_s, k_s, v_s, mask=attn_mask)
            
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(context)
        
        return output, attn

def get_activation_fn(activation):
    if activation.lower() == "relu":
        return nn.relu
    elif activation.lower() == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f'{activation} is not available. You can use "relu" or "gelu"')

class _TSTEncoderLayer(nn.Module):
    def __init__(self, q_len: int, d_model: int, n_heads: int, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, d_ff: int = 256, store_attn: bool = False,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., 
                 bias: bool = True, activation: str = "gelu", res_attention: bool = True, 
                 pre_norm: bool = False, pe: str = 'zero', learn_pe: bool = True, 
                 fc_dropout: float = 0., head_dropout: float = 0, padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention_type: str = 'add', 
                 verbose: bool = False, **kwargs):
        
        super().__init__()
        
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, 
                                           proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if norm == 'LayerNorm':
            self.norm_attn = nn.LayerNorm(d_model)
        elif norm == 'BatchNorm':
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            raise ValueError("norm must be either 'LayerNorm' or 'BatchNorm'")

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(fc_dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if norm == 'LayerNorm':
            self.norm_ffn = nn.LayerNorm(d_model)
        elif norm == 'BatchNorm':
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, 
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        
        ## Multi-Head attention
        src2, attn = self.self_attn(src, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src

class _TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., pre_norm=False,
                        activation='gelu', res_attention=True, n_layers=1, store_attn=False):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                           norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                           pre_norm=pre_norm, activation=activation, res_attention=res_attention,
                           store_attn=store_attn) 
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: 
                output = mod(output, prev=scores)
        else:
            for mod in self.layers: 
                output = mod(output)
        return output

# ============================================================================
# TRANSAPP WITH TST INTEGRATION
# ============================================================================

class TransApp_TST(nn.Module):
    """
    TransApp with Time Series Transformer (TST) architecture
    Maintains the same interface as original TransApp for easy replacement
    """
    
    def __init__(self, 
                 max_len=1024, c_in=1,
                 mode="classif",
                 n_embed_blocks=1, 
                 encoding_type="noencoding",
                 n_encoder_layers=3,
                 kernel_size=5,
                 d_model=64, pffn_ratio=2, n_head=4,
                 prenorm=True, norm="LayerNorm",
                 activation='gelu',
                 store_att=False, attn_dp_rate=0.2, head_dp_rate=0.1, dp_rate=0.2,
                 att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                 c_reconstruct=1, apply_gap=False, nb_class=2,
                 # TST specific parameters
                 res_attention=True, fc_dropout=0., head_dropout=0.):
        
        super().__init__()
  
        self.c_in = c_in
        self.d_model = d_model
        self.mode = mode
        self.nb_class = nb_class
        self.max_len = max_len
        
        #============ Dilated Conv Embedding (Same as TransApp) ============#
        layers = []
        for i in range(n_embed_blocks):
            layers.append(DilatedBlock(c_in=c_in if i==0 else d_model, 
                                       c_out=d_model, kernel_size=kernel_size))
        layers.append(Transpose(1, 2))
        self.EmbedBlock = torch.nn.Sequential(*layers) 
            
        #============ Positional Encoding (Enhanced) ============#
        if encoding_type == 'learnable':
            self.PosEncoding = LearnablePositionalEncoding1D(d_model, max_len=max_len)
        elif encoding_type == 'fixed':
            self.PosEncoding = PositionalEncoding1D(d_model)
        elif encoding_type == 'tst_learnable':
            # TST style learnable positional encoding
            W_pos = torch.empty((max_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
            self.W_pos = nn.Parameter(W_pos, requires_grad=True)
            self.pos_dropout = nn.Dropout(dp_rate)
            self.PosEncoding = None
        elif encoding_type == 'noencoding':
            self.PosEncoding = None
            self.W_pos = None
        else:
            raise ValueError('Type of encoding {} unknown, only "learnable", "fixed", "tst_learnable" or "noencoding" supported.'
                             .format(encoding_type))
        
        #============ TST Encoder ============#
        self.EncoderBlock = _TSTEncoder(
            q_len=max_len, 
            d_model=d_model, 
            n_heads=n_head,
            d_ff=d_model * pffn_ratio,
            norm=norm,
            attn_dropout=attn_dp_rate,
            dropout=dp_rate,
            pre_norm=prenorm,
            activation=activation,
            res_attention=res_attention,
            n_layers=n_encoder_layers,
            store_attn=store_att
        )
        
        #============ Pretraining Head (Same as TransApp) ============#
        layers = []
        layers.append(nn.Linear(d_model, c_reconstruct, bias=True))
        layers.append(nn.Dropout(head_dp_rate))
        self.PredHead = torch.nn.Sequential(*layers)
        
        #============ Classification Head (Same as TransApp) ============#
        layers = []
        if apply_gap:
            layers.append(Transpose(1,2))
            layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten(start_dim=1))
        if apply_gap:
            layers.append(nn.Linear(d_model, nb_class, bias=True))
        else:
            layers.append(nn.Linear(max_len*d_model, nb_class, bias=True))
        layers.append(nn.Dropout(head_dp_rate))
        self.ClassifHead = torch.nn.Sequential(*layers)
                      
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def freeze_params(self, model_part, rq_grad=False):
        for name, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)
    
    def forward(self, x) -> torch.Tensor:
        # Dilated Conv Embedding Block
        x = self.EmbedBlock(x)  # [B, L, D]
        
        # Add Positional Encoding
        if self.PosEncoding is not None:
            x = x + self.PosEncoding(x)
        elif hasattr(self, 'W_pos') and self.W_pos is not None:
            # TST style learnable positional encoding
            x = self.pos_dropout(x + self.W_pos)
            
        # Forward TST Encoder
        x = self.EncoderBlock(x)
        
        # Forward Head
        if self.mode=="pretraining":
            x = self.PredHead(x).permute(0, 2, 1)
        else:
            x = self.ClassifHead(x)
                      
        return x

# ============================================================================
# CONVENIENCE FUNCTION FOR MODEL INSTANTIATION
# ============================================================================

def get_transapp_tst_model(m, win, dim_model, mode="pretraining", 
                          large_version=False, path_select_core=None,
                          # TST specific options
                          use_tst_pos_encoding=True, 
                          norm="BatchNorm",  # TST typically uses BatchNorm
                          res_attention=True,
                          nb_class=2,  # Add nb_class as a parameter
                          **kwargs):
    """
    Get TransApp with TST architecture - drop-in replacement for get_model_inst
    
    Parameters:
        m: int - n channel of input time series
        win: int - length of input subsequence
        dim_model: int - model dimension
        mode: str - 'pretraining' or 'classif'
        large_version: boolean - if true, use 5 encoder layers instead of 3
        path_select_core: str - path to pretrained instance
        use_tst_pos_encoding: bool - whether to use TST-style learnable positional encoding
        norm: str - normalization type ('BatchNorm' or 'LayerNorm')
        res_attention: bool - whether to use residual attention in TST
        nb_class: int - number of classes for classification
    """
    
    encoding_type = "tst_learnable" if use_tst_pos_encoding else "noencoding"
    
    TApp = TransApp_TST(
        max_len=win, c_in=m, mode=mode,
        n_embed_blocks=1, encoding_type=encoding_type,
        n_encoder_layers=5 if large_version else 3,
        kernel_size=5, d_model=dim_model, pffn_ratio=2, 
        n_head=4 if large_version else 2,
        prenorm=True, norm=norm, activation='gelu',
        store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
        att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
        c_reconstruct=1, apply_gap=True, nb_class=nb_class,  # Use the parameter
        # TST specific
        res_attention=res_attention,
        **kwargs
    )

    if path_select_core is not None:
        try:
            checkpoint = torch.load(path_select_core, map_location='cpu', weights_only=False)
            TApp.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✅ Loaded weights from {path_select_core}")
        except Exception as e:
            print(f"⚠️ Could not load weights: {e}")
            print("   Proceeding with random initialization...")

    return TApp