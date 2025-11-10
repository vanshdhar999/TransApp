#################################################################################################################
#
# @copyright : ©2023 EDF - Modified for DMSA-TST Hybrid
# @author : Adrien Petralia (Modified for DMSA integration)
# @description : TransApp with TST (Time Series Transformer) architecture enhanced with DMSA option
# @component: src/TransAppModel/
# @file : TransApp_TST_DMSA.py
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
# DMSA-ENHANCED TST COMPONENTS IMPLEMENTATION
# ============================================================================

class _ScaledDotProductAttention_DMSA(nn.Module):
    """Enhanced Scaled Dot Product Attention with optional diagonal masking (DMSA)"""
    def __init__(self, d_k: int, temperature: float = 1.0, mask_diag: bool = False): 
        super().__init__()
        self.d_k = d_k
        self.temperature = temperature
        self.mask_diag = mask_diag
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        # q: [bs, n_heads, q_len, d_k]
        # k: [bs, n_heads, d_k, k_len] 
        # v: [bs, n_heads, k_len, d_v]
        bs, n_heads, q_len = q.size(0), q.size(1), q.size(2)
        
        scores = torch.matmul(q, k) / (self.d_k ** 0.5 * self.temperature)
        
        # Apply diagonal masking (DMSA) if enabled
        if self.mask_diag:
            diag_mask = DiagonalMask(bs, q_len, device=q.device)
            # DiagonalMask creates mask for [B, 1, L, L], we need [B, n_heads, L, L]
            dmsa_mask = diag_mask.mask.expand(-1, n_heads, -1, -1)
            scores.masked_fill_(dmsa_mask, -1e9)
        
        # Apply additional mask if provided
        if mask is not None: 
            scores.masked_fill_(mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context, attn

class _MultiHeadAttention_DMSA(nn.Module):
    """Multi-Head Attention with optional DMSA support"""
    def __init__(self, d_model: int, n_heads: int, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, res_attention: bool = True, 
                 attn_dropout: float = 0., proj_dropout: float = 0., 
                 qkv_bias: bool = True, lsa: bool = False, mask_diag: bool = False):
        super().__init__()
        
        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads
        
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.mask_diag = mask_diag
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        # Learnable temperature for scaled attention
        if lsa:
            self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))
        else:
            self.temperature = 1.0
            
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention_DMSA(d_k, temperature=self.temperature, 
                                                       mask_diag=mask_diag)
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

        # Attention with optional DMSA
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

class _TSTEncoderLayer_DMSA(nn.Module):
    """TST Encoder Layer with optional DMSA support"""
    def __init__(self, q_len: int, d_model: int, n_heads: int, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, d_ff: int = 256, store_attn: bool = False,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., 
                 bias: bool = True, activation: str = "gelu", res_attention: bool = True, 
                 pre_norm: bool = False, pe: str = 'zero', learn_pe: bool = True, 
                 fc_dropout: float = 0., head_dropout: float = 0, padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention_type: str = 'add', 
                 mask_diag: bool = False, verbose: bool = False, **kwargs):
        
        super().__init__()
        
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads

        # Multi-Head attention with DMSA support
        self.res_attention = res_attention
        self.mask_diag = mask_diag
        self.self_attn = _MultiHeadAttention_DMSA(d_model, n_heads, d_k, d_v, 
                                                 attn_dropout=attn_dropout, 
                                                 proj_dropout=dropout, 
                                                 res_attention=res_attention,
                                                 mask_diag=mask_diag)

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
        src2, attn = self.self_attn(src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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

class _TSTEncoder_DMSA(nn.Module):
    """TST Encoder with optional DMSA support"""
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, activation="gelu", 
                 res_attention=True, n_layers=1, pre_norm=False, pe='zero', learn_pe=True, 
                 fc_dropout=0., head_dropout=0., padding_var=None, attn_mask=None, 
                 res_attention_type='add', store_attn=False, mask_diag=False, verbose=False, **kwargs):
        
        super().__init__()
        
        self.layers = nn.ModuleList([_TSTEncoderLayer_DMSA(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                          attn_dropout=attn_dropout, dropout=dropout,
                                                          bias=bias, activation=activation, res_attention=res_attention,
                                                          pre_norm=pre_norm, pe=pe, learn_pe=learn_pe,
                                                          fc_dropout=fc_dropout, head_dropout=head_dropout,
                                                          padding_var=padding_var, attn_mask=attn_mask,
                                                          res_attention_type=res_attention_type, store_attn=store_attn,
                                                          mask_diag=mask_diag, verbose=verbose, **kwargs) 
                                    for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        for mod in self.layers: 
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output

# ============================================================================
# HYBRID DMSA-TST MODEL
# ============================================================================

class TransApp_TST_DMSA(nn.Module):
    """
    Hybrid TransApp model combining TST architecture with optional DMSA support
    
    This model allows testing both standard self-attention and DMSA within the TST framework
    to determine which attention mechanism works best for appliance detection tasks.
    """
    
    def __init__(self, c_in, c_out, seq_len, max_seq_len=1024, 
                 d_model=128, n_heads=16, e_layers=3, d_ff=256,
                 dropout=0.1, fc_dropout=0., head_dropout=0.,
                 individual=False, patch_len=16, stride=8, padding_patch=None,
                 revin=True, affine=True, subtract_last=False,
                 decomposition=True, kernel_size=25,
                 verbose=False, **kwargs):
        
        super().__init__()
        
        # Store configuration
        self.mask_diag = kwargs.get('mask_diag', False)  # DMSA toggle
        self.model_name = f"TST{'_DMSA' if self.mask_diag else ''}"
        
        # Print configuration
        if verbose:
            mask_type = "DMSA (Diagonally Masked)" if self.mask_diag else "Standard Self-Attention"
            print(f"🧬 Initializing Hybrid TST Model with {mask_type}")
            print(f"📊 Model: {self.model_name}, d_model: {d_model}, layers: {e_layers}, heads: {n_heads}")
        
        # ===== EMBEDDING BLOCK (from TransApp) =====
        self.embedding_block = nn.Sequential(
            # Initial 1D convolution
            Conv1dSamePadding(in_channels=c_in, out_channels=64, kernel_size=8),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            
            # Dilated convolutional blocks  
            DilatedBlock(c_in=64, c_out=64, kernel_size=3, dilation_list=[1]),
            DilatedBlock(c_in=64, c_out=64, kernel_size=3, dilation_list=[2]),  
            DilatedBlock(c_in=64, c_out=64, kernel_size=3, dilation_list=[4]),
            DilatedBlock(c_in=64, c_out=64, kernel_size=3, dilation_list=[8]),
            
            # Final projection to d_model
            Conv1dSamePadding(in_channels=64, out_channels=d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            
            # Transpose for transformer: [batch, channels, seq] -> [batch, seq, channels]
            Transpose(1, 2)
        )
        
        # ===== TST ENCODER WITH DMSA SUPPORT =====
        self.encoder = _TSTEncoder_DMSA(
            q_len=seq_len, d_model=d_model, n_heads=n_heads, d_k=None, d_v=None,
            d_ff=d_ff, norm='BatchNorm', attn_dropout=dropout, dropout=dropout,
            bias=True, activation="gelu", res_attention=True, n_layers=e_layers,
            pre_norm=False, pe='zero', learn_pe=True, fc_dropout=fc_dropout,
            head_dropout=head_dropout, padding_var=None, attn_mask=None,
            res_attention_type='add', store_attn=False, 
            mask_diag=self.mask_diag,  # DMSA configuration
            verbose=verbose
        )
        
        # ===== TASK HEADS =====
        # For pretraining (reconstruction)
        self.head_nf = d_model
        self.n_vars = 1  # Single target variable for reconstruction
        self.head = nn.Linear(self.head_nf, c_out * self.n_vars)
        
        # For classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(d_model, c_out)
        
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"📈 Model Statistics:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Attention Type: {'DMSA' if self.mask_diag else 'Standard'}")

    def forecast(self, x_enc):
        """Forward pass for forecasting/reconstruction tasks"""
        # Input: [batch, seq_len, features]
        x_enc = x_enc.permute(0, 2, 1)  # [batch, features, seq_len] for conv
        
        # Embedding
        enc_out = self.embedding_block(x_enc)  # [batch, seq_len, d_model]
        
        # TST Encoder with optional DMSA
        enc_out = self.encoder(enc_out)  # [batch, seq_len, d_model]
        
        # Reconstruction head
        output = self.head(enc_out)  # [batch, seq_len, c_out]
        output = output[:, :, 0]  # Select single target [batch, seq_len]
        
        return output

    def classification(self, x_enc):
        """Forward pass for classification tasks"""
        # Input: [batch, channels, seq_len] for Conv1D layers
        # No permutation needed - data is already in correct format
        
        # Embedding
        enc_out = self.embedding_block(x_enc)  # [batch, seq_len, d_model]
        
        # TST Encoder with optional DMSA
        enc_out = self.encoder(enc_out)  # [batch, seq_len, d_model]
        
        # Global Average Pooling + Classification
        enc_out = enc_out.permute(0, 2, 1)  # [batch, d_model, seq_len]
        output = self.gap(enc_out)  # [batch, d_model, 1]
        output = self.flatten(output)  # [batch, d_model]
        output = self.dropout(output)
        output = self.fc(output)  # [batch, c_out]
        
        return output

    def forward(self, x_enc, task="classification"):
        """Forward pass - supports both pretraining and classification"""
        if task == "forecast" or task == "reconstruction":
            return self.forecast(x_enc)
        else:
            return self.classification(x_enc)