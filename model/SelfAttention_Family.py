import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
import os

 
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)

        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        #print('output shape',V.shape)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class CrossModalResidualAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalResidualAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: [T, B, D] ; key, value: [T_key, B, D]
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(query + attn_output)
        return output, attn_weights
    
class PatchFusionWithSeasonTrend(nn.Module):
    def __init__(self, embed_dim, d_model, num_heads, dropout=0.1, fusion_method='sum'):
        super(PatchFusionWithSeasonTrend, self).__init__()
        self.attn_patch_season = CrossModalResidualAttention(d_model, num_heads, dropout)
        self.attn_patch_trend = CrossModalResidualAttention(d_model, num_heads, dropout)
       
        self.fusion_method = fusion_method
        self.adjust_dim = nn.Linear(embed_dim, d_model)
        if fusion_method == 'concat':
            self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.residual_model = nn.Linear(d_model, d_model)

    def forward(self, F_patch, F_season, F_trend):
        F_patch_season, attn_season = self.attn_patch_season(F_patch, F_season, F_season)
        F_patch_trend, attn_trend = self.attn_patch_trend(F_patch, F_trend, F_trend)
       
        residual_season = F_patch - F_patch_season  
        residual_trend = F_patch - F_patch_trend 

        residual_season_corrected = self.residual_model(residual_season)
        residual_trend_corrected = self.residual_model(residual_trend)

        if self.fusion_method == 'sum':
            F_patch_fused = F_patch_season + F_patch_trend + residual_season_corrected + residual_trend_corrected
        elif self.fusion_method == 'concat':
            F_patch_fused = torch.cat([F_patch_season, F_patch_trend], dim=-1)
            F_patch_fused = self.fusion_proj(F_patch_fused) + residual_season_corrected + residual_trend_corrected
        else:
            raise ValueError("Unsupported fusion_method. Use 'sum' or 'concat'.")

        return F_patch_fused, (attn_season, attn_trend)    