import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad_(False)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_size, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        batch_size, seq_len = x.shape[:2]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))#(batch_size*n_vars,num_patches,patch_size)

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), n_vars



class Frequency(nn.Module):
    def __init__(self, top_k: int, rfft: bool = True):
        super(Frequency, self).__init__()
        self.top_k = top_k
        self.rfft = rfft
    
    def forward(self, x):
        # Fourier transform
        if self.rfft:
            xf = torch.fft.rfft(x, dim=-1)
        else:
            xf = torch.fft.fft(x, dim=-1)

        xf_abs = xf.abs()
        
        top_k = min(self.top_k, xf.size(-1))

        _, indices = torch.topk(xf_abs, top_k, dim=-1)

        xf_topk = torch.gather(xf, dim=-1, index=indices)
        xf_filtered = torch.zeros_like(xf)
        xf_filtered.scatter_(-1, indices, xf_topk)
        
        if self.rfft:
            trend_x = torch.fft.irfft(xf_filtered, n=x.size(-1), dim=-1).float()
        else:
            trend_x = torch.fft.ifft(xf_filtered, dim=-1).real.float()

        season_x = x - trend_x  
        
        return season_x, trend_x 

class FrequencyEmbedding(nn.Module):
    def __init__(self, d_model: int, top_k: int, rfft: bool = True, value_embedding_bias: bool = False):
        super(FrequencyEmbedding, self).__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.value_embedding_bias = value_embedding_bias
        self.freq = Frequency(top_k=top_k, rfft=rfft)
        self.value_embedding = nn.Linear(1 , d_model, bias=value_embedding_bias)

    def forward(self, x):
        batch_size, n_vars, seq_len = x.shape
        # Fourier decomposition
        season_x, trend_x = self.freq(x)  #(batch_size, n_vars, seq_len)
        # Flatten for linear layer
        season_x = season_x.permute(0, 2, 1) #(batch_size, seq_len, n_vars)
        trend_x = trend_x.permute(0, 2, 1)
        
        season_x = self.value_embedding(season_x)
        trend_x = self.value_embedding(trend_x)
       
        return season_x, trend_x
