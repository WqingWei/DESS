import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.embed import *
from layers.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer, PatchFusionWithSeasonTrend
 
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

class Model(nn.Module):
    def __init__(self, configs, enc_in, c_out):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.enc_in = enc_in
        self.c_out = c_out
        self.is_first_dataset = True
        self.previous_prompt = None

        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_size, configs.stride, configs.padding, configs.dropout)
        self.frequency_embedding = FrequencyEmbedding(configs.d_model, configs.top_k, rfft=True)

        self.patch_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.num_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )  

        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_size) / configs.stride + 1)
        self.head = FlattenHead(enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
    
        self.projection = nn.Linear(configs.d_model, 1, bias=True)
        
        self.patch_fusion = PatchFusionWithSeasonTrend(embed_dim=enc_in, 
                                                       d_model=configs.d_model,
                                                       num_heads=configs.n_heads, 
                                                       dropout=configs.dropout, 
                                                       fusion_method='sum')
        
        self.vae = VAE(input_dim=enc_in, 
                       hidden_dim=configs.d_model, 
                       latent_dim=c_out, 
                       output_dim=configs.d_model)
        
        self.vae_prompt = nn.Parameter(torch.zeros(1, configs.seq_len, configs.d_model), requires_grad=True)

    def save_prompt(self, prompt):
        self.previous_prompt = prompt.detach().clone()
        
    def adjust_prompt_and_concat(self, fused_patch, prompt):
        current_prompt_dim = prompt.size(2)
        target_dim = fused_patch.size(2)  
        batch_size = fused_patch.size(0)
        seq_len = fused_patch.size(1)
        if prompt.size(0) != batch_size:
                prompt = prompt[:batch_size, :, :]
        if prompt.size(1) != seq_len:
            prompt = prompt[:, :seq_len, :]
        
        if current_prompt_dim < target_dim:
            padding_size = target_dim - current_prompt_dim
            prompt = torch.nn.functional.pad(prompt, (0, padding_size))
        elif current_prompt_dim > target_dim:
            prompt = prompt[:, :, :target_dim]       
        combined_out = fused_patch + prompt
    
        return combined_out

    def generate_vae_prompt(self, x_enc):
        batch_size = x_enc.size(0)
        seq_len = x_enc.size(1) 
        dim = x_enc.size(2)
        x_mean = torch.mean(x_enc, dim=1)
        mu, log_var = self.vae.encode(x_mean)
        z = self.vae.reparameterize(mu, log_var)#[batch_size, enc_in]

        if z.size(1) < self.d_model:
            z_expanded = torch.nn.functional.pad(z, (0, self.d_model - z.size(1)))
        else:
            z_expanded = z[:, :self.d_model]
      
        expanded_prompt = z_expanded.unsqueeze(1).expand(batch_size, self.seq_len, self.d_model)
        
        return expanded_prompt[:1].detach().clone()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        # Apply patch embedding
        x_enc_orig = x_enc.clone() 
        x = x_enc.permute(0, 2, 1).contiguous()   # [B, C, L]
        x = x.view(B * C, L) 

        x_patch = x.unsqueeze(1) # [B*C, 1, L]
        patch, _ = self.patch_embedding(x_patch)

        # Apply frequency embedding for season and trend 
        season, trend = self.frequency_embedding(x_patch)   # [B*C, L, D]
 
        patch, _ = self.patch_encoder(patch)  # [B*C, L, D]
        season, _ = self.patch_encoder(season)  # [B*C, L, D]
        trend, _ = self.patch_encoder(trend)  # [B*C, L, D]
        
        fused_patch, (attn_season, attn_trend) = self.patch_fusion(patch, season, trend)
       
        if self.training: 
            current_prompt = self.vae_prompt
        else:
            if self.previous_prompt is not None:
                current_prompt = self.previous_prompt
            else:
                current_prompt = self.vae_prompt
       
        current_prompt = current_prompt.expand(B * C, -1, -1)
        combined_out = self.adjust_prompt_and_concat(fused_patch, current_prompt)
       
        out = self.projection(combined_out).squeeze(-1)
        out = out.view(B, C, L).permute(0, 2, 1).contiguous()  # [B, L, C]
       
        return out  

        
