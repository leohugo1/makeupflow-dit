import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms,utils

class SwiGLUMP(nn.Module):
    def __init__(self, in_features,hidden_features=None,out_features=None,drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features,hidden_features * 2)
        self.fc2 = nn.Linear(hidden_features,out_features)

        self.drop = nn.Dropout(drop)
    
    def forward(self,x):
        x = self.fc1(x)
        x,gate = x.chunk(2,dim=-1)
        x = x * F.silu(gate)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim,context_dim,num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim //num_heads) ** -0.5
        self.to_q = nn.Linear(dim,dim,bias=False)
        self.to_kv = nn.Linear(context_dim,dim * 2,bias=False)
        self.to_out = nn.Linear(dim,dim)
    def forward(self,x, context):
        B,N,C =x.shape
        q = self.to_q(x)

        kv = self.to_kv(context).chunk(2,dim=-1)
        k,v = map(lambda t: t.reshape(B,-1,self.num_heads,C//self.num_heads).transpose(1,2),kv)
        q = q.reshape(B,N,self.num_heads,C//self.num_heads).transpose(1,2)

        out = F.scaled_dot_product_attention(q,k,v)
        out = out.transpose(1,2).reshape(B,C,N)

        return self.to_out(out)

class Block(nn.Module):
    def __init__(self,dim, num_heads,context_dim,mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim,elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim,num_heads,batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = CrossAttention(dim,context_dim,num_heads)

        self.norm3 = nn.LayerNorm(dim,elementwise_affine=False)

        self.mlp = SwiGLUMP(dim,int(dim * mlp_ratio),drop=0.1)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim,bias=True)
        )
    
    def forward(self,x,c,t_emb):
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6,dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(self.modulate(self.norm1(x),shift_msa,scale_msa))[0]

        x = x + self.cross_attn(self.norm2(x),c)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm3(x),shift_mlp,scale_mlp))

        return x
    
    def modulate(self,x,shift,scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    

class DiT(nn.Module):
    def __init__(self, input_size = 64, patch_size=2,in_channels=4,hidden_size =768,depth=12,num_heads=12,mlp_ratio=4,style_dim=512):
        super().__init__()
        self.in_channels =in_channels
        self.patch_size = patch_size
        self.num_patchs = (input_size // patch_size) ** 2
        
        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels,hidden_size)

        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )

        self.style_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim,hidden_size)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1,self.num_patchs,hidden_size))

        self.blocks =nn.ModuleList([
            Block(hidden_size,num_heads,hidden_size,mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size,elementwise_affine=False),
            nn.Linear(hidden_size,patch_size * patch_size * in_channels)
        )

        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed,std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight,0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,0)
    
    def timestep_embedding(self,timesteps,dim,max_period=10000):
        half = dim // 2 
        freqs = torch.exp(-math.log(max_period) * torch.arange( start=0,end=half,dtype=torch.float32)/ half).to(timesteps.device)
        args = timesteps[:,None].float() * freqs[None,:]
        embedding = torch.cat([torch.sin(args),torch.cos(args)],dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding,torch.zeros_like(embedding[:,:1])],dim=-1)
        return embedding
    
    def unpatchify(self,x):
        p = self.patch_size
        h = w = int(x.shape[1] **0.5)
        x = x.reshape(shape=(x.shape[0],h,w,p,p,self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq',x)
        imgs = x.reshape(shape=(x.shape[0],self.in_channels,h*p,h*p))
        return imgs

    def forward(self,x,t,style_vector):
        B,C,H,W = x.shape
        x = x.reshape(B,C,H// self.patch_size,self.patch_size,W//self.patch_size,self.patch_size)
        x = torch.einsum('nchpwq->nhwpqc',x).reshape(B,-1,self.patch_size**2 * C)
        x = self.x_embedder(x) + self.pos_embed

        t_emb = self.t_embedder(self.timestep_embedding(t,x.shape[-1]))

        style_ctx = self.style_proj(style_vector).unsqueeze(1)

        for block in self.blocks:
            x = block(x,style_ctx,t_emb)
        
        x = self.final_layer(x)
        return self.unpatchify(x)