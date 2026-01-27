import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class SwiGLUMP(nn.Module):
    def __init__(self, in_features, hidden_features = None,out_features = None,drop=0. ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features,hidden_features * 2)
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
       x = self.fc1(x)
       x, gate = x.chunk(2,dim=-1)
       x = x * F.silu(gate)
       x = self.drop(x)
       x = self.fc2(x)
       x = self.drop(x)
       return x

class Attention(nn.Module):
    def __init__(self,dim, num_heads=8,qkv_bias=True,attn_drop=0.,proj_drop=0.):
        super().__init__() 
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3,bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v =qkv[0],qkv[1],qkv[2]

        x = F.scaled_dot_product_attention(
            q,k,v,
            dropout_p=self.attn_drop if self.training else 0.,
            is_causal=False
        )
        x = x.transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4,drop=0.,attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim,num_heads,attn_drop,drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUMP(dim,int(dim * mlp_ratio),drop)
    
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class StyleVit(nn.Module):
    def __init__(self,embed_dim=384,depth=12,num_heads=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128,embed_dim,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim,embed_dim,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(embed_dim)
        )

        self.peg = nn.Conv2d(embed_dim,embed_dim,kernel_size=3,padding=1,groups=embed_dim)
        self.style_token = nn.Parameter(torch.zeros(1,1,embed_dim))


        self.blocks = nn.ModuleList([
            Block(embed_dim,num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.style_head = nn.Linear(embed_dim,512)

        nn.init.trunc_normal_(self.style_token,std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight,std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        B = x.shape[0]

        x = self.stem(x)
        H,W = x.shape[2],x.shape[3]

        x = x + self.peg(x)
        x = x.flatten(2).transpose(1,2)

        style_token = self.style_token.expand(B,-1,-1)
        x = torch.cat((style_token,x),dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.style_head(x[:,0])
