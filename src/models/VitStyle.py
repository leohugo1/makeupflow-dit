import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, embed_dim=384, depth=12, num_heads=6, style_dim=512, identity_dim=512):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )
        self.peg = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.style_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

       
        self.style_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, style_dim)
        )

 
        self.identity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, identity_dim),
            nn.BatchNorm1d(identity_dim)
        )

        nn.init.trunc_normal_(self.style_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem(x)
        x = x + F.gelu(self.peg(x))
        x = x.flatten(2).transpose(1, 2)
        style_token = self.style_token.expand(B, -1, -1)
        x = torch.cat((style_token, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  

    def forward(self, x):
        features = self.forward_features(x)
        z_style = self.style_head(features)
        z_identity = self.identity_head(features)
        return z_style, z_identity

    def forward_style(self, x):
        features = self.forward_features(x)
        return self.style_head(features)
    

def info_nce(z1, z2, temperature=0.07):
    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)

    sim_matrix = torch.matmul(representations, representations.T) / temperature
    sim_matrix = torch.clamp(sim_matrix, -100, 100)

    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    pos_indices = torch.arange(2 * batch_size, device=z1.device)
    pos_indices = (pos_indices + batch_size) % (2 * batch_size)

    positives = sim_matrix[torch.arange(2 * batch_size), pos_indices].unsqueeze(1)
    negatives = sim_matrix[~mask].view(2 * batch_size, -1)
    logits = torch.cat([positives, negatives], dim=1)

    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)

    return F.cross_entropy(logits, labels)


def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1, z2, lambda_bt=0.005):
    B = z1.size(0)
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

    c = torch.matmul(z1_norm.T, z2_norm) / B 

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(c).pow_(2).sum()

    return on_diag + lambda_bt * off_diag


def orthogonal_loss(z_style, z_identity):

    z_identity = z_identity.detach()
    dot = (z_style * z_identity).sum(dim=1)  
    return dot.pow(2).mean()



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s 
        self.m = m 
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m 
        
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        

def style_loss(z_style1, z_style2, z_identity1, z_identity2, temperature=0.15, lambda_bt=0.005, w_orth=0.5):
    z_s1 = F.normalize(z_style1, dim=1)
    z_s2 = F.normalize(z_style2, dim=1)
    z_i1 = F.normalize(z_identity1, dim=1)
    z_i2 = F.normalize(z_identity2, dim=1)

    l_nce = info_nce(z_s1, z_s2, temperature)
    l_bt  = barlow_twins_loss(z_style1, z_style2, lambda_bt)
    l_orth = (orthogonal_loss(z_s1, z_i1) + orthogonal_loss(z_s2, z_i2)) / 2

    return l_nce, l_bt, l_orth, l_nce + l_bt + w_orth * l_orth