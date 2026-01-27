import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchmetrics.image import StructuralSimilarityIndexMeasure
import mlflow
import lpips
from datasets import load_dataset
import os
from tqdm import tqdm



class ResnetBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(32,in_ch)
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.norm2 = nn.GroupNorm(32,out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,padding=1)

        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self,x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))

        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.norm = nn.GroupNorm(32,in_ch)
        self.q = nn.Conv2d(in_ch,in_ch,1)
        self.k = nn.Conv2d(in_ch,in_ch,1)
        self.v = nn.Conv2d(in_ch,in_ch,1)

        self.proj_out = nn.Conv2d(in_ch,in_ch,1)
    def forward(self,x):
        h_ = self.norm(x)
        q,k,v = self.q(h_), self.k(h_), self.v(h_)
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k) * (c**-0.5)
        w_ = F.softmax(w_,dim=-1)
        v = v.reshape(b,c,h*w)
        h_ = torch.bmm(v,w_.permute(0,2,1))
        h_ = h_.reshape(b,c,h,w)
        return x + self.proj_out(h_)


class Encoder(nn.Module):
    def __init__(self, latent_ch = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(3,128,3,padding=1)
        self.down = nn.ModuleList([
            nn.Sequential(ResnetBlock(128,128),nn.Conv2d(128,128,3,stride=2,padding=1)),
            nn.Sequential(ResnetBlock(128,256),nn.Conv2d(256,256,3,stride=2,padding=1)),
            nn.Sequential(ResnetBlock(256,512),nn.Conv2d(512,512,3,stride=2,padding=1))
        ])
        self.mid = nn.Sequential(ResnetBlock(512,512),AttnBlock(512),ResnetBlock(512,512))
        self.norm = nn.GroupNorm(32,512)
        self.conv_out = nn.Conv2d(512,2 * latent_ch,3,padding=1)
    def forward(self,x):
        h = self.conv_in(x)

        for block in self.down: h = block(h)
        h = self.mid(h)
        return self.conv_out(F.silu(self.norm(h)))

class Decoder(nn.Module):
    def __init__(self,latent_ch=4):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_ch,512,3,padding=1)
        self.mid = nn.Sequential(ResnetBlock(512,512),AttnBlock(512),ResnetBlock(512,512))
        self.up = nn.ModuleList([
            nn.Sequential(ResnetBlock(512,512),nn.Upsample(scale_factor=2)),
            nn.Sequential(ResnetBlock(512,256),nn.Upsample(scale_factor=2)),
            nn.Sequential(ResnetBlock(256,128),nn.Upsample(scale_factor=2))
        ])
        self.norm = nn.GroupNorm(32,128)
        self.conv_out = nn.Conv2d(128,3,3,padding=1)
    def forward(self,z):
        h = self.conv_in(z)
        h = self.mid(h)
        for block in self.up: h = block(h)
        return self.conv_out(F.silu(self.norm(h)))
    
class VAE(nn.Module):
    def __init__(self,latent_ch=4):
        super().__init__()
        self.encoder = Encoder(latent_ch)
        self.decoder = Decoder(latent_ch)
    
    def reparameterize(self,h):
        mu, log_var =torch.chunk(h,2,dim=1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        return z,mu,log_var

    def forward(self,x):
        h = self.encoder(x)

        z, mu, log_var = self.reparameterize(h)
        recon_x = self.decoder(z)
        return recon_x,mu,log_var
    def encode(self,x):
        h = self.encoder(x)
        mu,_ =torch.chunk(h,2,dim=1)
        return mu
    def decode(self,z):
        return self.decoder(z)

def vae_loss(x,x_recon,mu,log_var,loss_fn_vgg):
    recon_loss = F.l1_loss(x,x_recon)
    perceptual_loss = loss_fn_vgg(x,x_recon).mean()
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total= recon_loss + 1.0 * perceptual_loss + 0.00001 * kl_loss
    return total,recon_loss,perceptual_loss,kl_loss


