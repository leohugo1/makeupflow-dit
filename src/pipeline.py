import torch
import os
from PIL import Image
from torchvision import transforms, utils
from src.models.VAE import VAE
from src.models.VitStyle import StyleVit
from src.models.Dit import DiT

def load_pipeline(checkpoint_dit, checkpoint_vae, checkpoint_style, device):
    vae = VAE().to(device)
    style_vit = StyleVit().to(device)
    dit = DiT(input_size=64).to(device)

    vae.load_state_dict(torch.load(checkpoint_vae, map_location=device)['model_state_dict'])
    style_vit.load_state_dict(torch.load(checkpoint_style, map_location=device)['model_state_dict'])
    dit.load_state_dict(torch.load(checkpoint_dit, map_location=device)['model_state_dict'])

    vae.eval().requires_grad_(False)
    style_vit.eval().requires_grad_(False)
    dit.eval()

    return vae, style_vit, dit

@torch.no_grad()
def run_makeup_transfer(vae, style_vit, dit, bare_path, ref_path, device, steps=30):

    t512 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    t256 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_bare = t512(Image.open(bare_path).convert("RGB")).unsqueeze(0).to(device)
    img_ref = t256(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)

    style_emb = style_vit(img_ref)
    x_bare = vae.encode(img_bare) * 0.18215
    
    xt = x_bare.clone()
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((1,), i / steps, device=device)
        v_pred = dit(xt, t, style_emb)
        xt = xt + v_pred * dt

    
    res = vae.decode(xt / 0.18215)
    return res

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_vae = f"checkpoints/vae_epoch_xxx_step_xxx.pth"
    ckpt_style = f'style_vit_epoch_xx.pth'
    ckpt_dit = f"checkpoints/Dit_epoch_xxx.pth"

    vae, style, dit = load_pipeline(ckpt_dit, ckpt_vae, ckpt_style, device)

    output = run_makeup_transfer(vae, style, dit, "minha_foto.jpg", "referencia_estilo.jpg", device)
    utils.save_image(output, "resultado_transferencia.png", normalize=True, value_range=(-1, 1))