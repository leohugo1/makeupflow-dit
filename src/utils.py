import torch
from torchvision import utils,transforms


@torch.no_grad()
def generate_makeup(dit_model, style_vit, vae, img_bare, img_reference,epoch, steps=25):
    device = next(dit_model.parameters()).device
    
    style_emb = style_vit(img_reference.to(device))
    

    x_bare = vae.encode(img_bare.to(device)) * 0.18215

    xt = x_bare.clone()
    
    dt = 1.0 / steps 
    
    for i in range(steps):
        t_curr = (i / steps) 
        t_tensor = torch.full((1,), t_curr, device=device)
        
       
        v_pred = dit_model(xt, t_tensor, style_emb)
        
      
        xt = xt + v_pred * dt
    
    out_img = vae.decode(xt / 0.18215)
    utils.save_image(out_img, f"results/preview_epoch_{epoch}.png")


def style_transform_hf(examples):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    return {
        "anchor": [transform(img.convert("RGB")) for img in examples["makeup_image"]],
        "positive": [transform(img.convert("RGB")) for img in examples["makeup_image"]], 
        "negative": [transform(img.convert("RGB")) for img in examples["bare_image"]]
    }

def dit_transform_hf(examples):
    transform_512 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    transform_256 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        "bare_512": [transform_512(img.convert("RGB")) for img in examples["bare_image"]],
        "makeup_512": [transform_512(img.convert("RGB")) for img in examples["makeup_image"]],
        "makeup_256": [transform_256(img.convert("RGB")) for img in examples["makeup_image"]]
    }

def vae_transform_hf(examples):
    transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
    return {
            "pixel_values": [transform(img.convert("RGB")) for img in examples["image"]]
        }
