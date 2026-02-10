import torch
import torch.nn.functional as F
from torchvision import utils,transforms
from PIL import Image
from torch.utils.data import Dataset
import os

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

@torch.no_grad()
def validate_style_consistency(model, device, image_paths):
    model.eval()
    embeddings = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])

    ])

    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        emb = model(img)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb)

    embeddings = torch.cat(embeddings)
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    return similarity_matrix


class StyleDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.5, 0.05),
                transforms.GaussianBlur(3),

                transforms.ToTensor(),

                
                transforms.RandomErasing(
                    p=0.3,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value='random'
                ),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        self.image_paths = []
        for p_id in os.listdir(root_dir):
            p_path = os.path.join(root_dir, p_id)
            if os.path.isdir(p_path):
                for img in os.listdir(p_path):
                    if "makeup" in img:
                        self.image_paths.append(
                            os.path.join(p_path, img)
                        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        view1 = self.transform(img)
        view2 = self.transform(img)

        return view1, view2



def vae_transform_hf(examples):
    transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
    return {
            "pixel_values": [transform(img.convert("RGB")) for img in examples["image"]]
        }
