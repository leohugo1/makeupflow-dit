import torch
import torch.nn.functional as F
from torchvision import utils,transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import json
from pathlib import Path
import random

@torch.no_grad()
def generate_makeup(dit_model, style_vit, vae, img_bare, img_reference, epoch, steps=25):
    device = next(dit_model.parameters()).device

    style_emb = style_vit.forward_style(img_reference.to(device))

    x_bare = vae.encode(img_bare.to(device)) * 0.18215
    xt = x_bare.clone()

    dt = 1.0 / steps

    for i in range(steps):
        t_curr  = i / steps
        t_tensor = torch.full((xt.shape[0],), t_curr, device=device)
        v_pred  = dit_model(xt, t_tensor, style_emb)
        xt      = xt + v_pred * dt

    out_img = vae.decode(xt / 0.18215)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    ref_vis = (img_reference.to(device) * std + mean).clamp(0, 1)

    ref_vis = F.interpolate(ref_vis, size=(512, 512), mode='bilinear', align_corners=False)

    bare_vis = (img_bare.to(device) * 0.5 + 0.5).clamp(0, 1)

    out_vis = (out_img * 0.5 + 0.5).clamp(0, 1)
    grid = torch.cat([bare_vis, ref_vis, out_vis], dim=3)  
    utils.save_image(grid, f"results/preview_epoch_{epoch}.png")

@torch.no_grad()
def validate_style_consistency(model, device, image_paths):
    model.eval()
    embeddings = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

    ])

    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        emb = model.forward_style(img)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb)

    embeddings = torch.cat(embeddings)
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    return similarity_matrix

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class StyleDataset(Dataset):
    def __init__(self, root_dir: str, style_labels_path: str, is_train: bool = True):
        self.root_dir = root_dir
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
                transforms.GaussianBlur(3),
                transforms.ToTensor(),
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

        print(f"Carregando anotações de estilo: {style_labels_path}")
        with open(style_labels_path, 'r') as f:
            data = json.load(f)

        root_abs = Path(root_dir).resolve()

        def resolve_path(p: str) -> str:
          
            parts = Path(p.replace('\\', '/')).parts
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 6:
                    person_id = part
                    filename  = parts[i + 1] if i + 1 < len(parts) else Path(p).name
                    return str(root_abs / person_id / filename)
            p_obj = Path(p.replace('\\', '/'))
            return str(root_abs / p_obj.parent.name / p_obj.name)

        self.path_to_style: dict[str, int] = {
            resolve_path(k): v
            for k, v in data['labels'].items()
        }

        self.style_groups: dict[int, list[str]] = {
            int(k): [resolve_path(p) for p in v]
            for k, v in data['style_groups'].items()
        }

        self.path_to_person: dict[str, str] = {}
        self.person_to_label: dict[str, int] = {}
        self.samples: list[str] = []
        person_set = set()

        for path in self.path_to_style:
            person_id = Path(path).parent.name
            self.path_to_person[path] = person_id
            person_set.add(person_id)
            self.samples.append(path)

        for i, person_id in enumerate(sorted(person_set)):
            self.person_to_label[person_id] = i

        self.num_identities = len(person_set)

        print(f"Dataset carregado:")
        print(f"  Imagens:     {len(self.samples)}")
        print(f"  Identidades: {self.num_identities}")
        print(f"  Clusters:    {len(self.style_groups)}")
        print(f"  Média por cluster: {len(self.samples) / len(self.style_groups):.0f} imgs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        anchor_path    = self.samples[idx]
        anchor_style   = self.path_to_style[anchor_path]
        anchor_person  = self.path_to_person[anchor_path]
        identity_label = self.person_to_label[anchor_person]

        candidates = [
            p for p in self.style_groups[anchor_style]
            if self.path_to_person[p] != anchor_person
        ]

        if candidates:
            positive_path = random.choice(candidates)
        else:
            positive_path = anchor_path

        anchor_img   = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')

        view1 = self.transform(anchor_img)
        view2 = self.transform(positive_img)

        return view1, view2, identity_label



def vae_transform_hf(examples):
    transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
    return {
            "pixel_values": [transform(img.convert("RGB")) for img in examples["image"]]
        }


class DitDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        style_labels_path: str = None,
        cross_identity_style: bool = True,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.cross_identity_style = cross_identity_style

        self.transform_512 = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform_256 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        
        self.samples = []  
        self.makeup_paths = [] 

        for person_dir in sorted(self.root_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            bare_path = person_dir / "bare.jpg"
            if not bare_path.exists():
                continue

            makeup_files = sorted(person_dir.glob("makeup_*.jpg"))
            if not makeup_files:
                continue

            for makeup_path in makeup_files:
                self.samples.append((str(bare_path), str(makeup_path)))
                self.makeup_paths.append(str(makeup_path))

        print(f"DitDataset carregado:")
        print(f"  Pares (bare, makeup): {len(self.samples)}")

        self.style_groups = None
        self.path_to_style = None
        self.path_to_person = None

        if style_labels_path and cross_identity_style:
            print(f"  Carregando clusters CLIP: {style_labels_path}")
            with open(style_labels_path, 'r') as f:
                data = json.load(f)

            def resolve(p):
                parts = Path(p.replace('\\', '/')).parts
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 6:
                        return str(self.root_dir / part / parts[i + 1])
                p_obj = Path(p.replace('\\', '/'))
                return str(self.root_dir / p_obj.parent.name / p_obj.name)

            self.path_to_style = {
                resolve(k): v for k, v in data['labels'].items()
            }
            self.style_groups = {
                int(k): [resolve(p) for p in v]
                for k, v in data['style_groups'].items()
            }
            self.path_to_person = {
                p: Path(p).parent.name
                for p in self.path_to_style
            }
            print(f"  Clusters disponíveis: {len(self.style_groups)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_style_ref(self, makeup_path: str) -> str:
        if (
            self.cross_identity_style
            and self.style_groups is not None
            and makeup_path in self.path_to_style
        ):
            style_id = self.path_to_style[makeup_path]
            person_id = self.path_to_person[makeup_path]

            candidates = [
                p for p in self.style_groups[style_id]
                if self.path_to_person.get(p) != person_id
            ]

            if candidates:
                return random.choice(candidates)

        return makeup_path

    def __getitem__(self, idx: int) -> dict:
        bare_path, makeup_path = self.samples[idx]
        style_ref_path = self._get_style_ref(makeup_path)

        bare_img      = Image.open(bare_path).convert('RGB')
        makeup_img    = Image.open(makeup_path).convert('RGB')
        style_ref_img = Image.open(style_ref_path).convert('RGB')

        return {
            "bare_512":   self.transform_512(bare_img),
            "makeup_512": self.transform_512(makeup_img),
            "style_224":  self.transform_256(style_ref_img),
        }