import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils
from torchmetrics.image import StructuralSimilarityIndexMeasure
import mlflow
import lpips
from datasets import load_dataset
import os
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm
from src.models.VAE import VAE,vae_loss
from src.models.VitStyle import StyleVit,style_loss,ArcMarginProduct
from src.models.Dit import DiT
from src.utils import generate_makeup,vae_transform_hf,StyleDataset,validate_style_consistency,accuracy,DitDataset

def train_vae(model,dataloader,optmizer,lpips_loss_fn,scaler,device,config,num_epochs=50):
    model.train()

    print(f"Iniciando treino no dispositivo: {device}")
    avg_loss = 0
    current_epoch = config.get('epoch', 1)
    global_step = config.get('step', 0)

    batches_per_epoch = len(dataloader)
    batches_to_skip = global_step % batches_per_epoch
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    mlflow.set_experiment(experiment_name="MakeupFlow-VAE")

    with mlflow.start_run(run_name="8f4f949cfd984deb98ed13ada003b81c"):

        mlflow.log_params({
            "lr":1e-6,
            "batch_size":dataloader.batch_size,
            "latent_ch":4,
            "resolution": "512x512",
            "device": device,
            "p_weight":2.5,
            "kl_weight":0.000005
        })

        for epoch in range(current_epoch,num_epochs + 1):
            if epoch == current_epoch and batches_to_skip > 0:
                print(f"Resumindo Época {epoch}: Pulando {batches_to_skip} batches...")
                indices = list(range(batches_to_skip * dataloader.batch_size, len(dataloader.dataset)))
                subset_dataset = torch.utils.data.Subset(dataloader.dataset, indices)
                current_dataloader = torch.utils.data.DataLoader(
                    subset_dataset, 
                    batch_size=dataloader.batch_size, 
                    shuffle=False
                )
            else:
                current_dataloader = dataloader
                batches_to_skip = 0
                global_step = 0

            pbar = tqdm(enumerate(current_dataloader), 
                        total=len(current_dataloader), 
                        desc=f"Epoch {epoch}/{num_epochs}")
            
            total_epoch_loss = 0

            for _, batch in pbar:
                global_step += 1
                x = batch["pixel_values"].to(device)

                optmizer.zero_grad()
                with torch.amp.autocast(device_type=device,dtype=torch.bfloat16):
                    recon,mu,logvar = model(x)
                    total_loss,r_loss,p_loss,k_loss = vae_loss(x,recon,mu,logvar,lpips_loss_fn)

                scaler.scale(total_loss).backward()
                scaler.unscale_(optmizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optmizer)
                scaler.update()

                total_epoch_loss += total_loss.item()
                if global_step % 100 == 0:
                    with torch.no_grad():
                        ssim_val = ssim_metric(recon, x)
                    mlflow.log_metric("loss_total", total_loss.item(), step=global_step)
                    mlflow.log_metric("loss_recon_l1", r_loss.item(), step=global_step)
                    mlflow.log_metric("loss_perceptual", p_loss.item(), step=global_step)
                    mlflow.log_metric("loss_kl", k_loss.item(), step=global_step)
                    mlflow.log_metric("ssim", ssim_val.item(), step=global_step)

                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{total_loss.item():.4f}"
                })

                if global_step % 1000 == 0:
                    
                    with torch.no_grad():
                        comparison = torch.cat([x[:1], recon[:1]], dim=0)
                        utils.save_image(
                            comparison, 
                            f"results/epoch_{epoch}_step_{global_step}.png", 
                            normalize=True, 
                            value_range=(-1, 1),
                            nrow=2
                        )
                        mlflow.log_artifact(f"results/epoch_{epoch}_step_{global_step}.png")
                        checkpoint_path = f"checkpoints/vae_epoch_{epoch}_step_{global_step}.pth"
                        torch.save({
                            'epoch': epoch,
                            'step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optmizer.state_dict(),
                            'loss': avg_loss,
                        }, checkpoint_path)
            mlflow.log_metric("avg_epoch_loss", total_epoch_loss / len(dataloader), step=epoch)
            avg_loss = total_epoch_loss / len(dataloader)   


def Train_vit_style(model,arcface,dataloader,optimizer,scaler,device,config,scheduler,num_epochs=300):
    print(f"Iniciando treino no dispositivo: {device}")
    accumulation_steps = 4
    temperature   = 0.15
    lambda_bt     = 0.001 
    w_arcface     = 0.5
    current_epoch = config.get('epoch', 1)

    mlflow.set_experiment(experiment_name="style_vit")
    with mlflow.start_run(run_id="66074030a5424a93bb12f5e5f059bbbf"):
        mlflow.log_params({
            "model_type":    "StyleVit_DualHead",
            "embed_dim":     384,
            "style_dim":     512,
            "identity_dim":  512,
            "batch_size":    dataloader.batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch": dataloader.batch_size * accumulation_steps,
            "temperature":   temperature,
            "lambda_bt":     lambda_bt,
            "w_arcface":     w_arcface,
        })
        for epoch in range(current_epoch,num_epochs):
            model.train()
            arcface.train()
            optimizer.zero_grad()
            total_style_loss = 0
            id_acc = 0.0 
            pbar = tqdm(enumerate(dataloader),desc=f"Epoch {epoch}/{num_epochs}")
            for it, (view1,view2,identity_labels) in pbar:
                step = (epoch - 1 ) * len(dataloader) + it
                
                view1 = view1.to(device)
                view2 = view2.to(device)
                identity_labels = identity_labels.to(device)

                

                with torch.amp.autocast(device_type=device,dtype=torch.bfloat16):
                    z_style1, z_id1 = model(view1)
                    z_style2, z_id2 = model(view2)

                    w_orth_dynamic = max(0.005, min(0.02, id_acc * 0.1))
                    l_nce, l_bt, l_orth, loss_style = style_loss(
                        z_style1, z_style2,
                        z_id1,    z_id2,
                        temperature=temperature,
                        lambda_bt=lambda_bt,
                        w_orth=w_orth_dynamic
                    )

                    z_id_all  = torch.cat([F.normalize(z_id1, dim=1),
                                           F.normalize(z_id2, dim=1)], dim=0).float()
                    id_labels = torch.cat([identity_labels, identity_labels], dim=0)
                    arcface_logits = arcface(z_id_all, id_labels)
                    loss_arcface   = F.cross_entropy(arcface_logits, id_labels, label_smoothing=0.1)

                    loss = loss_style + w_arcface * loss_arcface
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                if (it + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) +
                        list(arcface.parameters()),
                        max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                with torch.no_grad():
                    zs1 = F.normalize(z_style1, dim=1)
                    zs2 = F.normalize(z_style2, dim=1)
                    d_pos    = torch.norm(zs1 - zs2, p=2, dim=1).mean().item()
                    id_acc   = (arcface_logits.argmax(dim=1) == id_labels).float().mean().item()

                    zi1 = F.normalize(z_id1, dim=1)
                    style_id_corr = (zs1 * zi1).sum(dim=1).abs().mean().item()

                total_style_loss += l_nce.item() 

                if step % 50 == 0:
                    mlflow.log_metric("loss_nce",       l_nce.item(),       step=step)
                    mlflow.log_metric("loss_barlow",    l_bt.item(),        step=step)
                    mlflow.log_metric("loss_orth",      l_orth.item(),      step=step)
                    mlflow.log_metric("loss_arcface",   loss_arcface.item(),step=step)
                    mlflow.log_metric("d_pos",          d_pos,              step=step)
                    mlflow.log_metric("id_acc",         id_acc,             step=step)
                    mlflow.log_metric("style_id_corr",  style_id_corr,      step=step)

                pbar.set_postfix({
                    "nce":   f"{l_nce.item():.4f}",
                    "bt":    f"{l_bt.item():.4f}",
                    "orth":  f"{l_orth.item():.4f}",
                    "arc":   f"{loss_arcface.item():.3f}",
                    "id%":   f"{id_acc:.1%}",
                    "corr":  f"{style_id_corr:.3f}",
                    "d_pos": f"{d_pos:.3f}",
                })

            scheduler.step()
            optimizer.param_groups[1]['lr'] = 3e-4
            optimizer.param_groups[0]['lr'] = max(optimizer.param_groups[0]['lr'], 1e-7)
            avg_loss = total_style_loss / len(dataloader)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"===> Epoch {epoch} Finalizada | style Loss: {avg_loss:.4f}")
            state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            if (epoch) % 2 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict':state_dict,
                    'arcface_state_dict':arcface.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict()
                },f'checkpoints/style_vit_epoch_{epoch}.pth')


def Train_dit(dit_model, style_model, vae, dataloader, optimizer,scaler,scheduler,device,config,num_epochs=50):

    print(f"Iniciando Treino via Flow Matching na {torch.cuda.get_device_name(0)}...")
    current_epoch = config.get('epoch', 1)
    mlflow.set_experiment(experiment_name="dit_makeup")
    with mlflow.start_run(run_name="dit_flow_matching"):
        mlflow.log_params({
            "model":          "DiT",
            "input_size":     512,
            "patch_size":     2,
            "hidden_size":    768,
            "depth":          12,
            "num_heads":      12,
            "style_dim":      512,
            "batch_size":     dataloader.batch_size,
            "num_epochs":     num_epochs,
            "flow_matching":  True,
        })
        for epoch in range(current_epoch,num_epochs):
            dit_model.train()
            total_loss = 0.0
            pbar = tqdm(enumerate(dataloader),desc=f"Epoch {epoch}/{num_epochs}")

            for it, batch in pbar:
                step = epoch * len(dataloader) + it

                img_bare = batch["bare_512"].to(device)     
                img_makeup = batch["makeup_512"].to(device) 
                img_makeup_style = batch["style_224"].to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    x_bare = vae.encode(img_bare) * 0.18215
                    x_makeup = vae.encode(img_makeup) * 0.18215
                    style_emb = style_model.forward_style(img_makeup_style)
                
                batch_size = x_bare.shape[0]

                t = torch.rand(batch_size,device=device)
                t_vis = t.view(batch_size,1,1,1)

                xt = (1.0 - t_vis) * x_bare + t_vis * x_makeup
                target = x_makeup - x_bare

                with torch.amp.autocast(device_type=device,dtype=torch.bfloat16):
                    pred = dit_model(xt,t,style_emb)
                    loss = F.mse_loss(pred,target)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dit_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                if step % 50 == 0:
                    mlflow.log_metric("loss_step", loss.item(), step=step)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"===> Epoch {epoch} Finalizada | Loss: {avg_loss:.4f}")

            if epoch % 2 == 0:
                amostra_bare = batch["bare_512"][0:1].to(device)
                amostra_referencia = batch["style_224"][0:1].to(device)
                out_path = f"results/preview_epoch_{epoch}.png"
                generate_makeup(dit_model,style_model,vae,amostra_bare,amostra_referencia,epoch)
                mlflow.log_artifact(out_path)
                state_dict = dit_model._orig_mod.state_dict() if hasattr(dit_model, '_orig_mod') else dit_model.state_dict()
                torch.save({
                    'epoch':              epoch,
                    'model_state_dict':   state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':  scaler.state_dict(),
                }, f"checkpoints/dit_epoch_{epoch}.pth")

def get_config():
    return {
        'epoch':46,
        'step':0,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available else "cpu"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str,
                        choices=['train_vae', 'train_style_vit','train_dit'])
    args = parser.parse_args()
    config = get_config()

    if args.phase == 'train_vae':
        hf_dataset = load_dataset("saitsharipov/CelebA-HQ", split="train")
        hf_dataset.set_transform(vae_transform_hf)

        dataloader = DataLoader(hf_dataset, batch_size=2, shuffle=True,num_workers=2,pin_memory=True,drop_last=True)

        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler(device=device)
        model_filename = f"checkpoints/vae_epoch_{config['epoch']}_step_{config['step']}.pth"
        if os.path.exists(model_filename):
            checkpoint = torch.load(model_filename,map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for g in optimizer.param_groups:
                g['lr'] = config.get('lr')
            config['epoch'] = checkpoint.get('epoch',1)
            config['step'] = checkpoint.get('step',0)
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        train_vae(model,dataloader,optimizer,loss_fn_vgg,scaler,device,config)
    elif args.phase == 'train_style_vit':
        epoch = config.get('epoch')
        model_filename = f'checkpoints/style_vit_epoch_{epoch}.pth'
        dataset = StyleDataset(
            root_dir="./FFHQ-Makeup/extracted",
            style_labels_path="./makeup_style_labels.json"
        )
        num_identities = dataset.num_identities
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
        
        model = StyleVit().to(device)
        arcface = ArcMarginProduct(
        in_features=512,
        out_features=num_identities,
        s=16.0,
        m=0.2
        ).to(device)
        scaler = torch.amp.GradScaler(device=device)
        optimizer = optim.AdamW([
            {'params': model.parameters(),    'lr': 5e-5,  'name': 'vit'},
            {'params': arcface.parameters(),  'lr':  3e-4,  'name': 'arcface'},
        ], weight_decay=0.1)
        num_warmup_epochs = 10
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=1e-7
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[num_warmup_epochs]
        )
        if os.path.exists(model_filename):
            config["epoch"] = config.get('epoch') + 1
            print(f"Carregando checkpoint: {model_filename}")
            checkpoint = torch.load(model_filename,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            arcface.load_state_dict(checkpoint['arcface_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            for params in optimizer.param_groups:
                if params['name'] == 'arcface':
                    params['lr'] = 3e-4
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"LR ViT:     {optimizer.param_groups[0]['lr']}")
        print(f"LR ArcFace: {optimizer.param_groups[1]['lr']}")
        image_files = glob(os.path.join("./vit_test", "*.jpg")) + \
              glob(os.path.join("./vit_test", "*.png")) + \
              glob(os.path.join("./vit_test", "*.jpeg"))

        if not image_files:
            print("Erro: Nenhuma imagem encontrada na pasta ./vit_test")
        else:
            result = validate_style_consistency(model, device, image_files)
            print(result)
        Train_vit_style(model,arcface,train_loader,optimizer,scaler,device,config,scheduler)
    elif args.phase == 'train_dit':
        vae = VAE().to(device)
        style_vit = StyleVit().to(device)
        vae.requires_grad_(False)
        style_vit.requires_grad_(False)
        vae.eval()
        style_vit.eval()
        dit = DiT().to(device)

        dataset = DitDataset(root_dir="./FFHQ-Makeup/extracted")
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(dit.parameters(),lr=1e-4,weight_decay=0.01) 
        scaler = torch.amp.GradScaler(device=device)
        warmup_epochs  = 5
        total_epochs   = 150

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-7
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        vae_path = f"checkpoints/vae_epoch_10_step_7000.pth"
        style_vit_path = f'style_vit_epoch_46.pth'
        model_filename = f"checkpoints/Dit_epoch_{config['epoch']}.pth"

        vae_checkpoint = torch.load(vae_path,map_location=device)
        vae.load_state_dict(vae_checkpoint['model_state_dict'])

        style_vit_checkpoint = torch.load(style_vit_path,map_location=device)
        style_vit.load_state_dict(style_vit_checkpoint['model_state_dict'])

        if os.path.exists(model_filename):
            checkpoint = torch.load(model_filename,map_location=device)
            dit.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        Train_dit(dit,style_vit,vae,train_loader,optimizer,scaler,scheduler,device,config,num_epochs=150)


