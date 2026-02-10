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
from glob import glob
import argparse
from tqdm import tqdm
from src.models.VAE import VAE,vae_loss
from src.models.VitStyle import StyleVit,info_nce
from src.models.Dit import DiT
from src.utils import generate_makeup,vae_transform_hf,StyleDataset,validate_style_consistency

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


def Train_vit_style(model,dataloader,optmizer,scaler,device,config,scheduler,num_epochs=300):
    print(f"Iniciando treino no dispositivo: {device}")
    accumulation_steps = 4
    current_epoch = config.get('epoch', 1)
    mlflow.set_experiment(experiment_name="style_vit")
    with mlflow.start_run(run_id="3590f3881c8a491ebb4a649f8771001f"):
        mlflow.log_params({
            "model_type": "StyleVit_From_Scratch",
            "embed_dim": 384,
            "batch_size": dataloader.batch_size,
            "accumulation_steps": 4,
            "effective_batch_size": dataloader.batch_size * 4,
            "learning_rate_max": 1e-4,
            "warmup_epochs": 10,
            "weight_decay": 0.1,
            "temperature": 0.1,
            "negatives_strategy": "cross_identity",
            "optimizer": "AdamW"
        })
        temperature = 0.1
        for epoch in range(current_epoch,num_epochs +1):
            model.train()
            optimizer.zero_grad()
            total_loss = 0
            pbar = tqdm(enumerate(dataloader),desc=f"Epoch {epoch}/{num_epochs}")
            for it, (view1,view2) in pbar:
                step = epoch * len(dataloader) + it
                view1 = view1.to(device)
                view2 = view2.to(device)


                with torch.amp.autocast(device_type=device,dtype=torch.bfloat16):
                    z1 = model(view1)
                    z2 = model(view2)

                    loss = info_nce(z1,z2,temperature)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (it + 1) % accumulation_steps == 0:
                    scaler.unscale_(optmizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                with torch.no_grad():
                    z1_n = F.normalize(z1, dim=1)
                    z2_n = F.normalize(z2, dim=1)
                    
                    d_pos = torch.norm(z1_n - z2_n, p=2, dim=1).mean().item()
                    avg_norm = torch.norm(z1, p=2, dim=1).mean().item()
                mlflow.log_metric("batch_loss", loss.item(), step=step)
                mlflow.log_metric("distance_positive", d_pos, step=step)
                mlflow.log_metric("embedding_norm", avg_norm, step=step)

                total_loss += loss.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "d_pos": f"{d_pos:.3f}",
                    "norm": f"{avg_norm:.2f}"
                })
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"===> Epoch {epoch} Finalizada | Average Loss: {avg_loss:.4f}")
            state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            if (epoch) % 2 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict':state_dict,
                    'optimizer_state_dict':optmizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict()
                },f'checkpoints/style_vit_epoch_{epoch}.pth')


def Train_dit(dit_model, style_model, vae, dataloader, optimizer,scaler,device,num_epochs=50):

    print(f"Iniciando Treino via Flow Matching na {torch.cuda.get_device_name(0)}...")

    for epoch in range(0,num_epochs):
        dit_model.train()

        pbar = tqdm(enumerate(dataloader),desc=f"Epoch {epoch}/{num_epochs}")

        for it, batch in pbar:
            img_bare = batch["bare_512"].to(device)     
            img_makeup = batch["makeup_512"].to(device) 
            img_makeup_style = batch["makeup_256"].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                x_bare = vae.encode(img_bare) * 0.18215
                x_makeup = vae.encode(img_makeup) * 0.18215
                style_emb = style_model(img_makeup_style)
            
            batch_size = x_bare.shape[0]

            t = torch.rand(batch_size,device=device)
            t_vis = t.view(batch_size,1,1,1)

            noise = torch.randn_like(x_bare)
            xt = (1.0 - t_vis) * x_bare + t_vis * noise
            target = x_makeup - x_bare

            with torch.amp.autocast(device_type=device,dtype=torch.bfloat16):
                pred = dit_model(xt,t,style_emb)
                loss = F.mse_loss(pred,target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if epoch % 5 == 0:
            amostra_bare = batch["bare_512"][0:1].to(device)
            amostra_referencia = batch["makeup_256"][0:1].to(device)
            generate_makeup(dit_model,style_model,vae,amostra_bare,amostra_referencia,epoch)

def get_config():
    return {
        'epoch':18,
        'step':0,
        'lr':1e-4
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
            root_dir="./FFHQ-Makeup/extracted"
        )
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

        model = StyleVit().to(device)
        scaler = torch.amp.GradScaler(device=device)
        optimizer = optim.AdamW(model.parameters(), lr=config.get('lr'), weight_decay=0.1)
        num_warmup_epochs = 20
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=num_warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300 - num_warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[num_warmup_epochs]
        )
        if os.path.exists(model_filename):
            config["epoch"] = config.get('epoch') + 1
            print(f"Carregando checkpoint: {model_filename}")
            checkpoint = torch.load(model_filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if optimizer.param_groups[0]['lr'] > 0.01: 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4

            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
 
        model = torch.compile(model,mode="reduce-overhead")
        Train_vit_style(model,train_loader,optimizer,scaler,device,config,scheduler)
    elif args.phase == 'train_dit':
        vae = VAE().to(device)
        style_vit = StyleVit().to(device)
        vae.requires_grad_(False)
        style_vit.requires_grad_(False)
        vae.eval()
        style_vit.eval()
        dit = DiT().to(device)

        dataset = load_dataset("cyberagent/FFHQ-Makeup", split="train")
        #dataset.set_transform(dit_transform_hf)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(dit.parameters(),lr=config.get('lr'),weight_decay=0.01) 
        scaler = torch.amp.GradScaler(device=device)

        vae_path = f"checkpoints/vae_epoch_xx_step_xx.pth"
        style_vit_path = f'style_vit_epoch_xx.pth'
        model_filename = f"checkpoints/Dit_epoch_{config['epoch']}.pth"

        vae_checkpoint = torch.load(vae_path,map_location=device)
        vae.load_state_dict(vae_checkpoint['model_state_dict'])

        style_vit_checkpoint = torch.load(style_vit_path,map_location=device)
        style_vit.load_state_dict(style_vit_checkpoint['model_state_dict'])

        if os.path.exists(model_filename):
            checkpoint = torch.load(model_filename,map_location=device)
            dit.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        dit = torch.compile(dit)
        style_vit = torch.compile(style_vit)

        Train_dit(dit,style_vit,vae,train_loader,optimizer,scaler,device)


