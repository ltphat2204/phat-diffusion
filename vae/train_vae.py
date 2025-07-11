# vae/train_vae.py
import os
import time
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models import Encoder, Decoder, reparameterize

# ── Enable cuDNN autotune & TF32 for Ampere GPUs ──
torch.backends.cudnn.enabled      = True
torch.backends.cudnn.benchmark    = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

class ImageDataset(Dataset):
    """Flat-folder dataset for processed images."""
    def __init__(self, folder, transform=None):
        self.paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_vae(cfg):
    # Device & memory format
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    mf = torch.channels_last  # for faster convs

    # Parse hyperparameters (ensure correct types)
    vae_cfg    = cfg["vae"]
    latent_dim = int(vae_cfg["latent_dim"])
    base_ch    = int(vae_cfg["channels"])
    lr         = float(vae_cfg["lr"])
    beta       = float(vae_cfg["kl_beta"])
    batch_size = int(vae_cfg["batch_size"])
    epochs     = int(vae_cfg["epochs"])

    var_dir    = vae_cfg.get("output_dir", "models/vae")
    os.makedirs(var_dir, exist_ok=True)

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [0,1]→[-1,1]
    ])
    ds = ImageDataset(cfg["proc_data_dir"], transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Models (moved to channels_last memory format)
    enc = Encoder(in_channels=3, base_channels=base_ch, latent_dim=latent_dim)
    dec = Decoder(out_channels=3, base_channels=base_ch, latent_dim=latent_dim)
    enc = enc.to(device=device, memory_format=mf)
    dec = dec.to(device=device, memory_format=mf)

    # Optimizer & loss
    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    recon_loss = nn.MSELoss(reduction="sum")

    # Checkpoint directory
    ckpt_dir = os.path.join(cfg["output_dir"], "vae")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    for ep in range(1, epochs+1):
        enc.train(); dec.train()
        total_loss = 0.0
        loop = tqdm(dl, desc=f"Epoch {ep}/{epochs}", unit="batch")
        for imgs in loop:
            # Move to GPU with channels_last
            imgs = imgs.to(device=device, memory_format=mf)

            # Forward
            mu, logvar = enc(imgs)
            z          = reparameterize(mu, logvar)
            recon      = dec(z)

            # Losses
            L_recon = recon_loss(recon, imgs)
            L_kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss    = L_recon + beta * L_kl

            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        # Save checkpoints each epoch
        torch.save(enc.state_dict(), os.path.join(ckpt_dir, "enc_latest.pth"))
        torch.save(dec.state_dict(), os.path.join(ckpt_dir, "dec_latest.pth"))

    print("✅ VAE training complete!")
    torch.save(enc.state_dict(), os.path.join(var_dir, "enc.pth"))
    torch.save(dec.state_dict(), os.path.join(var_dir, "dec.pth"))
    print(f"Models saved to {var_dir}")

if __name__ == "__main__":
    config = load_config()
    train_vae(config)
