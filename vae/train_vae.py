import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models import Encoder, Decoder, reparameterize

class ImageDataset(Dataset):
    """Flat-folder dataset of 512x512 JPEGs."""
    def __init__(self, folder, transform=None):
        self.paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg","jpeg","png"))
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
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    mf     = torch.channels_last

    # Hyperparameters
    vae_cfg    = cfg["vae"]
    latent_dim = int(vae_cfg["latent_dim"])
    base_ch    = int(vae_cfg["channels"])
    lr         = float(vae_cfg["lr"])
    beta_max   = float(vae_cfg["kl_beta"])
    batch_size = int(vae_cfg["batch_size"])
    epochs     = int(vae_cfg["epochs"])

    # DataLoader with parallel workers
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [0,1]→[-1,1]
    ])
    ds = ImageDataset(cfg["proc_data_dir"], transform=transform)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        drop_last=True, persistent_workers=True, prefetch_factor=2
    )

    # Model init
    enc = Encoder(in_channels=3, base_channels=base_ch, latent_dim=latent_dim)\
          .to(device=device, memory_format=mf)
    dec = Decoder(out_channels=3, base_channels=base_ch, latent_dim=latent_dim)\
          .to(device=device, memory_format=mf)

    # Optimizer & loss
    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    recon_loss = nn.MSELoss(reduction="sum")

    # Checkpoint directory
    ckpt_dir = os.path.join(cfg["output_dir"], "vae")
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(1, epochs+1):
        enc.train(); dec.train()
        running_loss = 0.0
        # linearly warm-up β from 0 → beta_max
        beta = beta_max * (ep / epochs)

        loop = tqdm(dl, desc=f"Epoch {ep}/{epochs}", unit="batch")
        for imgs in loop:
            imgs = imgs.to(device=device, memory_format=mf)
            mu, logvar = enc(imgs)                   # [B,latent,32,32]
            z = reparameterize(mu, logvar)           # [B,latent,32,32]
            recon = dec(z)                           # [B,3,512,512]

            L_rec = recon_loss(recon, imgs)
            L_kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss  = L_rec + beta * L_kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            loop.set_postfix(avg_loss=running_loss / (loop.n + 1))

        # overwrite a rolling checkpoint each epoch
        torch.save(enc.state_dict(), os.path.join(ckpt_dir, "enc_latest.pth"))
        torch.save(dec.state_dict(), os.path.join(ckpt_dir, "dec_latest.pth"))

    print("✅ VAE training complete! Checkpoints at:", ckpt_dir)

if __name__ == "__main__":
    cfg = load_config()
    train_vae(cfg)
