import os
import yaml
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16 as _orig_vgg16, VGG16_Weights
import torchvision.models as _tv_models
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips
from models import Encoder, Decoder, reparameterize

_tv_models.vgg16 = lambda *args, pretrained=None, **kwargs: _orig_vgg16(weights=VGG16_Weights.IMAGENET1K_V1, **kwargs)

class ImageDataset(Dataset):
    """Flat-folder dataset of 512×512 images."""
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
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    mf = torch.channels_last
    vae_cfg = cfg["vae"]
    base_ch = int(vae_cfg["channels"])
    latent_dim = int(vae_cfg["latent_dim"])
    lr = float(vae_cfg["lr"])
    beta_max = float(vae_cfg["kl_beta"])
    perc_weight = float(vae_cfg.get("perc_weight", 0.0))
    batch_size = int(vae_cfg["batch_size"])
    epochs = int(vae_cfg["epochs"])

    perceptual = lpips.LPIPS(net="vgg").to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = ImageDataset(cfg["proc_data_dir"], transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    enc = Encoder(in_channels=3, base_channels=base_ch, latent_dim=latent_dim).to(device=device, memory_format=mf)
    dec = Decoder(out_channels=3, base_channels=base_ch, latent_dim=latent_dim).to(device=device, memory_format=mf)

    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    recon_loss = nn.MSELoss(reduction="sum")

    ckpt_dir = os.path.join(cfg["output_dir"], "vae")
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        enc.train(); dec.train()
        loop = tqdm(dl, desc=f"Epoch {ep}/{epochs}", unit="batch")
        running_loss = 0.0

        t = ep / epochs
        beta = beta_max * (1 - torch.cos(torch.tensor(t * 3.1415926535))) / 2

        for imgs in loop:
            imgs = imgs.to(device=device, memory_format=mf)
            mu, logvar = enc(imgs)
            z = reparameterize(mu, logvar)
            recon = dec(z)
            L_rec = recon_loss(recon, imgs)
            L_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            L_perc = perceptual(recon, imgs).mean() * perc_weight
            loss = L_rec + beta * L_kl + L_perc
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            loop.set_postfix(avg_loss=running_loss / (loop.n + 1))

        sched.step()
        torch.save(enc.state_dict(), os.path.join(ckpt_dir, "enc_latest.pth"))
        torch.save(dec.state_dict(), os.path.join(ckpt_dir, "dec_latest.pth"))

    print("✅ VAE training complete! Checkpoints in:", ckpt_dir)

if __name__ == "__main__":
    cfg = load_config()
    train_vae(cfg)
