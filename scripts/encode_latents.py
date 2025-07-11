import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from vae.models import Encoder, reparameterize

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.paths[idx])

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # load VAE encoder
    enc = Encoder(
        in_channels=3,
        base_channels=int(cfg["vae"]["channels"]),
        latent_dim=int(cfg["vae"]["latent_dim"])
    ).to(device).eval()
    enc.load_state_dict(torch.load(cfg["vae"]["encoder_path"], map_location=device))

    # data
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = ImageDataset(cfg["proc_data_dir"], transform=tf)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    # output folder for latents
    lat_dir = os.path.join("data", "latents")
    os.makedirs(lat_dir, exist_ok=True)

    with torch.no_grad():
        for imgs, fnames in tqdm(dl, desc="Encoding latents"):
            imgs = imgs.to(device)
            mu, logvar = enc(imgs)
            z = reparameterize(mu, logvar)  # [B, latent_dim]
            z = z.cpu()
            for zi, name in zip(z, fnames):
                torch.save(zi, os.path.join(lat_dir, name.replace(".jpg", ".pt")))

    print(f"✅ Saved {len(ds)} latent files to {lat_dir}")

if __name__ == "__main__":
    main()
