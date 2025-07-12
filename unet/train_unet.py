import os
import yaml
import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from models import load_unet, load_scheduler, load_clip

class LatentTextDataset(Dataset):
    """Loads spatial latents (CxHxW) and captions."""
    def __init__(self, lat_dir, captions_file, tokenizer, max_length=77):
        import json
        self.lat_dir = lat_dir
        self.examples = [json.loads(l) for l in open(captions_file, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        lat = torch.load(os.path.join(self.lat_dir, ex["file"]))  # [C, H, W]
        tokens = self.tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return lat, tokens.input_ids[0], tokens.attention_mask[0]

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_unet(cfg):
    # ── Device & Performance Flags ──────────────────
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    mf = torch.channels_last

    # ── Hyperparameters ─────────────────────────────
    unet_cfg   = cfg["unet"]
    batch_size = int(unet_cfg["batch_size"])
    lr         = float(unet_cfg["lr"])
    max_steps  = int(unet_cfg["max_steps"])
    ckpt_every = int(unet_cfg["ckpt_every"])

    # ── Load models & scheduler ──────────────────────
    unet      = load_unet(device=device, memory_format=mf)
    scheduler = load_scheduler()
    tokenizer, text_encoder = load_clip(device=device)

    # ── Enable gradient checkpointing ────────────────
    unet.enable_gradient_checkpointing()

    # ── CLIP projection 512→768 ──────────────────────
    clip_dim  = text_encoder.config.hidden_size       # 512
    cross_dim = unet.config.cross_attention_dim       # 768
    proj      = torch.nn.Linear(clip_dim, cross_dim).to(device)

    # ── Dataset & DataLoader ─────────────────────────
    ds = LatentTextDataset(
        lat_dir       = cfg["latent_dir"],
        captions_file = cfg["captions_file"],
        tokenizer     = tokenizer
    )
    num_workers = min(multiprocessing.cpu_count(), 8)
    dl = DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = True,
        prefetch_factor    = 2,
        persistent_workers = True,
        drop_last          = True
    )

    # ── Optimizer & AMP scaler ───────────────────────
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(proj.parameters()), lr=lr
    )
    scaler = GradScaler()

    # ── Training loop ────────────────────────────────
    ckpt_dir = os.path.join(cfg["output_dir"], "unet")
    os.makedirs(ckpt_dir, exist_ok=True)
    dl_iter = iter(dl)
    tbar = tqdm(range(1, max_steps+1), desc="UNet training", unit="step")

    for step in tbar:
        # fetch next batch (wrap around if needed)
        try:
            latents, input_ids, attn_mask = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            latents, input_ids, attn_mask = next(dl_iter)

        # to device & proper memory format
        latents   = latents.to(device=device).to(memory_format=mf)
        input_ids = input_ids.to(device=device)
        attn_mask = attn_mask.to(device=device)

        # sample noise & timesteps
        noise     = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device
        )
        noisy = scheduler.add_noise(latents, noise, timesteps)

        # get CLIP embeddings + projection
        with torch.no_grad():
            clip_emb = text_encoder(input_ids, attention_mask=attn_mask)[0]  # [B,L,512]
        text_emb = proj(clip_emb)                                             # [B,L,768]

        # forward + loss under autocast (now passing device_type)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            noise_pred = unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(noise_pred, noise)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        tbar.set_postfix(loss=f"{loss.item():.4f}")

        # rolling checkpoint
        if step % ckpt_every == 0 or step == max_steps:
            ckpt_path = os.path.join(ckpt_dir, "unet_latest.pth")
            torch.save(unet.state_dict(), ckpt_path)

    print(f"✅ UNet training complete! Latest checkpoint at:\n  {ckpt_path}")

if __name__ == "__main__":
    cfg = load_config()
    train_unet(cfg)
