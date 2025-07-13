# sample.py
import os
import yaml
import argparse
import torch
from torchvision.utils import save_image
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from vae.models import Decoder

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # ── Parse CLI args ───────────────────────────────
    parser = argparse.ArgumentParser(description="Generate with custom Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps",  type=int, default=50)
    parser.add_argument("--scale",  type=float, default=7.5)
    parser.add_argument("--out",    type=str, default="out.png")
    args = parser.parse_args()

    # ── Load config & set up device ──────────────────
    cfg    = load_config()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    mf     = torch.channels_last

    # ── Load VAE decoder ─────────────────────────────
    dec = Decoder(
        out_channels=3,
        base_channels=int(cfg["vae"]["channels"]),
        latent_dim=int(cfg["vae"]["latent_dim"])
    ).to(device=device, memory_format=mf)
    dec_ckpt = cfg["vae"]["decoder_path"]
    assert os.path.exists(dec_ckpt), f"Decoder checkpoint not found: {dec_ckpt}"
    dec.load_state_dict(torch.load(dec_ckpt, map_location=device))
    dec.eval()

    # ── Load UNet & weights ──────────────────────────
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).to(device=device, memory_format=mf)
    unet_ckpt = os.path.join(cfg["output_dir"], "unet", "unet_latest.pth")
    assert os.path.exists(unet_ckpt), f"UNet checkpoint not found: {unet_ckpt}"
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
    unet.eval()

    # ── Load scheduler ────────────────────────────────
    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    scheduler.set_timesteps(args.steps)

    # ── Load CLIP-B/32 + projection ───────────────────
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")\
                                  .to(device).eval()
    clip_dim  = text_encoder.config.hidden_size
    cross_dim = unet.config.cross_attention_dim
    proj      = torch.nn.Linear(clip_dim, cross_dim).to(device)

    # ── Tokenize prompt & get text embeddings ───────────
    tokens    = tokenizer(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    input_ids = tokens.input_ids.to(device)
    attn_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        clip_emb = text_encoder(input_ids, attention_mask=attn_mask)[0]
    text_emb = proj(clip_emb)

    # ── Prepare initial spatial latent noise ────────────
    latent_dim   = int(cfg["vae"]["latent_dim"])
    spatial_size = 32  # matches VAE downsample (512→256→128→64→32)
    latents = torch.randn((1, latent_dim, spatial_size, spatial_size), device=device)
    latents = latents.to(device=device, memory_format=mf)

    # ── Denoising loop (DDIM) ─────────────────────────
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(latents, t, encoder_hidden_states=text_emb).sample
        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    # ── Decode latent → image ─────────────────────────
    with torch.no_grad():
        img = dec(latents)  # [1,3,512,512]
    img = (img + 1) * 0.5  # map from [-1,1] to [0,1]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(img, args.out)

    print(f"✨ Generated image saved to {args.out}")

if __name__ == "__main__":
    main()
