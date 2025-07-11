# sample.py
import os
import yaml
import argparse
import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionPipeline
)
from transformers import CLIPTokenizer, CLIPTextModel

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate with your custom SD model")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps",  type=int, default=50)
    parser.add_argument("--scale",  type=float, default=7.5)
    parser.add_argument("--out",    type=str,   default="out.png")
    args = parser.parse_args()

    # ── Load config & device ─────────────────────────
    cfg    = load_config()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # ── Load a spatial VAE (4×64×64 latents) ─────────
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"  # official SD VAE weights
    ).to(device)

    # ── Load your fine-tuned UNet ────────────────────
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).to(device)
    unet_ckpt = os.path.join(cfg["output_dir"], "unet", "unet_latest.pth")
    if not os.path.isfile(unet_ckpt):
        raise FileNotFoundError(f"Cannot find UNet checkpoint at {unet_ckpt}")
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device))

    # ── Load CLIP text encoder/tokenizer (large model) ──
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")\
                                .to(device).eval()

    # ── Scheduler (DDIM) ───────────────────────────────
    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    # ── Build the pipeline ─────────────────────────────
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,        # you can omit or add your own
        feature_extractor=None
    ).to(device)
    # optional optimizations
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    # ── Generate ───────────────────────────────────────
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale
    ).images[0]

    # ── Save output ────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    image.save(args.out)
    print(f"✨ Generated image saved to {args.out}")

if __name__ == "__main__":
    main()
