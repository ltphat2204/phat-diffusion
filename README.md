# Phat-SD: From Scratch Stable Diffusion Pipeline

This repository implements a full “from-scratch” Stable Diffusion-style pipeline, tailored to Renaissance-sculpture images:

1. **Data collection** (Unsplash scraper)  
2. **Preprocessing & augmentation** (512x512 JPGs)  
3. **VAE training** (spatial latents)  
4. **Latent encoding** (save 4x64x64 tensors)  
5. **Caption preparation** (map JPG → PT + text)  
6. **Latent UNet training** (text-conditioned diffusion)  
7. **Sampling** (DDIM + classifier-free guidance)

---

## 📋 Prerequisites

- **CUDA-enabled GPU** (RTX 3060 or better)  
- **Python 3.10+**  
- `git`, `make` (optional)

---

## 🚀 Quickstart

### 1. Clone & venv

```bash
git clone https://github.com/yourname/phat-sd.git
cd phat-sd
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install diffusers transformers accelerate safetensors datasets pillow tqdm pyyaml requests einops xformers
```

### 3. Configure

- Copy `.env.example → .env`, fill in:
  ```ini
  UNSPLASH_ACCESS_KEY=your_unsplash_key
  ```
- Review `configs/default.yaml`—adjust:
  - Data paths: `raw_data_dir`, `proc_data_dir`, `latent_dir`  
  - Model hyperparameters (VAE, UNet)  
  - `output_dir` for checkpoints  
  - `device: "cuda"`  

---

## 🔧 Pipeline Steps

Follow in order:

### A. Download images

```bash
python scripts/download.py --start-page 1
```

---

### B. Preprocess & augment

```bash
python scripts/preprocess.py --augment 2
```

---

### C. Train VAE

```bash
python vae/train_vae.py
```

---

### D. Encode latents

```bash
python scripts/encode_latents.py
```

---

### E. Prepare captions

```bash
python scripts/prepare_captions.py
```

---

### F. Train Latent UNet

```bash
python unet/train_unet.py
```

---

### G. Sample

```bash
python sample.py \
  --prompt "A marble bust of David, Renaissance style" \
  --steps 50 \
  --scale 7.5 \
  --out outputs/david.png
```

---

## 🛠️ Tips & Troubleshooting

- **Disk space**: use rolling checkpoints (`enc_latest.pth`, `unet_latest.pth`) to save space.  
- **Rate-limit**: Unsplash free tier → 50 req/hr. Use `--start-page` to resume.  
- **Validation**: split 5–10 % of latents to monitor overfitting.  
- **Performance**: DataLoader with `num_workers`, `pin_memory`, `persistent_workers`.  
- **Guidance**: sample with `guidance_scale ≥ 7` for sharper outputs.

---

## 📁 Project Structure

```
.
├── .venv/               
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── latents/
│   ├── captions_raw.jsonl
│   └── captions.jsonl
├── scripts/
│   ├── download.py
│   ├── preprocess.py
│   ├── encode_latents.py
│   └── prepare_captions.py
├── vae/
│   ├── models.py
│   └── train_vae.py
├── unet/
│   ├── models.py
│   └── train_unet.py
├── sample.py
└── README.md
```

---

## 📜 License

MIT © Your Name  
Feel free to adapt and extend!
