import os
import yaml
import argparse
import random
from PIL import Image
from tqdm import tqdm

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    min_dim = min(w, h)
    left   = (w - min_dim) // 2
    top    = (h - min_dim) // 2
    return img.crop((left, top, left + min_dim, top + min_dim))

def augment_image(img: Image.Image) -> Image.Image:
    # random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # random rotation between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    return img

def preprocess_image(in_path, out_path, size, quality):
    img = Image.open(in_path).convert("RGB")
    if img.width < size[0] or img.height < size[1]:
        print(f"⚠️  Skipping too-small image: {in_path}")
        return False

    img = center_crop(img)
    img = img.resize(size, resample=Image.LANCZOS)
    img.save(out_path, format="JPEG", quality=quality)
    return True

def main(augment_count: int):
    cfg = load_config()
    raw_dir   = cfg["raw_data_dir"]
    proc_dir  = cfg["proc_data_dir"]
    size      = tuple(cfg["preprocess"]["target_size"])
    quality   = cfg["preprocess"]["quality"]

    os.makedirs(proc_dir, exist_ok=True)
    files = [f for f in os.listdir(raw_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in tqdm(files, desc="Preprocessing images"):
        src = os.path.join(raw_dir, fname)
        base, ext = os.path.splitext(fname)

        dst = os.path.join(proc_dir, f"{base}.jpg")
        ok = preprocess_image(src, dst, size, quality)
        if not ok:
            continue

        for i in range(augment_count):
            aug = Image.open(dst)
            aug = augment_image(aug)
            aug_name = os.path.join(proc_dir, f"{base}_aug{i+1}.jpg")
            aug.save(aug_name, format="JPEG", quality=quality)

    print(f"✅ Done! Processed images (and {augment_count}x augmentations) saved to {proc_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess and optionally augment images")
    p.add_argument(
        "--augment", "-a", type=int, default=0,
        help="Number of augmented versions to create per image"
    )
    args = p.parse_args()
    main(args.augment)
