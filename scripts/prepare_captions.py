import os, json, re

RAW_FILE = "data/captions_raw.jsonl"
OUT_FILE = "data/captions.jsonl"
LAT_DIR  = "data/latents"

raw_caps = {}
with open(RAW_FILE, encoding="utf-8") as rf:
    for line in rf:
        obj = json.loads(line)
        raw_caps[obj["filename"]] = obj["text"]

pattern = re.compile(r"^(?P<base>.+?)(?:_aug\d+)?\.pt$")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as wf:
    for fname in sorted(os.listdir(LAT_DIR)):
        if not fname.endswith(".pt"):
            continue
        m = pattern.match(fname)
        if not m:
            print(f"⚠️  Skipping unexpected file: {fname}")
            continue

        base = m.group("base")
        jpg  = f"{base}.jpg"
        text = raw_caps.get(jpg)

        if not text:
            print(f"⚠️  No caption found for {jpg}; skipping {fname}")
            continue

        entry = {"file": fname, "text": text}
        wf.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Wrote captions for {len(raw_caps)} originals + augmentations → {OUT_FILE}")
