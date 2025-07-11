import os
import time
import argparse
import requests
import yaml
from dotenv import load_dotenv
import json

# ─── 1. Load .env & Config ────────────────────────
load_dotenv()
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
if not UNSPLASH_ACCESS_KEY:
    raise ValueError("❌ Missing UNSPLASH_ACCESS_KEY in your .env")

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
RAW_DIR     = cfg["raw_data_dir"]
TOTAL_PAGES = cfg["download"]["total_pages"]
PER_PAGE    = cfg["download"]["per_page"]
QUERY       = cfg["download"]["query"]

os.makedirs(RAW_DIR, exist_ok=True)

# ─── 2. Downloader ────────────────────────────────
def download_image(url, path):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)

def fetch_page(page):
    api_url = "https://api.unsplash.com/search/photos"
    headers = {
        "Accept-Version": "v1",
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }
    params = {"query": QUERY, "page": page, "per_page": PER_PAGE}
    resp = requests.get(api_url, headers=headers, params=params, timeout=10)

    # Rate‐limit handling
    if resp.status_code == 403:
        reset = resp.headers.get("X-Ratelimit-Reset")
        wait = int(reset) if reset and reset.isdigit() else 3600
        print(f"⚠️ Rate limit hit. Sleeping for {wait}s…")
        time.sleep(wait + 5)
        return fetch_page(page)

    if resp.status_code != 200:
        print(f"❌ HTTP {resp.status_code} on page {page}: {resp.text[:200]}")
        return False

    try:
        data = resp.json()
    except ValueError:
        print(f"❌ Invalid JSON on page {page}: {resp.text[:200]}")
        return False

    results = data.get("results", [])
    if not results:
        print(f"ℹ️ No results on page {page}.")
        return True

    for i, photo in enumerate(results):
        fname = f"img_p{page}_{i}.jpg"
        out_path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(out_path):
            download_image(photo["urls"]["regular"], out_path)
            print(f"[Page {page}] downloaded {fname}")

        caption = (
            photo.get("alt_description")
            or photo.get("description")
            or cfg["download"]["query"]
        )

        raw_caps = os.path.join("data", "captions_raw.jsonl")
        os.makedirs(os.path.dirname(raw_caps), exist_ok=True)
        with open(raw_caps, "a", encoding="utf-8") as cf:
            cf.write(json.dumps({
                "filename": fname,
                "text": caption
            }, ensure_ascii=False) + "\n")

    return True

# ─── 3. CLI & Main Loop ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable Unsplash downloader")
    parser.add_argument(
        "--start-page", "-s",
        type=int,
        default=1,
        help="Page number to start downloading from (1-based)"
    )
    args = parser.parse_args()

    print(f"🔄 Starting downloads from page {args.start_page} of {TOTAL_PAGES}")
    for pg in range(args.start_page, TOTAL_PAGES + 1):
        ok = fetch_page(pg)
        if not ok:
            print(f"⏸ Stopped at page {pg}. Retry with --start-page {pg}")
            break
        time.sleep(1)
    else:
        print("✅ All pages attempted!")
