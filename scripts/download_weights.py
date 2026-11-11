"""
Utility script to download full uncompressed float32 model checkpoints from GitHub Releases.
"""
import os
import urllib.request
import sys

REPO = "uipo9243-svg/Fly-CfC"
TAG = "v1.0.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

FILES = [
    "fly_cfc_v1_final.pth",
    "fly_cfc_v2_3d.pth",
]

def download_weights():
    os.makedirs("models", exist_ok=True)
    for fname in FILES:
        dest = os.path.join("models", fname)
        if os.path.exists(dest):
            print(f"[SKIP] {dest} already exists.")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"[DOWNLOADING] {fname} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[DONE] Saved to {dest} ({os.path.getsize(dest)/1024/1024:.2f} MB)")
        except Exception as e:
            print(f"[WARN] Could not download {fname}: {e}")
            print(f"Note: fp16 checkpoints are already included in the models/ directory and ready to run.")

if __name__ == "__main__":
    download_weights()
