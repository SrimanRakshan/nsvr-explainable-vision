#!/usr/bin/env python3
"""
Step 1: Download & setup CLEVR dataset (train/val/test).
"""

import os
import hashlib
import requests
from tqdm import tqdm
from zipfile import ZipFile

CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
CLEVR_MD5 = "b11922020e72d0cd9154779b2d3d07d2"
OUT_DIR = "data/clevr"
ZIP_PATH = os.path.join(OUT_DIR, "CLEVR_v1.0.zip")


def md5(fname, chunk_size=8192):
    h = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with open(out_path, "wb") as f, tqdm(
        desc="Downloading CLEVR",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))


def extract(zip_path, out_dir):
    print("Extracting...")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    if not os.path.exists(ZIP_PATH):
        print("Downloading CLEVR dataset...")
        download(CLEVR_URL, ZIP_PATH)
    else:
        print("Zip already exists, skipping download.")

    print("Verifying checksum...")
    if md5(ZIP_PATH) != CLEVR_MD5:
        raise ValueError("MD5 checksum does not match! Delete and re-download.")
    print("Checksum OK.")

    extract(ZIP_PATH, OUT_DIR)
    print("Done. CLEVR dataset ready at:", OUT_DIR)


if __name__ == "__main__":
    main()