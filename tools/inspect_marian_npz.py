#!/usr/bin/env python3
"""Diagnose Marian npz: list all tensor keys with shapes, print any model config,
and highlight anything that could be the output projection weight.

Usage:
  python3 tools/inspect_marian_npz.py /path/to/model.npz
"""
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 2:
        print("usage: inspect_marian_npz.py MODEL.NPZ", file=sys.stderr)
        sys.exit(1)
    npz = np.load(sys.argv[1])

    print("== all keys ==")
    for k in sorted(npz.files):
        v = npz[k]
        print(f"  {k:60s}  shape={str(v.shape):20s}  dtype={v.dtype}")

    print("\n== output-projection candidates ==")
    for k in sorted(npz.files):
        kl = k.lower()
        if "logit" in kl or kl in ("wemb", "encoder_wemb", "decoder_wemb"):
            v = npz[k]
            print(f"  {k:60s}  shape={str(v.shape):20s}")

    print("\n== config / yaml blobs ==")
    for k in npz.files:
        if "yml" in k.lower() or "model.yml" in k.lower() or k.startswith("special"):
            v = npz[k]
            print(f"\n--- {k} (shape={v.shape}, dtype={v.dtype}) ---")
            try:
                data = v.tobytes().decode("utf-8", errors="replace")
                print(data)
            except Exception as e:
                print(f"  (could not decode: {e})")


if __name__ == "__main__":
    main()
