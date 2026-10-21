#!/usr/bin/env python3
"""Compare Marian's dumped h_tilde to CT2's dumped h_tilde (both float32,
length D = dim-emb).

Usage:
  python3 tools/compare_htilde.py \
      --marian /tmp/marian.htilde.bin \
      --ct2    /tmp/ct2.htilde.bin
"""
import argparse
from pathlib import Path

import numpy as np


def stats(name, x):
    print(f"  {name:7s}: n={x.size}  min={x.min():+.4f} max={x.max():+.4f} "
          f"mean={x.mean():+.4f} std={x.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marian", required=True)
    ap.add_argument("--ct2", required=True)
    args = ap.parse_args()

    m = np.frombuffer(Path(args.marian).read_bytes(), dtype=np.float32)
    c = np.frombuffer(Path(args.ct2).read_bytes(), dtype=np.float32)

    print("[stats]")
    stats("marian", m)
    stats("ct2",    c)

    if m.size != c.size:
        print(f"\nSIZE MISMATCH: marian={m.size} ct2={c.size} -- aborting diff")
        return

    diff = m - c
    cos = float(np.dot(m, c) / (np.linalg.norm(m) * np.linalg.norm(c) + 1e-12))
    print("\n[diff]")
    print(f"  max|.|={np.abs(diff).max():.6f}  mean|.|={np.abs(diff).mean():.6f}  "
          f"cosine={cos:.4f}  pearson={np.corrcoef(m, c)[0,1]:.4f}")

    # Scale-ratio check: if marian = alpha * ct2 (any constant), alpha stays nearly constant.
    mask = np.abs(c) > 1e-3
    if mask.any():
        ratio = m[mask] / c[mask]
        print(f"  ratio marian/ct2 (|ct2|>1e-3): mean={ratio.mean():+.4f} "
              f"std={ratio.std():.4f} min={ratio.min():+.4f} max={ratio.max():+.4f}")

    order = np.argsort(-np.abs(diff))[:10]
    print("\n[top-10 deviating dims]")
    for i in order:
        print(f"  dim={int(i):>4}  marian={m[i]:+.4f}  ct2={c[i]:+.4f}  diff={diff[i]:+.4f}")


if __name__ == "__main__":
    main()
