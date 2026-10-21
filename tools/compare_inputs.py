#!/usr/bin/env python3
"""Compare each of the three deep-output inputs (embedding, decoder state,
aligned context) side-by-side between Marian and CT2.

Usage:
  python3 tools/compare_inputs.py --dir /tmp
"""
import argparse
from pathlib import Path

import numpy as np


def load(p):
    return np.frombuffer(Path(p).read_bytes(), dtype=np.float32)


def _vec_stats(a, b, label):
    diff = a - b
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    cos = float(np.dot(a, b) / denom)
    print(f"    [{label}] n={a.size}  max|.|={np.abs(diff).max():.6f}  "
          f"mean|.|={np.abs(diff).mean():.6f}  cosine={cos:.4f}")


def report(label, m, c):
    print(f"\n=== {label} ===")
    print(f"  marian: n={m.size}  min={m.min():+.4f} max={m.max():+.4f} "
          f"mean={m.mean():+.4f} std={m.std():.4f}")
    print(f"  ct2   : n={c.size}  min={c.min():+.4f} max={c.max():+.4f} "
          f"mean={c.mean():+.4f} std={c.std():.4f}")
    if m.size != c.size:
        print(f"  SIZE MISMATCH: marian={m.size} ct2={c.size}")
        return
    diff = m - c
    cos = float(np.dot(m, c) / (np.linalg.norm(m) * np.linalg.norm(c) + 1e-12))
    print(f"  max|.|={np.abs(diff).max():.6f}  mean|.|={np.abs(diff).mean():.6f}  "
          f"cosine={cos:.4f}  pearson={np.corrcoef(m, c)[0,1]:.4f}")

    # If size is even, break into two halves and report per-half.
    # For bi-GRU memory (2*dim_rnn=4048) this isolates fwd vs bwd lanes.
    # For attention context (also 2*dim_rnn, read from memory) same split applies.
    if m.size % 2 == 0:
        half = m.size // 2
        _vec_stats(m[:half], c[:half], "lo-half")
        _vec_stats(m[half:], c[half:], "hi-half")

    order = np.argsort(-np.abs(diff))[:5]
    for i in order:
        half_tag = ""
        if m.size % 2 == 0:
            half_tag = "  (lo)" if i < m.size // 2 else "  (hi)"
        print(f"    dim={int(i):>4}  marian={m[i]:+.4f}  ct2={c[i]:+.4f}  "
              f"diff={diff[i]:+.4f}{half_tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp",
                    help="directory containing marian.{emb,state,ctx,htilde}.bin "
                         "and ct2.{emb,state,ctx,htilde}.bin")
    args = ap.parse_args()
    d = Path(args.dir)

    for name in ("memory", "emb", "state", "ctx", "s1", "alignment", "htilde"):
        mp = d / f"marian.{name}.bin"
        cp = d / f"ct2.{name}.bin"
        if not mp.exists() or not cp.exists():
            print(f"[skip {name}] missing: "
                  f"marian={'ok' if mp.exists() else 'MISSING'} "
                  f"ct2={'ok' if cp.exists() else 'MISSING'}")
            continue
        report(name, load(mp), load(cp))


if __name__ == "__main__":
    main()
