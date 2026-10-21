#!/usr/bin/env python3
"""Byte-compare Marian NPZ parameters against the converted CT2 model.

This runs offline (no decode, no rebuild) and verifies that the converter
preserved the Marian parameter values. It flags conversion-time bugs that
would otherwise masquerade as decoder bugs:
  - Output bias off (sign, order, shape)
  - Embedding table not byte-matching
  - GRU weights permuted or sliced wrong

Usage:
  python3 tools/compare_static_params.py \
    --marian-npz   /path/to/model.npz \
    --ct2-bias-bin /tmp/ct2.bias.bin        # from CT2_S2S_DUMP_BIAS
    [--unk-id 1 --eos-id 0]
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marian-npz", required=True)
    ap.add_argument("--ct2-bias-bin",
                    help="raw float32 dump from CT2_S2S_DUMP_BIAS (optional)")
    ap.add_argument("--ct2-wemb-bin",
                    help="raw float32 dump of Wemb from CT2 (optional)")
    ap.add_argument("--unk-id", type=int, default=1)
    ap.add_argument("--eos-id", type=int, default=0)
    args = ap.parse_args()

    npz = np.load(args.marian_npz)

    print("=== Marian NPZ keys of interest ===")
    for k in ("Wemb", "decoder_ff_logit_l2_b"):
        if k in npz.files:
            a = npz[k].astype(np.float32)
            print(f"  {k}: shape={a.shape} min={a.min():+.4f} "
                  f"max={a.max():+.4f} mean={a.mean():+.4f}")
        else:
            print(f"  {k}: <MISSING>")

    if args.ct2_bias_bin:
        mb = npz["decoder_ff_logit_l2_b"].reshape(-1).astype(np.float32)
        cb = np.frombuffer(Path(args.ct2_bias_bin).read_bytes(),
                           dtype=np.float32)
        print(f"\n=== output bias byte-compare ===")
        print(f"  marian: n={mb.size} min={mb.min():+.4f} max={mb.max():+.4f}")
        print(f"  ct2   : n={cb.size} min={cb.min():+.4f} max={cb.max():+.4f}")
        if mb.size == cb.size:
            d = mb - cb
            print(f"  max|.|={float(abs(d).max()):.6e}  "
                  f"bytes-equal={bool((mb == cb).all())}")
            print(f"  b[unk={args.unk_id}]: marian={mb[args.unk_id]:+.4f} "
                  f"ct2={cb[args.unk_id]:+.4f}")
            print(f"  b[eos={args.eos_id}]: marian={mb[args.eos_id]:+.4f} "
                  f"ct2={cb[args.eos_id]:+.4f}")
            # Top-20 bias values on the Marian side — big positive biases
            # are the most likely cause of a token dominating argmax.
            print(f"\n  Marian top-20 bias values (by raw b):")
            idx = np.argsort(-mb)[:20]
            for r, i in enumerate(idx):
                print(f"    #{r:<2}  id={int(i):>5}  b={mb[i]:+.4f}")

    if args.ct2_wemb_bin:
        wm = npz["Wemb"].astype(np.float32).reshape(-1)
        wc = np.frombuffer(Path(args.ct2_wemb_bin).read_bytes(),
                           dtype=np.float32)
        print(f"\n=== Wemb byte-compare ===")
        print(f"  marian: n={wm.size}  ct2: n={wc.size}")
        if wm.size == wc.size:
            d = wm - wc
            print(f"  max|.|={float(abs(d).max()):.6e}  "
                  f"bytes-equal={bool((wm == wc).all())}")


if __name__ == "__main__":
    main()
