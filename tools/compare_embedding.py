#!/usr/bin/env python3
"""Compare CT2's `decoder/embeddings/weight` (= tied output weight via alias)
to Marian's embedding matrix.

Usage:
  # First extract CT2's embedding to a raw file:
  python3 tools/inspect_model_bin.py /path/to/ct2_s2s_model/model.bin \
      --name decoder/embeddings/weight --out /tmp/ct2.emb.bin

  # Then compare against the npz:
  python3 tools/compare_embedding.py --ct2 /tmp/ct2.emb.bin \
      --npz /path/to/model.npz --vocab 5001 --dim 512
"""
import argparse
from pathlib import Path

import numpy as np


MARIAN_EMB_KEYS = ("Wemb", "encoder_Wemb", "decoder_Wemb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True, help="raw float32 dump of decoder/embeddings/weight")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab", type=int, help="vocab size (optional, inferred otherwise)")
    ap.add_argument("--dim", type=int, help="embedding dim (optional, inferred otherwise)")
    args = ap.parse_args()

    raw = Path(args.ct2).read_bytes()
    ct2 = np.frombuffer(raw, dtype=np.float32)
    print(f"CT2 emb: {ct2.size} floats "
          f"min={ct2.min():.4f} max={ct2.max():.4f} mean={ct2.mean():.4f}")

    npz = np.load(args.npz)
    present = [k for k in MARIAN_EMB_KEYS if k in npz.files]
    if not present:
        print(f"no Marian embedding key found. keys with 'emb': "
              f"{[k for k in npz.files if 'emb' in k.lower()]}")
        return

    for k in present:
        mar = npz[k].astype(np.float32)
        print(f"\n{k}: shape={mar.shape} "
              f"min={mar.min():.4f} max={mar.max():.4f} mean={mar.mean():.4f}")
        flat = mar.reshape(-1)
        if flat.size != ct2.size:
            print(f"  SIZE MISMATCH: ct2={ct2.size} vs {k}={flat.size}")
            continue
        d = ct2 - flat
        print(f"  flat   max|diff|={np.abs(d).max():.6f} "
              f"mean|diff|={np.abs(d).mean():.6f} "
              f"identical={np.allclose(ct2, flat, atol=1e-5)}")

        # CT2 stores [V, D] row-major; Marian npz may be [V, D] or [D, V].
        if args.vocab and args.dim:
            v, d_ = args.vocab, args.dim
            ct2_mat = ct2.reshape(v, d_)
            for shape in ((v, d_), (d_, v)):
                try:
                    mar_mat = mar.reshape(*shape)
                except ValueError:
                    continue
                for transpose in (False, True):
                    cand = mar_mat.T if transpose else mar_mat
                    if cand.shape != ct2_mat.shape:
                        continue
                    diff = np.abs(ct2_mat - cand)
                    print(f"  as {shape}{'^T' if transpose else ''}: "
                          f"max|diff|={diff.max():.6f} "
                          f"identical={np.allclose(ct2_mat, cand, atol=1e-5)}")


if __name__ == "__main__":
    main()
