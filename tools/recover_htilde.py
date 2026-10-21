#!/usr/bin/env python3
"""Recover Marian's effective h_tilde from a logits dump, and compare it to
CT2's dumped h_tilde.

Since both sides share the output layer (W = Wemb, b = decoder_ff_logit_l2_b)
byte-for-byte, Marian's logits satisfy

    logits = h_tilde @ W.T + b

so h_tilde = (logits - b) @ pinv(W.T). With V > D the system is
overdetermined and has an exact solution (modulo numerics).

Usage:
  python3 tools/recover_htilde.py \
      --marian-logits /tmp/marian.logits.bin \
      --ct2-logits    /tmp/ct2.logits.bin    \
      --ct2-htilde    /tmp/ct2.htilde.bin    \
      --npz /path/to/model.npz
"""
import argparse
from pathlib import Path

import numpy as np


def read_floats(path):
    return np.frombuffer(Path(path).read_bytes(), dtype=np.float32)


def recover(logits, W, b):
    """Return h_tilde of shape [D] from logits [V] under logits = h_tilde @ W.T + b."""
    rhs = (logits - b).astype(np.float64)
    # W is [V, D]; least-squares solve W @ h = rhs
    sol, *_ = np.linalg.lstsq(W.astype(np.float64), rhs, rcond=None)
    return sol.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marian-logits", required=True)
    ap.add_argument("--ct2-logits", required=True)
    ap.add_argument("--ct2-htilde", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bias-key", default="decoder_ff_logit_l2_b")
    ap.add_argument("--emb-key", default="Wemb")
    args = ap.parse_args()

    npz = np.load(args.npz)
    W = npz[args.emb_key].astype(np.float32)  # [V, D]
    b = npz[args.bias_key].reshape(-1).astype(np.float32)  # [V]
    print(f"Wemb {W.shape}  bias {b.shape}")

    ct2_logits = read_floats(args.ct2_logits)
    mar_logits = read_floats(args.marian_logits)
    ct2_htilde = read_floats(args.ct2_htilde)

    assert ct2_logits.size == b.size, (ct2_logits.size, b.size)
    assert mar_logits.size == b.size, (mar_logits.size, b.size)
    assert ct2_htilde.size == W.shape[1], (ct2_htilde.size, W.shape[1])

    # Sanity-check the recovery on CT2's own logits: feed CT2 logits in,
    # solve, compare to CT2's actual h_tilde. Should be ~0 if math is right.
    ct2_recovered = recover(ct2_logits, W, b)
    err = ct2_recovered - ct2_htilde
    print(f"\n[sanity] recover(ct2_logits) vs ct2_htilde: "
          f"max|diff|={np.abs(err).max():.6f}  "
          f"mean|diff|={np.abs(err).mean():.6f}")

    # The actual comparison: Marian's effective h_tilde vs CT2's h_tilde.
    mar_htilde = recover(mar_logits, W, b)
    diff = mar_htilde - ct2_htilde
    print(f"\n[compare] marian_htilde vs ct2_htilde:")
    print(f"  marian: min={mar_htilde.min():.4f} max={mar_htilde.max():.4f} "
          f"mean={mar_htilde.mean():.4f} std={mar_htilde.std():.4f}")
    print(f"  ct2   : min={ct2_htilde.min():.4f} max={ct2_htilde.max():.4f} "
          f"mean={ct2_htilde.mean():.4f} std={ct2_htilde.std():.4f}")
    print(f"  diff  : max|.|={np.abs(diff).max():.4f}  "
          f"mean|.|={np.abs(diff).mean():.4f}  "
          f"cosine={float(np.dot(mar_htilde, ct2_htilde) / (np.linalg.norm(mar_htilde) * np.linalg.norm(ct2_htilde))):.4f}")

    # Where does CT2 h_tilde deviate most?
    order = np.argsort(-np.abs(diff))[:10]
    print("\nTop-10 deviating dims:")
    for i in order:
        print(f"  dim={int(i):>4}  marian={mar_htilde[i]:+.4f}  "
              f"ct2={ct2_htilde[i]:+.4f}  diff={diff[i]:+.4f}")


if __name__ == "__main__":
    main()
