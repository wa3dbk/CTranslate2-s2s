#!/usr/bin/env python3
"""Compare CT2's loaded output bias to Marian's `decoder_ff_logit_l2_b`.

Dump CT2's bias by running one translate call with:
  CT2_S2S_DUMP_BIAS=/tmp/ct2.bias.bin ./build/cli/ct2-translator ...

Then:
  python3 tools/compare_bias.py --ct2 /tmp/ct2.bias.bin --npz model.npz \
      [--vocab target_vocabulary.json]
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_floats(path):
    data = Path(path).read_bytes()
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


def load_vocab(path):
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return {i: tok for i, tok in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(v): k for k, v in raw.items()}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab")
    ap.add_argument("--key", default="decoder_ff_logit_l2_b")
    args = ap.parse_args()

    ct2 = read_floats(args.ct2)
    npz = np.load(args.npz)
    if args.key not in npz:
        print(f"keys containing 'logit_l2': {[k for k in npz.files if 'logit_l2' in k]}")
        raise SystemExit(f"key {args.key} not in npz")
    mar = npz[args.key].reshape(-1).astype(np.float32)
    vocab = load_vocab(args.vocab)

    print(f"CT2    len={len(ct2)}  min={ct2.min():.4f}  max={ct2.max():.4f}  mean={ct2.mean():.4f}")
    print(f"Marian len={len(mar)}  min={mar.min():.4f}  max={mar.max():.4f}  mean={mar.mean():.4f}")

    if len(ct2) != len(mar):
        raise SystemExit(f"size mismatch: {len(ct2)} vs {len(mar)}")

    diff = ct2 - mar
    print(f"\nelementwise  max|diff|={np.abs(diff).max():.6f}  mean|diff|={np.abs(diff).mean():.6f}")
    print(f"identical (atol=1e-5): {np.allclose(ct2, mar, atol=1e-5)}")

    # If not identical, highlight the biggest offenders.
    if not np.allclose(ct2, mar, atol=1e-5):
        order = np.argsort(-np.abs(diff))[:20]
        print("\nTop-20 by |CT2 - Marian|:")
        for i in order:
            tok = f"'{vocab[int(i)]}'" if vocab and int(i) in vocab else ""
            print(f"  id={int(i):>6}  {tok:<20}  CT2={ct2[i]:+.4f}  Marian={mar[i]:+.4f}  diff={diff[i]:+.4f}")

        # Fragment-vs-full word stratification: is the bug concentrated on @@ tokens?
        if vocab:
            frag_ids = [i for i, t in vocab.items() if t.endswith("@@")]
            full_ids = [i for i, t in vocab.items() if not t.endswith("@@")]
            if frag_ids and full_ids:
                frag_diff = np.abs(diff[frag_ids]).mean()
                full_diff = np.abs(diff[full_ids]).mean()
                print(f"\nmean|diff| on @@-fragments  : {frag_diff:.4f}  (n={len(frag_ids)})")
                print(f"mean|diff| on full-word tok : {full_diff:.4f}  (n={len(full_ids)})")


if __name__ == "__main__":
    main()
