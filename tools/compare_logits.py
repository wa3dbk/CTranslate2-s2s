#!/usr/bin/env python3
"""Compare CT2 vs Marian step-0 logits dumped as raw float32 buffers.

Produce dumps by rebuilding each with the logit-dump probes enabled and
running greedy, single-sentence decoding:

  CT2_S2S_DUMP_LOGITS=/tmp/ct2.logits.bin \
    ./build/cli/ct2-translator --model <model> --src one_line.bpe \
      --out /tmp/ct2.out --beam_size 1 --batch_size 1

  MARIAN_DUMP_LOGITS=/tmp/marian.logits.bin \
    marian-decoder -m model.npz -v src.yml trg.yml -b 1 \
      --mini-batch 1 --maxi-batch 1 < one_line.bpe > /tmp/marian.out

Then:

  python3 tools/compare_logits.py --ct2 /tmp/ct2.logits.bin \
      --marian /tmp/marian.logits.bin --vocab path/to/target_vocabulary.json

Vocab is optional; if omitted, token ids are shown without surface forms.
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path


def read_floats(path):
    data = Path(path).read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path}: size {len(data)} not divisible by 4")
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def load_vocab(path):
    """Load an id -> token mapping from target_vocabulary.json or Marian YAML."""
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        raw = json.loads(text)
        if isinstance(raw, list):
            return {i: tok for i, tok in enumerate(raw)}
        if isinstance(raw, dict):
            return {int(v): k for k, v in raw.items()}
        raise ValueError(f"unsupported JSON vocab format in {path}")
    # Marian YAML: flat `token: id` per line (quoted keys allowed).
    out = {}
    pat = re.compile(r'^"?(.*?)"?\s*:\s*(\d+)\s*$')
    for line in text.splitlines():
        m = pat.match(line.rstrip())
        if m:
            out[int(m.group(2))] = m.group(1)
    if not out:
        raise ValueError(f"no entries parsed from vocab {path}")
    return out


def rank_of(xs, tid):
    """Rank of tid in xs when sorted descending (0 = argmax)."""
    v = xs[tid]
    return sum(1 for x in xs if x > v)


def logsumexp(xs):
    m = max(xs)
    s = sum(math.exp(x - m) for x in xs)
    return m + math.log(s)


def topk(xs, k):
    return sorted(range(len(xs)), key=lambda i: -xs[i])[:k]


def pearson(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True)
    ap.add_argument("--marian", required=True)
    ap.add_argument("--vocab", help="target_vocabulary.json or Marian YAML (optional)")
    ap.add_argument("-k", "--topk", type=int, default=20)
    ap.add_argument("--unk-id", type=int, default=1)
    ap.add_argument("--eos-id", type=int, default=0)
    ap.add_argument("--marian-npz", help="original Marian NPZ (reads decoder_ff_logit_l2_b)")
    ap.add_argument("--ct2-bias-bin", help="CT2 bias dump from CT2_S2S_DUMP_BIAS")
    args = ap.parse_args()

    ct2 = read_floats(args.ct2)
    mar = read_floats(args.marian)
    vocab = load_vocab(args.vocab)

    print(f"CT2    len={len(ct2)}  min={min(ct2):.4f}  max={max(ct2):.4f}  argmax={ct2.index(max(ct2))}")
    print(f"Marian len={len(mar)}  min={min(mar):.4f}  max={max(mar):.4f}  argmax={mar.index(max(mar))}")

    if len(ct2) != len(mar):
        print(f"\nSIZE MISMATCH ({len(ct2)} vs {len(mar)}): vocab or format differs; aborting detailed diff.")
        sys.exit(1)

    # Both sides dump pre-softmax logits for non-factored vocab; normalize to
    # log-probs for a scale-invariant comparison (logit offset becomes 0).
    ct2_logp = [x - logsumexp(ct2) for x in ct2]
    mar_logp = [x - logsumexp(mar) for x in mar]

    print(f"\nargmax match: CT2={ct2.index(max(ct2))}  Marian={mar.index(max(mar))}  "
          f"{'YES' if ct2.index(max(ct2)) == mar.index(max(mar)) else 'NO'}")
    print(f"Pearson(logits raw)            = {pearson(ct2, mar):.4f}")
    print(f"Pearson(logprobs normalized)   = {pearson(ct2_logp, mar_logp):.4f}")
    print(f"max|ct2_logp - mar_logp|       = {max(abs(a - b) for a, b in zip(ct2_logp, mar_logp)):.4f}")

    def fmt_row(tid, a_val, b_val):
        tok = f"'{vocab[tid]}'" if vocab and tid in vocab else ""
        return f"  id={tid:>6}  {tok:<20}  CT2_logp={a_val:+.3f}  Marian={b_val:+.3f}  diff={a_val - b_val:+.3f}"

    print(f"\nTop-{args.topk} by CT2:")
    for tid in topk(ct2_logp, args.topk):
        print(fmt_row(tid, ct2_logp[tid], mar_logp[tid]))

    print(f"\nTop-{args.topk} by Marian:")
    for tid in topk(mar_logp, args.topk):
        print(fmt_row(tid, ct2_logp[tid], mar_logp[tid]))

    # Sentinel-id reporting: UNK/EOS ranks + raw logits on each side. A
    # chronically over-ranked UNK on the CT2 side was what motivated this
    # tool, so always print these even without --vocab.
    print(f"\nSentinels (rank is 0-indexed over V tokens):")
    for name, tid in (("unk", args.unk_id), ("eos", args.eos_id)):
        tok = f"'{vocab[tid]}'" if vocab and tid in vocab else ""
        print(f"  {name} id={tid} {tok:<16}  "
              f"CT2 rank={rank_of(ct2, tid):>5}  logit={ct2[tid]:+.4f}  "
              f"Marian rank={rank_of(mar, tid):>5}  logit={mar[tid]:+.4f}")

    # Output-bias byte-compare: static-param check that doesn't need re-decode.
    if args.marian_npz and args.ct2_bias_bin:
        try:
            import numpy as np
        except ImportError:
            print("\n[bias check skipped: numpy not available]")
            return
        print(f"\nOutput bias (decoder_ff_logit_l2_b) byte-compare:")
        npz = np.load(args.marian_npz)
        mb = npz["decoder_ff_logit_l2_b"].reshape(-1).astype(np.float32)
        cb = np.frombuffer(Path(args.ct2_bias_bin).read_bytes(), dtype=np.float32)
        print(f"  marian: n={mb.size} min={mb.min():+.4f} max={mb.max():+.4f}")
        print(f"  ct2   : n={cb.size} min={cb.min():+.4f} max={cb.max():+.4f}")
        if mb.size != cb.size:
            print(f"  SIZE MISMATCH")
        else:
            d = mb - cb
            print(f"  max|.|={float(abs(d).max()):.6e}  "
                  f"bytes-equal={bool((mb == cb).all())}")
            print(f"  b[unk={args.unk_id}]: marian={mb[args.unk_id]:+.4f} ct2={cb[args.unk_id]:+.4f}")
            print(f"  b[eos={args.eos_id}]: marian={mb[args.eos_id]:+.4f} ct2={cb[args.eos_id]:+.4f}")


if __name__ == "__main__":
    main()
