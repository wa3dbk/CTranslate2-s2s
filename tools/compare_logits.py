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
    """Load an id -> token mapping from target_vocabulary.json."""
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return {i: tok for i, tok in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(v): k for k, v in raw.items()}
    raise ValueError(f"unsupported vocab format in {path}")


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
    ap.add_argument("--vocab", help="Path to target_vocabulary.json (optional)")
    ap.add_argument("-k", "--topk", type=int, default=20)
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


if __name__ == "__main__":
    main()
