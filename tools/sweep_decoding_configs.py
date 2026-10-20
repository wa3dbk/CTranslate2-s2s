#!/usr/bin/env python3
"""Sweep ct2-translator decoding configs and rank by closeness to marian output.

Each config produces one hypothesis file. For every file we compute:
  * WER vs the marian hypothesis file (lower = closer to marian).
  * BLEU vs an optional human reference (higher = better translation).

Results are written to <out-dir>/results.csv sorted by WER ascending.

Example:
  python3 tools/sweep_decoding_configs.py \\
      --binary ./build/cli/ct2-translator \\
      --model  /home/benkheder/nllb/ctr/ct2_s2s_model \\
      --src    ./input.txt \\
      --marian-out ./output_marian.txt \\
      --reference ./reference.txt \\
      --out-dir ./sweep_out
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import sacrebleu
    HAVE_SACREBLEU = True
except ImportError:
    HAVE_SACREBLEU = False


# Edit this grid to taste. Each entry is a dict of CLI flags passed to ct2-translator.
# Keys map directly to --<key> flags. Boolean True adds the bare flag; False omits it.
DEFAULT_GRID = {
    "beam_size":             [5, 8, 12],
    "disable_unk":           [False, True],
    "repetition_penalty":    [1.0, 1.1, 1.2],
    "no_repeat_ngram_size":  [0, 3],
    "length_penalty":        [1.0, 1.2],
}

# Fixed flags applied to every run.
FIXED_FLAGS = {
    "device": "cpu",
    "max_decoding_length": 256,
    "min_decoding_length": 1,
}


def iter_configs(grid):
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def config_tag(cfg):
    parts = []
    for k, v in cfg.items():
        short = "".join(w[0] for w in k.split("_"))
        if isinstance(v, bool):
            parts.append(f"{short}{int(v)}")
        else:
            parts.append(f"{short}{v}")
    return "_".join(parts)


def build_argv(binary, model, src_file, out_file, cfg):
    argv = [binary, "--model", model, "--src", src_file, "--out", out_file]
    for k, v in FIXED_FLAGS.items():
        argv += [f"--{k}", str(v)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        else:
            argv += [f"--{k}", str(v)]
    return argv


def wer(hyp_lines, ref_lines):
    """Corpus-level WER: sum of per-sentence edit distances / sum of ref tokens."""
    if len(hyp_lines) != len(ref_lines):
        raise ValueError(f"line count mismatch: hyp={len(hyp_lines)} ref={len(ref_lines)}")
    total_dist = 0
    total_ref = 0
    for h, r in zip(hyp_lines, ref_lines):
        h_toks = h.split()
        r_toks = r.split()
        total_dist += levenshtein(h_toks, r_toks)
        total_ref += len(r_toks)
    return (total_dist / total_ref) if total_ref else float("inf")


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def read_lines(path, strip_bpe=False):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if strip_bpe:
        lines = [detok_bpe(ln) for ln in lines]
    return lines


def detok_bpe(s):
    # subword-nmt joiner: "foo@@ bar" -> "foobar". Idempotent on already-detok text.
    return s.replace("@@ ", "").replace("@@", "")


def run_one(binary, model, src_file, out_file, cfg, timeout):
    argv = build_argv(binary, model, src_file, out_file, cfg)
    try:
        proc = subprocess.run(
            argv, check=False, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    if proc.returncode != 0:
        return False, proc.stderr.decode("utf-8", errors="replace")[-500:]
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="Path to ct2-translator binary")
    ap.add_argument("--model", required=True, help="Path to CT2 model directory")
    ap.add_argument("--src", required=True, help="Path to source text file")
    ap.add_argument("--marian-out", required=True, help="Path to marian hypothesis file (closeness target)")
    ap.add_argument("--reference", help="Optional path to human reference for BLEU")
    ap.add_argument("--out-dir", required=True, help="Where to write hypothesis files + results.csv")
    ap.add_argument("--configs-json", help="Optional JSON file overriding the grid dict")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-run timeout seconds")
    ap.add_argument("--skip-existing", action="store_true", help="Reuse hypothesis files already present")
    ap.add_argument("--rescore-only", action="store_true",
                    help="Do not decode; only re-score hypothesis files already in --out-dir")
    ap.add_argument("--strip-bpe", action="store_true",
                    help="Remove subword-nmt '@@ ' joiners from hyp/marian/ref before scoring")
    args = ap.parse_args()

    if args.reference and not HAVE_SACREBLEU:
        print("warning: --reference provided but sacrebleu is not installed; BLEU column will be empty",
              file=sys.stderr)

    grid = DEFAULT_GRID
    if args.configs_json:
        with open(args.configs_json) as f:
            grid = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    marian_lines = read_lines(args.marian_out, strip_bpe=args.strip_bpe)
    ref_lines = read_lines(args.reference, strip_bpe=args.strip_bpe) if args.reference else None

    rows = []
    configs = list(iter_configs(grid))
    print(f"running {len(configs)} configs", file=sys.stderr)

    for idx, cfg in enumerate(configs, 1):
        tag = config_tag(cfg)
        out_file = out_dir / f"hyp_{tag}.txt"
        print(f"[{idx}/{len(configs)}] {tag}", file=sys.stderr)

        if args.rescore_only:
            if not out_file.exists():
                status, err = False, "missing hypothesis file (rescore-only)"
            else:
                status, err = True, ""
        elif args.skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            status, err = True, ""
        else:
            status, err = run_one(args.binary, args.model, args.src, str(out_file),
                                   cfg, args.timeout)

        row = {"tag": tag, **cfg, "wer_vs_marian": "", "bleu_vs_ref": "", "error": ""}
        if not status:
            row["error"] = err.replace("\n", " ")[:300]
            rows.append(row)
            continue

        hyp_lines = read_lines(out_file, strip_bpe=args.strip_bpe)
        try:
            row["wer_vs_marian"] = f"{wer(hyp_lines, marian_lines):.4f}"
        except ValueError as e:
            row["error"] = str(e)

        if ref_lines is not None and HAVE_SACREBLEU and not row["error"]:
            if len(hyp_lines) != len(ref_lines):
                row["error"] = f"ref line count mismatch: hyp={len(hyp_lines)} ref={len(ref_lines)}"
            else:
                bleu = sacrebleu.corpus_bleu(hyp_lines, [ref_lines])
                row["bleu_vs_ref"] = f"{bleu.score:.2f}"

        rows.append(row)

    def sort_key(r):
        try:
            return float(r["wer_vs_marian"])
        except ValueError:
            return float("inf")

    rows.sort(key=sort_key)

    csv_path = out_dir / "results.csv"
    fieldnames = ["tag"] + list(grid.keys()) + ["wer_vs_marian", "bleu_vs_ref", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {csv_path}", file=sys.stderr)
    for r in rows[:10]:
        print(f"  WER={r['wer_vs_marian']:>6}  BLEU={r['bleu_vs_ref']:>5}  {r['tag']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
