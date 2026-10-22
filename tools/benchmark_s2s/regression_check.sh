#!/usr/bin/env bash
# Regression gate for s2s optimizations.
#
# Translates a fixed source file with both greedy and beam decoding, records
# exact output hashes + optional BLEU, and either writes a baseline manifest
# or diffs the current run against one.
#
# Usage:
#   # Record a baseline before touching the code.
#   regression_check.sh record BASELINE_LABEL
#
#   # After each code change, compare.
#   regression_check.sh check BASELINE_LABEL CANDIDATE_LABEL
#
# Environment:
#   CT2_MODEL   Path to converted CT2 s2s model dir           (required)
#   SRC         Pre-tokenized source file, one sentence/line  (required)
#   REF         Reference target file (optional; enables BLEU)
#   BEAM        Beam size for the beam row (default: 12)
#   DUMP_DIR    Where to write manifests + hyps (default: /tmp/s2s_regression)
#   CT2_BIN     Path to ct2-translator (default: build/cli/ct2-translator)
#
# Pass criterion applied by `check`:
#   1. Exact output match on BOTH rows (greedy + beam) → PASS
#   2. If outputs differ: |BLEU_candidate - BLEU_baseline| <= BLEU_TOLERANCE
#      (default 0.2; set via env). Prints a diff summary.
#   3. Otherwise FAIL (non-zero exit).
#
# The exact-match path catches silent output changes from "numerical no-op"
# optimizations. The BLEU-tolerance path exists for Opt 4 (gate fusion)
# where fp32 operation reordering can perturb token ids without hurting
# quality.

set -euo pipefail

: "${CT2_MODEL:?set CT2_MODEL to the converted model directory}"
: "${SRC:?set SRC to a pre-tokenized source file}"
: "${BEAM:=12}"
: "${DUMP_DIR:=/tmp/s2s_regression}"
: "${CT2_BIN:=./build/cli/ct2-translator}"
: "${REF:=}"
: "${BLEU_TOLERANCE:=0.2}"

if [[ ! -x "$CT2_BIN" ]]; then
  echo "error: ct2-translator not found at $CT2_BIN. Build it or set CT2_BIN." >&2
  exit 1
fi

mkdir -p "$DUMP_DIR"

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

detok_bpe() { sed 's/@@ //g; s/@@$//' "$1"; }

compute_bleu() {
  local hyp="$1"
  if [[ -z "$REF" ]] || ! command -v sacrebleu >/dev/null 2>&1; then
    echo "n/a"; return
  fi
  local det="$DUMP_DIR/$(basename "$hyp").detok"
  detok_bpe "$hyp" > "$det"
  sacrebleu "$REF" -i "$det" --score-only --force 2>/dev/null | awk '{printf "%.2f", $1}'
}

# $1 = label, $2 = beam size
run_decode() {
  local label="$1"
  local beam="$2"
  local hyp="$DUMP_DIR/$label.hyp"
  "$CT2_BIN" --model "$CT2_MODEL" --device cpu \
    --inter_threads 1 --intra_threads 1 \
    --compute_type float32 \
    --batch_type examples --max_batch_size 32 \
    --beam_size "$beam" --length_penalty 1.0 \
    --src "$SRC" > "$hyp"
  echo "$hyp"
}

# $1 = run label → writes $DUMP_DIR/$1.manifest
write_manifest() {
  local label="$1"
  local greedy_hyp beam_hyp
  echo "  greedy decode..."
  greedy_hyp=$(run_decode "${label}_greedy" 1)
  echo "  beam=$BEAM decode..."
  beam_hyp=$(run_decode "${label}_beam"   "$BEAM")

  local greedy_sha beam_sha greedy_bleu beam_bleu
  greedy_sha=$(sha_of "$greedy_hyp")
  beam_sha=$(sha_of "$beam_hyp")
  greedy_bleu=$(compute_bleu "$greedy_hyp")
  beam_bleu=$(compute_bleu   "$beam_hyp")

  local manifest="$DUMP_DIR/$label.manifest"
  {
    echo "label=$label"
    echo "model=$CT2_MODEL"
    echo "src=$SRC"
    echo "beam=$BEAM"
    echo "greedy_hyp=$greedy_hyp"
    echo "greedy_sha=$greedy_sha"
    echo "greedy_bleu=$greedy_bleu"
    echo "beam_hyp=$beam_hyp"
    echo "beam_sha=$beam_sha"
    echo "beam_bleu=$beam_bleu"
  } > "$manifest"

  echo "manifest: $manifest"
  echo "  greedy sha=$greedy_sha  bleu=$greedy_bleu"
  echo "  beam   sha=$beam_sha  bleu=$beam_bleu"
}

load_field() {
  awk -F= -v k="$2" '$1 == k { sub(/^[^=]*=/, ""); print; exit }' "$1"
}

cmd_record() {
  local label="${1:?record requires a label}"
  echo "[record] $label"
  write_manifest "$label"
}

cmd_check() {
  local base_label="${1:?check requires BASELINE_LABEL}"
  local cand_label="${2:?check requires CANDIDATE_LABEL}"
  local base="$DUMP_DIR/$base_label.manifest"
  local cand="$DUMP_DIR/$cand_label.manifest"
  if [[ ! -f "$base" ]]; then
    echo "error: baseline manifest not found: $base" >&2; exit 2
  fi

  echo "[check] candidate=$cand_label vs baseline=$base_label"
  write_manifest "$cand_label"

  local bg_sha bb_sha cg_sha cb_sha
  bg_sha=$(load_field "$base" greedy_sha)
  bb_sha=$(load_field "$base" beam_sha)
  cg_sha=$(load_field "$cand" greedy_sha)
  cb_sha=$(load_field "$cand" beam_sha)

  if [[ "$bg_sha" == "$cg_sha" && "$bb_sha" == "$cb_sha" ]]; then
    echo "PASS (exact match on greedy + beam)"
    exit 0
  fi

  echo "outputs differ — falling back to BLEU tolerance (<= $BLEU_TOLERANCE)"
  local bg_bl bb_bl cg_bl cb_bl
  bg_bl=$(load_field "$base" greedy_bleu)
  bb_bl=$(load_field "$base" beam_bleu)
  cg_bl=$(load_field "$cand" greedy_bleu)
  cb_bl=$(load_field "$cand" beam_bleu)

  if [[ "$bg_bl" == "n/a" || "$cg_bl" == "n/a" ]]; then
    echo "FAIL: outputs differ and REF not set — cannot evaluate BLEU tolerance"
    echo "  baseline greedy_hyp=$(load_field "$base" greedy_hyp)"
    echo "  candidate greedy_hyp=$(load_field "$cand" greedy_hyp)"
    exit 1
  fi

  local greedy_delta beam_delta ok=1
  greedy_delta=$(awk -v a="$cg_bl" -v b="$bg_bl" 'BEGIN { d=a-b; if (d<0) d=-d; printf "%.3f", d }')
  beam_delta=$(awk   -v a="$cb_bl" -v b="$bb_bl" 'BEGIN { d=a-b; if (d<0) d=-d; printf "%.3f", d }')

  awk -v d="$greedy_delta" -v t="$BLEU_TOLERANCE" 'BEGIN { exit !(d+0 <= t+0) }' || ok=0
  awk -v d="$beam_delta"   -v t="$BLEU_TOLERANCE" 'BEGIN { exit !(d+0 <= t+0) }' || ok=0

  echo "  greedy: $bg_bl -> $cg_bl  (|Δ|=$greedy_delta)"
  echo "  beam  : $bb_bl -> $cb_bl  (|Δ|=$beam_delta)"
  if [[ "$ok" == "1" ]]; then
    echo "PASS (within BLEU tolerance)"
    exit 0
  fi
  echo "FAIL (BLEU drift exceeds tolerance)"
  exit 1
}

mode="${1:?usage: $0 {record|check} ...}"; shift
case "$mode" in
  record) cmd_record "$@" ;;
  check)  cmd_check  "$@" ;;
  *) echo "usage: $0 {record LABEL | check BASELINE_LABEL CANDIDATE_LABEL}" >&2; exit 2 ;;
esac
