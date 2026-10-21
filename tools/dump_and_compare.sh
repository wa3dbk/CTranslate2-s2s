#!/usr/bin/env bash
# One-shot dump of all parity probes on both sides + full comparison report.
#
# Prereqs (one rebuild each):
#   - CT2 built with the dump_row0_once / dump_htilde_once / dump_logits_once
#     / dump_bias_once hooks in src/layers/rnn.cc.
#   - marian built with the dumpOneShot hooks in beam_search.cpp and the
#     g_lastMlp* / g_lastEncoderMemory / g_lastGru1State / g_lastAttnAlignment
#     globals wired up.
#
# After this script, iterate over the printed reports — no further rebuilds
# needed unless you change the dump surface.
set -euo pipefail

: "${CT2_BIN:=./build/cli/ct2-translator}"
: "${MARIAN_BIN:=marian-decoder}"
: "${CT2_MODEL:?set CT2_MODEL to the converted model directory}"
: "${MARIAN_NPZ:?set MARIAN_NPZ to the Marian .npz file}"
: "${MARIAN_YML:?set MARIAN_YML to the matching decoder .yml}"
: "${VOCAB_SRC:?set VOCAB_SRC to the source Marian YAML vocab}"
: "${VOCAB_TGT:?set VOCAB_TGT to the target Marian YAML vocab}"
: "${SRC:?set SRC to a single-line BPE source file}"

DUMP_DIR="${DUMP_DIR:-/tmp/s2s_parity}"
STEP="${STEP:-0}"
UNK_ID="${UNK_ID:-1}"
EOS_ID="${EOS_ID:-0}"

mkdir -p "$DUMP_DIR"
cd "$(dirname "$0")/.."

echo "=== CT2 decode (step=$STEP dumps -> $DUMP_DIR/ct2.*.bin) ==="
CT2_S2S_DUMP_EMB="$DUMP_DIR/ct2.emb.bin" \
CT2_S2S_DUMP_STATE="$DUMP_DIR/ct2.state.bin" \
CT2_S2S_DUMP_CTX="$DUMP_DIR/ct2.ctx.bin" \
CT2_S2S_DUMP_MEMORY="$DUMP_DIR/ct2.memory.bin" \
CT2_S2S_DUMP_S1="$DUMP_DIR/ct2.s1.bin" \
CT2_S2S_DUMP_ALIGNMENT="$DUMP_DIR/ct2.alignment.bin" \
CT2_S2S_DUMP_HTILDE="$DUMP_DIR/ct2.htilde.bin" \
CT2_S2S_DUMP_LOGITS="$DUMP_DIR/ct2.logits.bin" \
CT2_S2S_DUMP_BIAS="$DUMP_DIR/ct2.bias.bin" \
"$CT2_BIN" --model "$CT2_MODEL" --device cpu \
  --src "$SRC" --out "$DUMP_DIR/ct2.out" \
  --beam_size 12 --disable_unk >/dev/null

echo "=== Marian decode (step=$STEP dumps -> $DUMP_DIR/marian.*.bin) ==="
MARIAN_DUMP_STEP="$STEP" \
MARIAN_DUMP_EMB="$DUMP_DIR/marian.emb.bin" \
MARIAN_DUMP_STATE="$DUMP_DIR/marian.state.bin" \
MARIAN_DUMP_CTX="$DUMP_DIR/marian.ctx.bin" \
MARIAN_DUMP_MEMORY="$DUMP_DIR/marian.memory.bin" \
MARIAN_DUMP_S1="$DUMP_DIR/marian.s1.bin" \
MARIAN_DUMP_ALIGNMENT="$DUMP_DIR/marian.alignment.bin" \
MARIAN_DUMP_HTILDE="$DUMP_DIR/marian.htilde.bin" \
MARIAN_DUMP_LOGITS="$DUMP_DIR/marian.logits.bin" \
"$MARIAN_BIN" -c "$MARIAN_YML" -m "$MARIAN_NPZ" \
  -v "$VOCAB_SRC" "$VOCAB_TGT" \
  --cpu-threads 1 --mini-batch 1 --maxi-batch 1 \
  --beam-size 12 --quiet \
  < "$SRC" > "$DUMP_DIR/marian.out"

echo
echo "=== forward-pass cosine report ==="
python3 tools/compare_inputs.py --dir "$DUMP_DIR"

echo
echo "=== logits + bias report ==="
python3 tools/compare_logits.py \
  --ct2    "$DUMP_DIR/ct2.logits.bin" \
  --marian "$DUMP_DIR/marian.logits.bin" \
  --vocab  "$VOCAB_TGT" \
  --unk-id "$UNK_ID" --eos-id "$EOS_ID" \
  --marian-npz   "$MARIAN_NPZ" \
  --ct2-bias-bin "$DUMP_DIR/ct2.bias.bin"
