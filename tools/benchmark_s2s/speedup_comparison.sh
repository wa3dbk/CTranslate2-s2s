#!/usr/bin/env bash
# Side-by-side speed benchmark: marian-decoder vs ct2-translator on the
# same converted s2s model. Reports wall-clock, output tokens/s, peak
# RSS, and optionally BLEU for three fixed thread configurations:
#
#   1. Marian baseline        --cpu-threads $CORES
#   2. CT2 apples-to-apples   --inter_threads $CORES --intra_threads 1
#   3. CT2 best               --inter_threads 1       --intra_threads $CORES
#
# A fourth row is added when WITH_INT8=1 (CT2 config 3 + --compute_type int8).
#
# This script bundles no model: bring your own Marian s2s npz plus the
# CT2-converted directory (see the "Marian export and inference" section
# of README.md for the conversion recipe).
#
# ---- Required environment ----
#   CT2_MODEL        Path to the CT2 model directory (output of ct2-marian-converter)
#   MARIAN_NPZ       Path to the original Marian .npz
#   MARIAN_YML       Path to the matching decoder .yml
#   VOCAB_SRC        Source-side Marian YAML vocab
#   VOCAB_TGT        Target-side Marian YAML vocab
#   SRC              Pre-tokenized source file (one sentence per line)
#
# ---- Optional ----
#   REF              Reference target file (triggers BLEU via sacrebleu).
#                    Both hyp files are stripped of subword-nmt "@@ " markers
#                    before scoring; if your model uses SentencePiece or a
#                    different tokenizer, score externally.
#   CORES            Thread budget (default: physical cores).
#                    SMT siblings should NOT be counted — hyperthreads
#                    sharing a BLAS FP unit hurt throughput.
#   BEAM             Beam size (default: 12).
#   LENGTH_PENALTY   Length normalization (default: 1.0; matches Marian --normalize 1.0).
#   MINI_BATCH       Marian --mini-batch and CT2 --max_batch_size (default: 32).
#   WITH_INT8        Set to 1 to add a CT2 int8 row.
#   DUMP_DIR         Output directory (default: /tmp/s2s_speedup).
#   CT2_BIN          Path to ct2-translator (default: build/cli/ct2-translator).
#   MARIAN_BIN       Path to marian-decoder (default: `which marian-decoder`).

set -euo pipefail

# ----- Required env check -----
: "${CT2_MODEL:?set CT2_MODEL to the converted model directory}"
: "${MARIAN_NPZ:?set MARIAN_NPZ to the Marian .npz file}"
: "${MARIAN_YML:?set MARIAN_YML to the matching decoder .yml}"
: "${VOCAB_SRC:?set VOCAB_SRC to the source Marian YAML vocab}"
: "${VOCAB_TGT:?set VOCAB_TGT to the target Marian YAML vocab}"
: "${SRC:?set SRC to a pre-tokenized source file (one sentence per line)}"

# ----- Defaults -----
: "${BEAM:=12}"
: "${LENGTH_PENALTY:=1.0}"
: "${MINI_BATCH:=32}"
: "${WITH_INT8:=0}"
: "${DUMP_DIR:=/tmp/s2s_speedup}"
: "${CT2_BIN:=./build/cli/ct2-translator}"
: "${MARIAN_BIN:=$(command -v marian-decoder || true)}"
: "${REF:=}"

if [[ -z "$MARIAN_BIN" || ! -x "$MARIAN_BIN" ]]; then
  echo "error: marian-decoder not found. Set MARIAN_BIN=/path/to/marian-decoder." >&2
  exit 1
fi
if [[ ! -x "$CT2_BIN" ]]; then
  echo "error: ct2-translator not found at $CT2_BIN. Build it or set CT2_BIN." >&2
  exit 1
fi

# ----- Core-count auto-detect (physical only; SMT siblings excluded) -----
detect_cores() {
  case "$(uname -s)" in
    Darwin)
      sysctl -n hw.physicalcpu
      ;;
    Linux)
      # One line per logical CPU; column 2 is physical-core-id. Unique those.
      lscpu -p=CORE 2>/dev/null | grep -Ev '^#' | sort -u | wc -l | tr -d ' '
      ;;
    *)
      getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
      ;;
  esac
}
: "${CORES:=$(detect_cores)}"
if ! [[ "$CORES" =~ ^[0-9]+$ ]] || [[ "$CORES" -lt 1 ]]; then
  echo "error: could not detect physical cores. Set CORES=N explicitly." >&2
  exit 1
fi

# ----- Env hygiene (prevent process-wide thread caps from fighting flags) -----
unset MKL_NUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS

mkdir -p "$DUMP_DIR"
NUM_LINES=$(wc -l < "$SRC" | tr -d ' ')

# ----- /usr/bin/time wrapper that normalizes macOS -l and Linux -v output -----
# Writes two files per run: <name>.hyp (translations) and <name>.time (metrics).
# Metrics file is key=value pairs: wall_seconds, peak_rss_mb.
time_run() {
  local name="$1"; shift
  local hyp="$DUMP_DIR/$name.hyp"
  local metrics="$DUMP_DIR/$name.time"
  local raw="$DUMP_DIR/$name.time.raw"

  # Warm-up: ignore timing but ensure disk cache + model load has happened.
  echo "    warm-up ($name)..."
  "$@" > /dev/null 2>&1 || true

  echo "    timed  ($name)..."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    /usr/bin/time -l "$@" > "$hyp" 2> "$raw"
    # macOS: "<N> maximum resident set size" in BYTES on recent macOS,
    # KILOBYTES on older. We assume bytes (10.13+) and divide by 1024^2.
    local rss_bytes
    rss_bytes=$(awk '/maximum resident set size/ {print $1}' "$raw")
    local wall
    wall=$(awk '/real/ {print $1}' "$raw")
    local rss_mb
    rss_mb=$(awk -v b="$rss_bytes" 'BEGIN {printf "%.1f", b/1048576}')
    printf 'wall_seconds=%s\npeak_rss_mb=%s\n' "$wall" "$rss_mb" > "$metrics"
  else
    /usr/bin/time -v "$@" > "$hyp" 2> "$raw"
    # Linux: "Elapsed (wall clock) time (h:mm:ss or m:ss)" + "Maximum resident set size (kbytes)".
    local wall_h
    wall_h=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "$raw")
    # Convert h:mm:ss.ss or m:ss.ss to seconds.
    local wall
    wall=$(awk -v t="$wall_h" 'BEGIN {
      n = split(t, p, ":"); s = 0;
      if (n == 3) s = p[1]*3600 + p[2]*60 + p[3];
      else if (n == 2) s = p[1]*60 + p[2];
      else s = p[1];
      printf "%.2f", s
    }')
    local rss_kb
    rss_kb=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$raw")
    local rss_mb
    rss_mb=$(awk -v k="$rss_kb" 'BEGIN {printf "%.1f", k/1024}')
    printf 'wall_seconds=%s\npeak_rss_mb=%s\n' "$wall" "$rss_mb" > "$metrics"
  fi
}

# ----- Count tokens in hyp file (whitespace-split, sum over all lines) -----
count_tokens() {
  awk '{ n += NF } END { print n+0 }' "$1"
}

# ----- Strip subword-nmt "@@ " markers (safe no-op if absent) -----
detok_bpe() {
  sed 's/@@ //g; s/@@$//' "$1"
}

# ----- BLEU via sacrebleu, operating on BPE-detokenized hypotheses -----
compute_bleu() {
  local hyp="$1"
  if [[ -z "$REF" ]]; then
    echo "n/a"
    return
  fi
  if ! command -v sacrebleu >/dev/null 2>&1; then
    echo "sacrebleu-not-installed"
    return
  fi
  local det="$DUMP_DIR/$(basename "$hyp").detok"
  detok_bpe "$hyp" > "$det"
  # --tokenize 13a is sacrebleu's default (mteval); --force skips length-match assert.
  sacrebleu "$REF" -i "$det" --score-only --force 2>/dev/null | awk '{printf "%.2f", $1}'
}

# ----- Reporting helpers -----
read_metrics() {
  # $1 = run name; sets globals WALL, RSS, TOKS, TPS, BLEU.
  WALL=$(awk -F= '/wall_seconds/{print $2}' "$DUMP_DIR/$1.time")
  RSS=$(awk -F= '/peak_rss_mb/{print $2}'  "$DUMP_DIR/$1.time")
  TOKS=$(count_tokens "$DUMP_DIR/$1.hyp")
  TPS=$(awk -v t="$TOKS" -v w="$WALL" 'BEGIN { if (w+0 > 0) printf "%.1f", t/w; else print "n/a" }')
  BLEU=$(compute_bleu "$DUMP_DIR/$1.hyp")
}

# ----- Pre-flight banner -----
cat <<EOF
========================================================================
s2s speed benchmark
  cores (physical)   : $CORES
  input              : $SRC ($NUM_LINES lines)
  beam / length pen. : $BEAM / $LENGTH_PENALTY
  mini-batch         : $MINI_BATCH
  dump dir           : $DUMP_DIR
  reference (BLEU)   : ${REF:-<not set, BLEU skipped>}
========================================================================
EOF

# =========================================================================
# Config 1: Marian baseline
# =========================================================================
echo "[1/3] Marian baseline (--cpu-threads $CORES)"
time_run marian \
  "$MARIAN_BIN" -c "$MARIAN_YML" -m "$MARIAN_NPZ" \
    -v "$VOCAB_SRC" "$VOCAB_TGT" \
    --cpu-threads "$CORES" \
    --mini-batch "$MINI_BATCH" --maxi-batch "$MINI_BATCH" --maxi-batch-sort src \
    --beam-size "$BEAM" --normalize "$LENGTH_PENALTY" \
    --quiet \
    -i "$SRC"

# =========================================================================
# Config 2: CT2 apples-to-apples (same core budget as Marian)
# =========================================================================
echo "[2/3] CT2 apples-to-apples (--inter_threads $CORES --intra_threads 1)"
time_run ct2_a2a \
  "$CT2_BIN" --model "$CT2_MODEL" --device cpu \
    --inter_threads "$CORES" --intra_threads 1 \
    --compute_type float32 \
    --batch_type examples --max_batch_size "$MINI_BATCH" \
    --beam_size "$BEAM" --length_penalty "$LENGTH_PENALTY" \
    --src "$SRC"

# =========================================================================
# Config 3: CT2 best (one batch, BLAS across all cores)
# =========================================================================
echo "[3/3] CT2 best (--inter_threads 1 --intra_threads $CORES)"
time_run ct2_best \
  "$CT2_BIN" --model "$CT2_MODEL" --device cpu \
    --inter_threads 1 --intra_threads "$CORES" \
    --compute_type float32 \
    --batch_type examples --max_batch_size "$MINI_BATCH" \
    --beam_size "$BEAM" --length_penalty "$LENGTH_PENALTY" \
    --src "$SRC"

# =========================================================================
# Config 4 (optional): CT2 int8
# =========================================================================
if [[ "$WITH_INT8" == "1" ]]; then
  echo "[4/4] CT2 int8 (--inter_threads 1 --intra_threads $CORES --compute_type int8)"
  time_run ct2_int8 \
    "$CT2_BIN" --model "$CT2_MODEL" --device cpu \
      --inter_threads 1 --intra_threads "$CORES" \
      --compute_type int8 \
      --batch_type examples --max_batch_size "$MINI_BATCH" \
      --beam_size "$BEAM" --length_penalty "$LENGTH_PENALTY" \
      --src "$SRC"
fi

# =========================================================================
# Report
# =========================================================================
echo
echo "========================================================================"
echo "Results (cores=$CORES, beam=$BEAM, length_penalty=$LENGTH_PENALTY, batch=$MINI_BATCH)"
echo "========================================================================"
echo
printf "| %-28s | %10s | %12s | %10s | %6s |\n" "Config" "Wall (s)" "Out tok/s" "RSS (MB)" "BLEU"
printf "|%s|%s|%s|%s|%s|\n" \
  "------------------------------" "------------" "--------------" "------------" "--------"

print_row() {
  local label="$1" name="$2"
  read_metrics "$name"
  printf "| %-28s | %10s | %12s | %10s | %6s |\n" "$label" "$WALL" "$TPS" "$RSS" "$BLEU"
}

print_row "Marian --cpu-threads=$CORES" marian
print_row "CT2 inter=$CORES intra=1 fp32" ct2_a2a
print_row "CT2 inter=1 intra=$CORES fp32" ct2_best
if [[ "$WITH_INT8" == "1" ]]; then
  print_row "CT2 inter=1 intra=$CORES int8" ct2_int8
fi

echo
echo "Hyps:    $DUMP_DIR/{marian,ct2_a2a,ct2_best}.hyp"
echo "Metrics: $DUMP_DIR/{marian,ct2_a2a,ct2_best}.time"
