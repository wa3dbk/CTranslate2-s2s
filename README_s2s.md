# CTranslate2 — MarianNMT `s2s` Model Support (Design Spec)

**Status:** Design — approved, implementation pending
**Author:** collaborative brainstorming (wa3dbk + Claude Opus 4.7)
**Date:** 2026-10-19
**Scope decision:** approach 1 (dedicated model type), scope "C" (narrow + explicit guards), GRU-only, CPU float32 + int8.

This document is the long-form technical reference for adding support for Marian's RNN-based seq2seq (`type: s2s`) models to CTranslate2. It is maintained alongside the implementation and should always reflect the current state of the port.

---

## 1. Why this exists

CTranslate2 today supports Marian **transformer** encoder-decoder models only (`python/ctranslate2/converters/marian.py` rejects everything else at the config-type check). Marian's older but still widely deployed RNN-based `s2s` models — single-layer bi-GRU encoder plus conditional-GRU decoder with Bahdanau attention, the architecture used by most pre-transformer WMT and OPUS releases — have no path into CT2. This project adds that path for one concrete target configuration, with explicit guards that fail loudly on anything outside that configuration so the port can be extended safely later.

## 2. Target model profile

The implementation is validated against a specific real-world model with the following `model.npz.yml`:

| Key | Value |
|---|---|
| `type` | `s2s` |
| `dim-vocabs` | `[5001, 5001]` (shared) |
| `dim-emb` | `1024` |
| `dim-rnn` | `2024` |
| `enc-type` | `bidirectional` |
| `enc-cell` | `gru` |
| `enc-cell-depth` | `1` |
| `enc-depth` | `1` |
| `dec-cell` | `gru` |
| `dec-cell-base-depth` | `2` (⇒ conditional GRU) |
| `dec-cell-high-depth` | `1` |
| `dec-depth` | `1` |
| `skip` | `False` |
| `layer-normalization` | `True` |
| `right-left` | `False` |
| `tied-embeddings-all` | `True` |
| `output-omit-bias` | `False` |
| `transformer-postprocess-emb` | `d` (dropout only; no-op at inference) |
| decoder beam | `beam-size: 12`, `normalize: 1` |

Decoding-time configuration in `model.npz.decoder.yml` reduces to: `beam-size: 12, normalize: 1, word-penalty: 0`.

## 3. Scope decisions and their rationale

### 3.1 Model-family scope (decided)

**Supported:** Marian `type: s2s` only.
**Rejected (with clear errors):** `transformer` is handled by the existing branch; `nematus`, `char_s2s`, `amun`, `bert`, `laser`, `classifier`, `encoder-pooler` are not supported by this converter.

### 3.2 Cell-type scope (decided)

**Supported:** GRU (`enc-cell: gru`, `dec-cell: gru`).
**Rejected:** `lstm`, `gru-nematus`, `tanh`, `relu`, `ssru`.

Rationale: released Marian `s2s` checkpoints are overwhelmingly GRU. LSTM support is a mechanical extension (new cell class, new spec entry) deferrable until a real model needs it.

### 3.3 Generalisation scope (decided — "C")

**Narrow + explicit guards.** The converter supports exactly the architectural shape of the target model. Every other combination raises `ValueError` at convert time, naming the offending option and expected value, with a "file an issue" pointer. No silent misbehaviour at inference.

Rationale: a flexible superset (Approach "B" in brainstorming) would ≈ double the converter and spec work and add test matrix for cases nobody has yet asked for. Narrow + guards delivers the user's actual model today, loud-fails any surprise tomorrow, and leaves clean seams for widening.

### 3.4 Platform scope (decided)

**Phase 1 ships:** CPU float32 and CPU int8.
**Phase 2 (future, not in this spec):** CUDA fp16/fp32. Design keeps the door open — every big matmul in the new layers goes through CT2's existing device-agnostic `Dense` + `ops::` primitives, so no C++ layer code needs to be rewritten to add CUDA; only build-system wiring and testing.
**Not in scope:** cuDNN-fused RNN kernels.

### 3.5 Format scope (decided — Q1)

**Supported input format:** Marian NPZ (`.npz`) + separate vocab `.yml` files. Marian `.bin` (memory-mappable) is **not** supported in phase 1; adding it is additive — same spec, different loader.

### 3.6 Feature scope (decided — Q2, Q3)

- **Alignment extraction** (`TranslationOptions.return_alignment`): **not** in phase 1. Attention produces alignments internally (needed to compute context), but we discard them. Future hook is ~30 lines in `RNNDecoder`.
- **Shortlist / vocabulary mapping** (`use_vmap`): **not** in phase 1. Target model doesn't use it.
- **Right-to-left decoding** (`right-left`): **not** in phase 1.
- **Factored vocab / lemma embeddings**: **not** in phase 1.
- **Guided alignment** (training-time only in Marian): **not applicable** at inference.

## 4. Architecture

Three layers, matching the structure CT2 uses for Transformer.

```
┌─────────────────────────── Python conversion ─────────────────────────────┐
│                                                                           │
│  model.npz + *.yml                                                        │
│       │                                                                   │
│       ▼                                                                   │
│  converters/marian.py (new `s2s` branch)                                  │
│       │                                                                   │
│       ▼                                                                   │
│  specs/rnn_spec.py  (RNNModelSpec)                                        │
│       │                                                                   │
│       ▼                                                                   │
│  <model_dir>/model.bin                                                    │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────── C++ runtime load ──────────────────────────────┐
│                                                                           │
│  model.bin                                                                │
│       │                                                                   │
│       ▼                                                                   │
│  models/model_factory.cc  (new "RNNModelSpec" entry)                      │
│       │                                                                   │
│       ▼                                                                   │
│  RNNModel  (new, src/models/rnn.{h,cc})                                   │
│       │                                                                   │
│       ├── BiGRUEncoder   (src/layers/rnn.{h,cc})                          │
│       └── RNNDecoder     (src/layers/rnn.{h,cc})                          │
│             │                                                             │
│             ├── init_state_proj (Dense + LN + tanh)                       │
│             ├── CGRUDecoderCell                                           │
│             │     ├── GRUCell gru1                                        │
│             │     ├── BahdanauAttention attn                              │
│             │     └── GRUCell gru2 (final=true)                           │
│             └── DeepOutput (hidden Dense + tied-transposed output)        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────── Inference call path (unchanged) ───────────────────┐
│                                                                           │
│  Translator::translate_batch()                                            │
│       │                                                                   │
│       ▼                                                                   │
│  SequenceToSequenceReplica       (existing, generic)                      │
│       │                                                                   │
│       ├── RNNModel::encode()      → memory, memory_lengths                │
│       └── decoding::beam_search() → calls RNNDecoder::operator()/step     │
│             (existing beam-search infra, no changes)                      │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** `SequenceToSequenceReplica`, `decoding::beam_search`, batching, length normalisation, prefix constraints, and the `Translator` Python API are all architecture-agnostic. The new code lives entirely inside `RNNModel` and its two layers. We do not touch beam search, batching, or the public API.

## 5. Weight mapping (Marian NPZ → CT2 spec)

Marian uses the plain `GRU` class (gamma1/gamma2 scales) when `enc-cell: gru` / `dec-cell: gru` with `layer-normalization: True`. This is distinct from the `GRUNematus` class (`_lns`/`_lnb` affine LN), which this converter does **not** handle.

### 5.1 Prefix inventory (names present in the NPZ)

| Role | Marian prefix |
|---|---|
| Encoder forward (1 layer, 1 cell) | `encoder_bi` |
| Encoder backward | `encoder_bi_r` |
| Decoder base cell 1 | `decoder_cell1` |
| Decoder base cell 2 (`final: true`) | `decoder_cell2` |
| Attention (single encoder) | `decoder` |
| Shared embedding | `Wemb` |
| Decoder init MLP | `decoder_ff_state_*` |
| Deep output hidden | `decoder_ff_logit_l1_*` |
| Deep output top (tied-transposed to `Wemb`) | `decoder_ff_logit_l2_*` |

### 5.2 Per-GRU-cell tensors

Reference: `marian/src/rnn/cells.h:228-266`.

| Marian name | Shape | Fate in CT2 spec |
|---|---|---|
| `{p}_U`, `{p}_Ux` | `[H, 2H]`, `[H, H]` | concat on last axis → `cell.recurrent_weight` `[H, 3H]` |
| `{p}_W`, `{p}_Wx` | `[I, 2H]`, `[I, H]` | concat → `cell.input_weight` `[I, 3H]` (absent for transition-only cells) |
| `{p}_b`, `{p}_bx` | `[1, 2H]`, `[1, H]` | concat → `cell.bias` `[1, 3H]` |
| `{p}_gamma1` | `[1, 3H]` | `cell.ln_input_scale` (only if dimInput > 0 & LN) |
| `{p}_gamma2` | `[1, 3H]` | `cell.ln_recurrent_scale` |

**Gate ordering invariant:** Marian's `gruOps` kernel consumes concatenated slices as `[reset_gate | update_gate | candidate]`. The converter preserves this order; the CT2 cell applies `σ·σ·tanh` on the matching column slices after bias add. This ordering is asserted in the Tier-1 weight-mapping unit test (see §9).

**`final=true` on `decoder_cell2`:** `cells.h:313-314` calls `gruOps(..., final=true)` for the second cGRU step. The semantics differ from the non-final branch; the CT2 `GRUCell` has a `final_` flag that replicates the exact `gruOps` behaviour (verified by direct reading of `tensor_operators.cpp:gruOps` during implementation).

### 5.3 Attention tensors

Reference: `marian/src/rnn/attention.h:53-91` (non-nematus LN branch, which is what our target model uses).

| Marian name | Shape | CT2 spec attr |
|---|---|---|
| `decoder_W_comb_att` | `[Hdec, Henc]` | `attn.query_weight` |
| `decoder_Wc_att` | `[Henc, Henc]` | `attn.memory_weight` |
| `decoder_U_att` | `[Henc, 1]` | `attn.v` |
| `decoder_b_att` | `[1, Henc]` | `attn.ln_context_bias` (LN shift, **not** affine bias) |
| `decoder_att_gamma1` | `[1, Henc]` | `attn.ln_context_scale` |
| `decoder_att_gamma2` | `[1, Henc]` | `attn.ln_state_scale` |

Marian's attention formula with LN:
```
mapped_context = LayerNorm(context @ Wc_att, γ_context, b_att)   # pre-computed once per encode
mapped_state   = LayerNorm(state   @ Wa,     γ_state)            # per step
e              = v^T · tanh(mapped_state + mapped_context)
α              = softmax(e, mask)                                 # over source positions
context_vector = Σᵢ αᵢ · memoryᵢ                                  # weighted sum of raw memory
```

The attribute name `ln_context_bias` is deliberately explicit about `b_att`'s role as LN shift, not a pre-LN affine bias — a known foot-gun.

### 5.4 Decoder-init MLP, deep-output head, embeddings

- **`decoder_ff_state_*`**: a `Dense(2·dim-rnn → dim-rnn)` with LN and tanh. Input = source-mask-weighted mean of encoder memory. Output = initial decoder hidden state.
- **`decoder_ff_logit_l1_*`**: a `Dense((dim-emb + dim-rnn + 2·dim-rnn) → dim-emb)` with LN and tanh. Input = concat of `[prev_emb, dec_state, context]`.
- **`decoder_ff_logit_l2_*`**: tied-transposed to `Wemb` (`tied-embeddings-all: True`). Only the bias is a stored tensor; the weight is a view onto `Wemb.T`.
- **`Wemb`** `[vocab, dim-emb]`: shared across source embedding, target embedding, and output projection. Converter asserts both source and target vocab files are identical.

### 5.5 Pre-compute at runtime (not at convert time)

The quantity `mapped_context = LayerNorm(memory @ Wc_att, γ_context, b_att)` is identical for every decoder step but depends on live encoder output, so it is computed once by `RNNModel::encode()` (via `BahdanauAttention::precompute()`), stored on the decoder state, and reused for the entire decode loop.

## 6. C++ layer breakdown

New code: two files only — `src/layers/rnn.{h,cc}` for reusable cell/attention/encoder/decoder, `src/models/rnn.{h,cc}` for the model class.

### `layers::GRUCell` — one Marian-style GRU step with LN

- **Members:** `Dense input_dense` (fused `[W | Wx]` + bias, 3H outputs), `Dense recurrent_dense` (fused `[U | Ux]`), optional `LayerNorm ln_input`, optional `LayerNorm ln_recurrent`, flag `final_`.
- **Signature:** `void operator()(const StorageView& input, const StorageView& prev_state, StorageView& new_state) const;`
- **Input optional:** supported but unused in this phase (both base cells always have input).
- **Int8:** the two `Dense` members are standard CT2 `Dense` → quantization paths work unchanged.

### `layers::BiGRUEncoder` — the whole encoder

- **Members:** `Embeddings embeddings`, `GRUCell fwd`, `GRUCell bwd`.
- **Signature:** `void operator()(const StorageView& ids, const StorageView& lengths, StorageView& memory, StorageView& memory_lengths);`
- **Internals:** sequence loop fwd `0..T`, bwd `T..0`; concatenates per-step hidden states on the last axis → `memory` shape `[B, T, 2H]`. Padded steps zeroed via existing `ops::Mul`.

### `layers::BahdanauAttention` — MLP additive attention with LN

- **Members:** `Dense query_proj` (Wa), `Dense memory_proj` (Wc_att + b_att), `LayerNorm ln_context`, `LayerNorm ln_state`, `StorageView v`.
- **Precompute API:** `precompute(const StorageView& memory, StorageView& projected_memory)`.
- **Step API:** `compute(query, projected_memory, memory, memory_mask, context, alignment)`.

### `layers::CGRUDecoderCell` — one decoder step

- **Members:** `GRUCell gru1`, `BahdanauAttention attn`, `GRUCell gru2` (with `final_=true`).
- **Signature:** `void operator()(input_emb, prev_state, projected_memory, memory, memory_mask, new_state, context, alignment);`
- **Flow:** `s1 = gru1(input_emb, prev_state)` → `(context, alignment) = attn(s1, projected_memory, memory, memory_mask)` → `new_state = gru2(context, s1)`. `new_state` carries forward; `context` and `input_emb` feed `DeepOutput`.

### `layers::DeepOutput` — Marian's two-layer logit head

- **Members:** `Dense hidden` (concat input → dim_emb, tanh, LN), `Dense tied_output` (weight = `embeddings.weight.transpose()`, own bias).
- **Signature:** `void operator()(emb, dec_state, context, logits);`

### `layers::RNNDecoder` — implements CT2's `Decoder` interface

- **Members:** `Embeddings embeddings`, `Dense init_state_proj` (ff_state), `CGRUDecoderCell cell`, `DeepOutput output`.
- **Overrides:** `set_state`, `operator()` (step), `get_vocabulary_size`, `output_type`.

### `models::RNNModel` — top-level model class

- Owns `BiGRUEncoder` + `RNNDecoder`. Implements `SequenceToSequenceModel`'s abstract hooks (`get_encoder`, `get_decoder`, `make_sequence_to_sequence_replica`).
- Registered in `src/models/model_factory.cc` under spec revision 1 with name `"RNNModelSpec"`.

### Explicit non-additions

- No fused custom GRU kernel (CPU or GPU).
- No new quantisation code — comes from `Dense`.
- No change to beam search, batching, length normalisation, or `Translator` API.

## 7. Decoder state & beam-search integration

### 7.1 Tensors carried in `DecoderState`

| Key | Shape | Gather? | Lifecycle |
|---|---|---|---|
| `"hidden"` | `[B·K, H]` | **yes** — beam-distinct | Written each step by `CGRUDecoderCell`; initialised by `init_state_proj(mean_pooled_memory)` |
| `"memory"` | `[B·K, S, 2H]` | yes — beam-replicated | Set once by `encode()`, tiled to `K` by `Decoder::replicate_state` |
| `"memory_lengths"` | `[B·K]` | yes — beam-replicated | Set once, tiled to `K` |
| `"projected_memory"` | `[B·K, S, Henc]` | yes — beam-replicated | Set once by `attn.precompute()` from the already-tiled `memory` |

### 7.2 Batch-vs-beam distinction

Historically the RNN decoder kept the memory-side tensors (`memory`, `memory_lengths`, `projected_memory`) batch-only (`[B, …]`) to save `K×` RAM, and the attention `compute()` tiled them to `[B·K, …]` on every step. That saved memory at the cost of three `repeat_batch` calls in the hot path, which measurable profiling showed were a non-trivial fraction of per-step cost.

Opt 2 of the s2s perf series flipped this: every state entry is now beam-replicated. The `Decoder` base class auto-tiles non-empty state via `repeat_batch(value, beam_size)` once before step 0, and per-step finished-hypothesis pruning gathers them by `beam_indices` through `Decoder::update_state`. Attention `compute()` sees matching batch sizes and does no per-step tiling.

- Memory cost: `K×` more RAM for `memory` + `projected_memory` + `memory_lengths`. For a typical batch=32, beam=12, S=20, 2H=1024 Marian model this is ~62 MiB of extra `memory` (vs ~5 MiB before), negligible compared to the ~3 GiB CT2 working-set.
- Correctness: covered by `RNNDecoderReplicateStateReplicatesAllEntriesAcrossBeams` in `tests/rnn_test.cc` — asserts the four state entries are all beam-replicated.

### 7.3 `set_state(DecoderState&)` — once per `translate_batch`

```
init = tanh(LN(mean_pooled(memory, mask) @ W_ff_state + b_ff_state, γ_ff_state))
state["hidden"]            = init                        # [B, H], expanded to [B·K, H] by BeamSearch
state["memory"]            = encoder_memory              # [B, S, 2H]
state["memory_lengths"]    = source_lengths              # [B]
state["projected_memory"]  = attn.precompute(memory)     # [B, S, Henc]
```

Mean-pool is weighted by source mask (matches `s2s.h:220-224`'s `weighted_average(context, mask, axis=-3)`). Debug-build asserts mask non-null.

### 7.4 `operator()(step, ids, state, logits)` — hot path

```
emb = embeddings(ids)
(new_hidden, context, alignment) = cell(emb,
                                        state["hidden"],
                                        state["projected_memory"],
                                        state["memory"],
                                        state["memory_lengths"])
logits = deep_output(emb, new_hidden, context)
state["hidden"] = new_hidden        # in-place
```

**Memory footprint:** O(B·K·H) regardless of step — notably better than Transformer's O(B·K·T·H) KV cache for very long outputs.

### 7.5 `update_state(gather_indices)`

Default `DecoderState` behaviour (gather every registered entry on axis 0) suffices — no override needed.

### 7.6 Decoding knobs

- `beam-size: 12` → `beam_size=12`.
- `normalize: 1` → `length_penalty=1.0`. Existing CT2 knob; already handled in `marian.py`'s transformer branch; reused.
- `word-penalty: 0` → default, no-op.

### 7.7 Explicitly out of scope

- No batch-pruning rebuild (`s2s.h:261-263`); CT2's beam search operates on fixed `batch×beam`.
- No alignment surfacing (see §3.6).

## 8. Converter guards & error handling

Entry point in `python/ctranslate2/converters/marian.py`:

```python
model_type = config["type"]
if model_type == "transformer":
    spec = _build_transformer_spec(config, model, vocabs)
elif model_type == "s2s":
    spec = _build_s2s_spec(config, model, vocabs)
else:
    raise NotImplementedError(
        f"Marian model type '{model_type}' is not supported; "
        f"supported types: 'transformer', 's2s'"
    )
```

### 8.1 Guard function

```
ValueError: Marian s2s converter: option 'enc-depth' = 2 is not supported
            (this converter currently requires 'enc-depth' = 1).
            File an issue with your model.npz.yml if you need this variant.
```

### 8.2 Guard list (hard constraints)

Architecture shape:
`enc-depth == 1`, `enc-cell-depth == 1`, `enc-type == "bidirectional"`, `enc-cell == "gru"`, `dec-depth == 1`, `dec-cell == "gru"`, `dec-cell-base-depth == 2`, `dec-cell-high-depth == 1`, `skip == False`, `layer-normalization == True`, `tied-embeddings-all == True`.

Features not implemented (user models may set these; we'd silently drop them):
`right-left == False`, `input-types == []`, `factors-dim-emb == 0`, `lemma-dim-emb == 0`, `factors-combine in {"", "sum"}`, `transformer-rnn-projection == False`, `transformer-pool == False`, `ulr == False`, `embedding-fix-src == False`, `embedding-fix-trg == False`.

### 8.3 Ignored silently

All training-only knobs (dropout, cost, optimiser, lr, clip, label-smoothing, exponential-smoothing, early-stopping, valid-*, train-*, data-*, save-*, log-*, seed, mini-batch*, maxi-batch*, shuffle*, after*, disp-*, quantize-*, check-*, precision, cost-scaling, gradient-*, sync-*, devices, num-devices, cpu-threads, workspace, sentencepiece-*, multi-loss-type, unlikelihood-loss, overwrite, no-reload, tempdir, sqlite*, pretrained-model, ignore-model-config), and all `bert-*` keys.

### 8.4 Flagged with WARN but not fail

`output-omit-bias == True` (both supported; just note); residual `transformer-*` keys with non-default values when `type == s2s`.

### 8.5 Weight-tensor validation

- Assert every expected tensor present (missing → error naming the tensor and prefix).
- Assert every NPZ tensor consumed except `special:model.yml` and `Wemb` (consumed twice: embedding + tied output). Unused tensors → error listing them.
- Assert shape invariants: `Wemb.shape == (vocab_size, dim_emb)`, encoder `_U.shape[0] == dim_rnn`, `decoder_cell1_W.shape[0] == dim_emb`, `decoder_cell2_W.shape[0] == 2 * dim_rnn`.

### 8.6 Vocabulary

Marian vocab is YAML (token → id). Existing `_load_vocab()` reused unchanged. With `tied-embeddings-all: True`, both vocab files must be identical; converter asserts and emits a single shared vocab in CT2's JSON format.

### 8.7 Converter self-smoke-test

After writing the model directory, instantiate a `Translator` on it with `compute_type="int8"` and run `translate_batch([["hello"]])`. Catches ABI mismatches between spec declarations and C++ loader. Failure prints model path and the exact `RuntimeError`.

## 9. Testing strategy

Three tiers, extending existing `tests/` (C++) and `python/tests/` (pytest).

### 9.1 Tier 1 — Python converter unit tests

File: `python/tests/test_marian_s2s_converter.py`.

- **Guard tests** (one parametrised test per guarded key, ~15 cases): mutate one key, assert `ValueError` with the key name.
- **Shape-invariant tests:** synthetic NPZ with wrong tensor shape → assert error names the tensor.
- **Weight-mapping test:** synthetic NPZ with distinct constants per slice (`_U = full(1.0)`, `_Ux = full(2.0)`, …); after conversion assert concat order in `cell.recurrent_weight`. **This is the single most important test** — catches gate-ordering regressions, which are the top silent-bug class in RNN ports.
- **Unused-tensor test:** add an extra tensor to a synthetic NPZ; assert converter fails naming it.

### 9.2 Tier 2 — C++ layer tests

File: extend `tests/layers_test.cc`.

Reference `.npz` files generated by `tests/data/rnn/generate_reference.py` (committed), containing NumPy-computed expected outputs for the exact Marian formulas.

- `GRUCellTest.ForwardMatchesReference` — LN on/off, with/without input, `final=true` and `final=false`.
- `BahdanauAttentionTest.PrecomputeAndStep` — fixed random memory + mask + state → expected context + alignment.
- `CGRUDecoderCellTest.OneStep` — composition correctness.
- `BiGRUEncoderTest.ForwardSequence` — T=5, B=2 with padding; concat order; zeroed padded positions.
- `RNNDecoderTest.StateReorderGather` — beam-gather after step 3; all four state entries (`hidden`, `memory`, `memory_lengths`, `projected_memory`) are now beam-replicated and reordered by `beam_indices` (see §7.2 for the policy change in Opt 2).

### 9.3 Tier 3 — End-to-end parity against Marian

File: `python/tests/test_marian_s2s_e2e.py`, marked `@pytest.mark.slow`.

- **Model:** the real `model.npz` from §2.
- **Fixture:** ~20 source sentences + Marian-produced translations (`beam-size: 12, normalize: 1, word-penalty: 0`), precomputed once with `marian-decoder` and committed as `.jsonl`. CI does **not** run Marian.
- **Assertion:** exact token-sequence match (fp32). Near-tie sentences excluded from the fixture by design (documented).
- **int8 variant:** `compute_type="int8"`, assert ≥ 90% exact-token agreement with the fp32 run (empirical threshold, documented).
- **CI:** gated by `--runslow`, Linux CPU job only.

### 9.4 Model-file hosting (Q4-adjacent)

The real `model.npz` is ~400 MB and does not live in git. The fixture pulls it from an HTTPS URL in a `conftest.py` fixture, cached in `~/.cache/ctranslate2-tests/`. CI skips Tier 3 if the download fails. URL to be supplied by the user before Tier 3 can run.

### 9.5 Explicitly not tested

- No perf/benchmark regression test.
- No GPU test (not shipping CUDA in phase 1).
- No fuzzing over config permutations — guards tested one at a time; explicit-fail design makes permutations safe-by-construction.

## 10. Risks and mitigations

| # | Risk | Likelihood × impact | Mitigation |
|---|---|---|---|
| 1 | GRU gate ordering wrong in concat | High × High | Tier-1 weight-mapping test with distinct per-slice constants |
| 2 | `final=true` semantics on `decoder_cell2` wrong | Medium × High | Read `tensor_operators.cpp:gruOps` end-to-end; Tier-2 cell test with `final=true` |
| 3 | Attention `b_att` written as affine bias not LN shift | Medium × Medium | Explicit `ln_context_bias` naming; Tier-2 attention test |
| 4 | Mean-pool NaN on zero-length source | Low × Medium | Assert `source_lengths > 0` in `encode()`; raise clear error |
| 5 | Shared-embedding bias mishandled (consume `_W`, drop `_b` or vice-versa) | Low × Medium | Unused-tensor assertion catches both directions |
| 6 | Tied-transposed shape mismatch (hidden layer output ≠ dim_emb) | Low × High | Convert-time shape invariant assertion |
| 7 | BLAS numeric drift vs Marian | Low × Low | Exact-token-match test design (not logit-match); near-tie sentences excluded from fixture |

## 11. Implementation phasing

Phasing becomes the input to the `writing-plans` skill. Each step ends with a working, tested increment.

1. **Scaffolding (0.5 d):** register empty `RNNModelSpec` in C++ and Python; smoke-test factory wiring with empty model directory.
2. **Converter + guards + Tier-1 tests (1–2 d).**
3. **`GRUCell` + Tier-2 test (1–2 d).** Hardest kernel-alignment work.
4. **`BiGRUEncoder` + Tier-2 test (0.5–1 d).**
5. **`BahdanauAttention` + Tier-2 test (1 d).**
6. **`CGRUDecoderCell` + `RNNDecoder` + `DeepOutput` + Tier-2 state/gather test (1–2 d).**
7. **`RNNModel` wiring + first end-to-end decode of a trivial input (0.5 d).**
8. **Tier-3 parity test against the real `model.npz` — debug/iterate (1–3 d, open-ended).**
9. **Int8 wiring + Tier-3 int8 test (0.5–1 d).**
10. **`README_s2s.md` finalisation + `CHANGELOG.md` entry (0.5 d).**

**Total realistic range: 8–14 engineering days** for a working, tested, int8-capable converter and runtime matching the target model exactly.

## 12. Deferred follow-ups

Items that are design-complete but intentionally out of phase 1. Each is a bounded addition that reuses the phase-1 scaffolding.

- Alignment extraction (`return_alignment=True`). ~30 lines in `RNNDecoder` to stash alignments on state.
- Shortlist / `use_vmap`. Mirrors existing Transformer support.
- Marian `.bin` loader. Additive to converter; same spec.
- CUDA fp16/fp32. Build-system + testing; no C++ layer code changes.
- LSTM / `gru-nematus` cell support. New cell class + new spec entry; `_build_s2s_spec` dispatches on `enc-cell` / `dec-cell`.
- `enc-depth > 1` with uni-directional top layers (per `s2s.h:95-120`). New guards relaxed; stacked RNN added to encoder.
- Right-to-left decoding. Reverse source feed + output reversal; converter flag.
- Factored vocab / lemma embeddings.

## 13. Glossary

| Term | Meaning |
|---|---|
| **s2s** | Marian's RNN-based seq2seq model type (as opposed to `transformer`). |
| **cGRU** | Conditional GRU — Marian's `dec-cell-base-depth: 2` scheme: GRU-step → attention → GRU-step. The second GRU uses `final=true`. |
| **Bahdanau attention** | MLP additive attention: `v^T · tanh(W_s · state + W_c · context)`. Marian's default s2s attention. |
| **`final=true`** | A Marian GRU kernel flag that changes the candidate-gate wiring; used for the second cell in a cGRU. |
| **Deep output** | Marian's two-layer logit head: concat `[prev_emb, dec_state, context]` → Dense+LN+tanh → tied-transposed output projection. |
| **Tied-transposed** | Output projection whose weight is `embeddings.weight.transpose()`; only the bias is a separate tensor. |
| **Batch-only vs beam-distinct state** | CT2 `DecoderState` entries can be flagged "shared across all beams of a batch" (not replicated K×) or "one per beam". |

---

*End of design spec. This document is the authoritative reference and is updated alongside implementation.*
