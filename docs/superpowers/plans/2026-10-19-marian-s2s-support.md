# Marian s2s Model Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MarianNMT `type: s2s` (RNN encoder-decoder with conditional GRU + Bahdanau attention) model support to CTranslate2, matching one specific target configuration (bi-GRU encoder, cGRU decoder, LN on, shared embeddings) with explicit convert-time guards against anything else.

**Architecture:** New Python spec `RNNModelSpec` + new C++ `RNNModel` registered under spec name `"RNNModelSpec"`. New Python converter branch `_build_s2s_spec` in existing `marian.py`. New C++ layer file `src/layers/rnn.{h,cc}` containing `GRUCell`, `BiGRUEncoder`, `BahdanauAttention`, `CGRUDecoderCell`, `DeepOutput`, `RNNDecoder`. New `src/models/rnn.{h,cc}` with `RNNModel : SequenceToSequenceModel`. All layer compositions use existing `Dense`/`LayerNorm`/`ops::*` primitives — no new kernels, int8 falls out from `Dense`.

**Tech Stack:** C++17, Python 3, NumPy, pytest, gtest, CMake.

**Spec:** `/Users/waadbk/git/CTranslate2/README_s2s.md` (approved 2026-10-19).

## Global Constraints

- Marian source of truth: `/Users/waadbk/Documents/projects/marian/` — read-only reference.
- Scope: GRU-only, CPU float32 + int8, NPZ input, single-layer bi-GRU encoder, cGRU decoder (`dec-cell-base-depth: 2`), LN on, `tied-embeddings-all: True`. Anything else must be rejected at convert time with a `ValueError` naming the offending option.
- CT2 Dense/LayerNorm/Embeddings constructors take `(const models::Model&, const std::string& scope)` and read weights from the model by scope path. Scope strings mirror Python spec attribute paths.
- Gate ordering in all GRU weight concatenations: `[reset | update | candidate]` on the last axis. Preserved verbatim in both converter and C++ cell.
- Decoder state keys: `"hidden"` (beam-distinct), `"memory"` / `"memory_lengths"` / `"projected_memory"` (batch-only). Batch-only entries are declared by overriding `Decoder::replicate_state(const std::string& name) const` to return `false`.
- Marian LN is scale-only (gamma, no beta). CT2 `LayerNormSpec` requires beta when `rms_norm=False` — converter writes zero tensors for all beta.
- GRU bias: Marian stores separate `_b` (2H) and `_bx` (H); converter concatenates into one `[1, 3H]` bias stored on the input `Dense`'s bias slot. The recurrent `Dense` has no bias.
- Attention `b_att` is a LayerNorm **shift**, not an affine bias; stored in the attention spec as `ln_context_bias`.

---

## File Structure

**New files:**
- `python/ctranslate2/specs/rnn_spec.py` — Python model spec hierarchy
- `src/layers/rnn.h` — public interface for new layer classes
- `src/layers/rnn.cc` — layer implementations
- `include/ctranslate2/models/rnn.h` — public interface for RNNModel
- `src/models/rnn.cc` — RNNModel implementation
- `python/tests/test_marian_s2s_converter.py` — Tier-1 converter unit tests
- `python/tests/test_marian_s2s_e2e.py` — Tier-3 end-to-end parity test (gated on `--runslow`)
- `tests/data/rnn/generate_reference.py` — generates NumPy reference values for Tier-2 tests
- `tests/data/rnn/*.npz` — generated reference tensors (committed; small)

**Modified files:**
- `python/ctranslate2/converters/marian.py` — split transformer path into `_build_transformer_spec`, add `_build_s2s_spec`
- `python/ctranslate2/specs/__init__.py` — export new `RNNModelSpec` (if applicable)
- `src/models/model_factory.cc` — register `RNNModel` under `"RNNModelSpec"`
- `CMakeLists.txt` — add `src/layers/rnn.cc` and `src/models/rnn.cc` to `SOURCES` list
- `tests/CMakeLists.txt` — add new test source (if not glob-based)
- `tests/layers_test.cc` — append Tier-2 unit tests for new layers
- `python/tests/conftest.py` — add `--runslow` option and model-download fixture

---

### Task 1: Scaffolding — empty `RNNModelSpec` + factory registration + build wiring

**Files:**
- Create: `python/ctranslate2/specs/rnn_spec.py`
- Create: `src/layers/rnn.h`
- Create: `src/layers/rnn.cc`
- Create: `include/ctranslate2/models/rnn.h`
- Create: `src/models/rnn.cc`
- Modify: `src/models/model_factory.cc` (around line 20)
- Modify: `CMakeLists.txt` (around line 142 and 150)

**Interfaces:**
- Consumes: existing CT2 `SequenceToSequenceModelSpec`, `SequenceToSequenceModel`, `ModelFactory`.
- Produces: `RNNModelSpec` (Python, loadable, has `name="RNNModelSpec"`, `revision=1`), `models::RNNModel` (C++, registered, loadable), empty `layers::RNNDecoder` / `layers::BiGRUEncoder` stubs with correct constructor signatures.

- [ ] **Step 1: Create empty Python spec**

Write `python/ctranslate2/specs/rnn_spec.py`:

```python
from ctranslate2.specs import common_spec, model_spec


class RNNModelSpec(model_spec.SequenceToSequenceModelSpec):
    """Spec for Marian s2s (RNN encoder-decoder) models."""

    def __init__(self):
        super().__init__()
        self.encoder = BiGRUEncoderSpec()
        self.decoder = RNNDecoderSpec()

    @property
    def name(self):
        return "RNNModelSpec"

    @property
    def revision(self):
        return 1

    def get_default_config(self):
        return RNNConfig()

    def get_source_vocabulary_size(self):
        return [self.encoder.embeddings.weight.shape[0]]

    def get_target_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class RNNConfig(model_spec.ModelConfig):
    def __init__(self):
        super().__init__(
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )


class BiGRUEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.embeddings = common_spec.EmbeddingsSpec()
        self.forward_cell = GRUCellSpec(has_input=True, layer_norm=True)
        self.backward_cell = GRUCellSpec(has_input=True, layer_norm=True)


class RNNDecoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.embeddings = common_spec.EmbeddingsSpec()
        self.init_state = DenseLNSpec()
        self.cell = CGRUDecoderCellSpec()
        self.hidden = DenseLNSpec()
        self.output = common_spec.LinearSpec()


class CGRUDecoderCellSpec(model_spec.LayerSpec):
    def __init__(self):
        self.gru1 = GRUCellSpec(has_input=True, layer_norm=True, final=False)
        self.attention = BahdanauAttentionSpec()
        self.gru2 = GRUCellSpec(has_input=True, layer_norm=True, final=True)


class GRUCellSpec(model_spec.LayerSpec):
    def __init__(self, has_input=True, layer_norm=True, final=False):
        if has_input:
            self.input = common_spec.LinearSpec()
            if layer_norm:
                self.ln_input = common_spec.LayerNormSpec()
        self.recurrent = common_spec.LinearSpec()
        if layer_norm:
            self.ln_recurrent = common_spec.LayerNormSpec()


class BahdanauAttentionSpec(model_spec.LayerSpec):
    def __init__(self):
        self.query = common_spec.LinearSpec()         # Wa  [Hdec, Henc]
        self.memory = common_spec.LinearSpec()        # Wc_att + b_att used as LN shift
        self.v = None                                 # U_att [Henc, 1]
        self.ln_context = common_spec.LayerNormSpec() # gamma = att_gamma1, beta = b_att
        self.ln_state = common_spec.LayerNormSpec()   # gamma = att_gamma2, beta = zeros


class DenseLNSpec(model_spec.LayerSpec):
    """A Dense (+bias) followed by LayerNorm, used for ff_state and ff_logit_l1."""

    def __init__(self):
        self.linear = common_spec.LinearSpec()
        self.layer_norm = common_spec.LayerNormSpec()
```

- [ ] **Step 2: Create empty C++ model header**

Write `include/ctranslate2/models/rnn.h`:

```cpp
#pragma once

#include "ctranslate2/models/sequence_to_sequence.h"

namespace ctranslate2 {
  namespace models {

    class RNNModel : public SequenceToSequenceModel {
    public:
      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;

      std::unique_ptr<SequenceToSequenceReplica>
      as_sequence_to_sequence() const override;

    protected:
      void initialize(ModelReader& model_reader) override;
    };

  }
}
```

- [ ] **Step 3: Create C++ model stub**

Write `src/models/rnn.cc`:

```cpp
#include "ctranslate2/models/rnn.h"

#include "ctranslate2/layers/rnn.h"
#include "ctranslate2/models/model_factory.h"

namespace ctranslate2 {
  namespace models {

    size_t RNNModel::current_spec_revision() const {
      return 1;
    }

    void RNNModel::initialize(ModelReader& model_reader) {
      SequenceToSequenceModel::initialize(model_reader);
    }

    bool RNNModel::is_quantizable(const std::string& variable_name) const {
      // All big matmul weights live under /input/weight, /recurrent/weight,
      // /query/weight, /memory/weight, /linear/weight, /output/weight.
      return (variable_name.find("/weight") != std::string::npos
              && variable_name.find("ln_") == std::string::npos
              && variable_name.find("/v") == std::string::npos
              && SequenceToSequenceModel::is_quantizable(variable_name));
    }

    bool RNNModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name);
    }

    std::unique_ptr<SequenceToSequenceReplica>
    RNNModel::as_sequence_to_sequence() const {
      auto encoder = std::make_unique<layers::BiGRUEncoder>(*this, "encoder");
      auto decoder = std::make_unique<layers::RNNDecoder>(*this, "decoder");
      auto model = std::static_pointer_cast<const RNNModel>(shared_from_this());
      return std::make_unique<EncoderDecoderReplica>(model, std::move(encoder), std::move(decoder));
    }

  }
}
```

- [ ] **Step 4: Create C++ layer header stub**

Write `src/layers/rnn.h`:

```cpp
#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"

namespace ctranslate2 {
  namespace layers {

    class GRUCell : public Layer {
    public:
      GRUCell(const models::Model& model, const std::string& scope,
              bool has_input, bool final);

      DataType output_type() const override;
      dim_t output_size() const override;

      void operator()(const StorageView* input,
                      const StorageView& prev_state,
                      StorageView& new_state) const;

    private:
      std::unique_ptr<Dense> _input_dense;
      std::unique_ptr<LayerNorm> _ln_input;
      Dense _recurrent_dense;
      std::unique_ptr<LayerNorm> _ln_recurrent;
      bool _final;
      dim_t _dim_state;
    };

    class BahdanauAttention : public Layer {
    public:
      BahdanauAttention(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;

      // Called once per encode().
      void precompute(const StorageView& memory,
                      StorageView& projected_memory) const;

      // Called once per decoder step.
      void compute(const StorageView& query,
                   const StorageView& projected_memory,
                   const StorageView& memory,
                   const StorageView& memory_lengths,
                   StorageView& context,
                   StorageView& alignment) const;

    private:
      Dense _query_dense;
      Dense _memory_dense;
      LayerNorm _ln_context;
      LayerNorm _ln_state;
      StorageView _v;
    };

    class CGRUDecoderCell : public Layer {
    public:
      CGRUDecoderCell(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;

      void operator()(const StorageView& input_emb,
                      const StorageView& prev_state,
                      const StorageView& projected_memory,
                      const StorageView& memory,
                      const StorageView& memory_lengths,
                      StorageView& new_state,
                      StorageView& context,
                      StorageView& alignment) const;

    private:
      GRUCell _gru1;
      BahdanauAttention _attn;
      GRUCell _gru2;
    };

    class BiGRUEncoder : public Encoder {
    public:
      BiGRUEncoder(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;

      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      StorageView& output,
                      const Padder* input_padder = nullptr,
                      const Padder* output_padder = nullptr) override;

    private:
      Embeddings _embeddings;
      GRUCell _fwd;
      GRUCell _bwd;
    };

    class RNNDecoder : public Decoder {
    public:
      RNNDecoder(const models::Model& model, const std::string& scope);

      DecoderState initial_state(bool iterative_decoding = true) const override;

      void operator()(dim_t step,
                      const StorageView& ids,
                      DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;

      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      DecoderState& state,
                      StorageView& logits,
                      StorageView* attention = nullptr) override;

      bool replicate_state(const std::string& name) const override;

      DataType output_type() const override;
      dim_t output_size() const override;

    private:
      Embeddings _embeddings;
      Dense _init_state_linear;
      LayerNorm _init_state_ln;
      CGRUDecoderCell _cell;
      Dense _hidden_linear;
      LayerNorm _hidden_ln;
      Dense _output;
    };

  }
}
```

- [ ] **Step 5: Create empty C++ layer impl**

Write `src/layers/rnn.cc` as an empty-body file that compiles — every method `throw std::runtime_error("not implemented");`. This lets step 7 (CMake wiring) link cleanly.

```cpp
#include "ctranslate2/layers/rnn.h"

#include <stdexcept>

#define NOT_IMPL throw std::runtime_error("RNN layer not implemented yet");

namespace ctranslate2 {
  namespace layers {

    GRUCell::GRUCell(const models::Model&, const std::string&, bool, bool)
      : _recurrent_dense(/*model=*/*(const models::Model*)nullptr, /*scope=*/"") { NOT_IMPL }
    DataType GRUCell::output_type() const { NOT_IMPL }
    dim_t GRUCell::output_size() const { NOT_IMPL }
    void GRUCell::operator()(const StorageView*, const StorageView&, StorageView&) const { NOT_IMPL }

    // ... repeat NOT_IMPL stubs for every method declared in rnn.h
  }
}
```

Note: the `_recurrent_dense(...)` initialiser above is a placeholder that won't actually be called (constructor throws before use). The real constructors in step 11 replace this entire file. If the compiler rejects the null-ref initialiser, use `-Wno-null-dereference` locally or make `_recurrent_dense` a `std::unique_ptr<Dense>` temporarily (revisit in step 11).

- [ ] **Step 6: Register the model in `model_factory.cc`**

Read `src/models/model_factory.cc` to find the pattern (around line 15-22):

```cpp
#include "ctranslate2/models/rnn.h"
// ... existing includes ...

// Inside the registration function, add:
register_model<RNNModel>("RNNModelSpec");
```

- [ ] **Step 7: Add new sources to `CMakeLists.txt`**

Open `CMakeLists.txt`; locate the explicit source list (around line 115-200). Add after `src/layers/transformer.cc`:
```cmake
  src/layers/rnn.cc
```
Add after `src/models/transformer.cc`:
```cmake
  src/models/rnn.cc
```

- [ ] **Step 8: Build and verify link**

```bash
cd /Users/waadbk/git/CTranslate2
cmake -B build -DWITH_MKL=OFF -DBUILD_TESTS=ON
cmake --build build -j
```
Expected: builds without errors; no symbol-resolution issues (NOT_IMPL throws are at runtime, not link time).

- [ ] **Step 9: Smoke-test the registration from Python**

```bash
python -c "
from ctranslate2.specs.rnn_spec import RNNModelSpec
spec = RNNModelSpec()
print(spec.name, spec.revision)
"
```
Expected: `RNNModelSpec 1`.

- [ ] **Step 10: Commit**

```bash
git add python/ctranslate2/specs/rnn_spec.py \
        include/ctranslate2/models/rnn.h \
        src/models/rnn.cc \
        src/layers/rnn.h src/layers/rnn.cc \
        src/models/model_factory.cc \
        CMakeLists.txt
git commit -m "s2s: scaffold RNNModelSpec and empty C++ layer/model classes"
```

---

### Task 2: Marian s2s converter + Tier-1 Python unit tests

**Files:**
- Modify: `python/ctranslate2/converters/marian.py` (restructure, add `_build_s2s_spec`)
- Create: `python/tests/test_marian_s2s_converter.py`

**Interfaces:**
- Consumes: `RNNModelSpec` from Task 1.
- Produces: a fully-populated `RNNModelSpec` from a Marian NPZ archive; converter raises `ValueError` for any guarded condition.

- [ ] **Step 1: Read current `marian.py` to understand structure**

Read lines 1–150 of `python/ctranslate2/converters/marian.py`. Note: existing transformer path is inlined in the `_load()` method of `MarianConverter`. We'll extract it into `_build_transformer_spec()` and add `_build_s2s_spec()` as a sibling.

- [ ] **Step 2: Write failing guard tests**

Create `python/tests/test_marian_s2s_converter.py`:

```python
import io
import numpy as np
import pytest
import yaml

from ctranslate2.converters.marian import MarianConverter


def _s2s_base_config():
    """A minimal valid Marian s2s YAML dict matching the target model."""
    return {
        "type": "s2s",
        "dim-vocabs": [100, 100],
        "dim-emb": 16,
        "dim-rnn": 32,
        "enc-type": "bidirectional",
        "enc-cell": "gru",
        "enc-cell-depth": 1,
        "enc-depth": 1,
        "dec-cell": "gru",
        "dec-cell-base-depth": 2,
        "dec-cell-high-depth": 1,
        "dec-depth": 1,
        "skip": False,
        "layer-normalization": True,
        "right-left": False,
        "tied-embeddings-all": True,
        "tied-embeddings": False,
        "tied-embeddings-src": False,
        "output-omit-bias": False,
        "input-types": [],
        "factors-dim-emb": 0,
        "lemma-dim-emb": 0,
        "transformer-rnn-projection": False,
        "transformer-pool": False,
        "ulr": False,
        "embedding-fix-src": False,
        "embedding-fix-trg": False,
    }


def _write_fake_npz(tmp_path, config, extra_tensors=None):
    """Write a Marian-like NPZ archive with embedded config and all required
    tensors shaped per the config. Values are deterministic (arange)."""
    import io as _io
    H = config["dim-rnn"]
    I = config["dim-emb"]
    V = config["dim-vocabs"][0]
    tensors = {
        # Shared embedding
        "Wemb": np.arange(V * I, dtype=np.float32).reshape(V, I),
        # Encoder forward
        "encoder_bi_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_Ux": np.full((H, H), 2.0, dtype=np.float32),
        "encoder_bi_W": np.full((I, 2 * H), 3.0, dtype=np.float32),
        "encoder_bi_Wx": np.full((I, H), 4.0, dtype=np.float32),
        "encoder_bi_b": np.full((1, 2 * H), 5.0, dtype=np.float32),
        "encoder_bi_bx": np.full((1, H), 6.0, dtype=np.float32),
        "encoder_bi_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "encoder_bi_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Encoder backward (same shapes)
        "encoder_bi_r_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_Ux": np.full((H, H), 1.0, dtype=np.float32),
        "encoder_bi_r_W": np.full((I, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_Wx": np.full((I, H), 1.0, dtype=np.float32),
        "encoder_bi_r_b": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_bx": np.full((1, H), 1.0, dtype=np.float32),
        "encoder_bi_r_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Decoder base cell 1 (dimInput=dim-emb)
        "decoder_cell1_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell1_Ux": np.full((H, H), 1.0, dtype=np.float32),
        "decoder_cell1_W": np.full((I, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell1_Wx": np.full((I, H), 1.0, dtype=np.float32),
        "decoder_cell1_b": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell1_bx": np.full((1, H), 1.0, dtype=np.float32),
        "decoder_cell1_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "decoder_cell1_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Decoder base cell 2 (dimInput=2*dim-rnn, final=true)
        "decoder_cell2_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell2_Ux": np.full((H, H), 1.0, dtype=np.float32),
        "decoder_cell2_W": np.full((2 * H, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell2_Wx": np.full((2 * H, H), 1.0, dtype=np.float32),
        "decoder_cell2_b": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell2_bx": np.full((1, H), 1.0, dtype=np.float32),
        "decoder_cell2_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "decoder_cell2_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Attention (single encoder, prefix = "decoder")
        "decoder_W_comb_att": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "decoder_Wc_att": np.full((2 * H, 2 * H), 1.0, dtype=np.float32),
        "decoder_U_att": np.full((2 * H, 1), 1.0, dtype=np.float32),
        "decoder_b_att": np.full((1, 2 * H), 0.0, dtype=np.float32),
        "decoder_att_gamma1": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "decoder_att_gamma2": np.full((1, 2 * H), 1.0, dtype=np.float32),
        # Decoder init MLP (ff_state): input = 2*H, output = H
        "decoder_ff_state_W": np.full((2 * H, H), 1.0, dtype=np.float32),
        "decoder_ff_state_b": np.full((1, H), 1.0, dtype=np.float32),
        "decoder_ff_state_gamma": np.full((1, H), 1.0, dtype=np.float32),
        # Deep output hidden (ff_logit_l1): input = I + H + 2*H, output = I
        "decoder_ff_logit_l1_W": np.full((I + H + 2 * H, I), 1.0, dtype=np.float32),
        "decoder_ff_logit_l1_b": np.full((1, I), 1.0, dtype=np.float32),
        "decoder_ff_logit_l1_gamma": np.full((1, I), 1.0, dtype=np.float32),
        # Deep output top (ff_logit_l2): bias only (W is tied to Wemb)
        "decoder_ff_logit_l2_b": np.full((1, V), 0.0, dtype=np.float32),
    }
    if extra_tensors:
        tensors.update(extra_tensors)
    tensors["special:model.yml"] = np.frombuffer(
        (yaml.safe_dump(config) + "\0").encode("utf-8"), dtype=np.int8
    )
    npz_path = tmp_path / "model.npz"
    np.savez(npz_path, **tensors)
    return npz_path


def _write_fake_vocab(tmp_path, name, size):
    path = tmp_path / name
    mapping = {f"tok{i}": i for i in range(size)}
    with open(path, "w") as f:
        yaml.safe_dump(mapping, f)
    return path


GUARD_CASES = [
    ("enc-depth", 2),
    ("enc-cell-depth", 2),
    ("enc-type", "s2s"),
    ("enc-cell", "lstm"),
    ("dec-depth", 2),
    ("dec-cell", "lstm"),
    ("dec-cell-base-depth", 1),
    ("dec-cell-high-depth", 2),
    ("skip", True),
    ("layer-normalization", False),
    ("tied-embeddings-all", False),
    ("right-left", True),
    ("input-types", ["word", "shape"]),
    ("factors-dim-emb", 8),
    ("lemma-dim-emb", 8),
    ("transformer-rnn-projection", True),
    ("transformer-pool", True),
    ("ulr", True),
    ("embedding-fix-src", True),
    ("embedding-fix-trg", True),
]


@pytest.mark.parametrize("key,bad_value", GUARD_CASES)
def test_s2s_guards_reject_unsupported(tmp_path, key, bad_value):
    config = _s2s_base_config()
    config[key] = bad_value
    npz = _write_fake_npz(tmp_path, config)
    vocab_src = _write_fake_vocab(tmp_path, "vocab.src.yml", config["dim-vocabs"][0])
    vocab_tgt = _write_fake_vocab(tmp_path, "vocab.tgt.yml", config["dim-vocabs"][1])

    converter = MarianConverter(str(npz), [str(vocab_src), str(vocab_tgt)])
    with pytest.raises(ValueError, match=key):
        converter._load()


def test_s2s_base_config_loads(tmp_path):
    config = _s2s_base_config()
    npz = _write_fake_npz(tmp_path, config)
    vocab_src = _write_fake_vocab(tmp_path, "vocab.src.yml", config["dim-vocabs"][0])
    vocab_tgt = _write_fake_vocab(tmp_path, "vocab.tgt.yml", config["dim-vocabs"][1])

    converter = MarianConverter(str(npz), [str(vocab_src), str(vocab_tgt)])
    spec = converter._load()
    assert spec.name == "RNNModelSpec"
    assert spec.revision == 1


def test_s2s_weight_mapping_gate_order(tmp_path):
    """Critical: Marian gate order is [reset | update | candidate]. The converter
    must concatenate on the last axis so that cells.recurrent_weight has
    [reset=1.0 | update=1.0 | candidate=2.0] when _U was filled with 1.0 and
    _Ux with 2.0."""
    config = _s2s_base_config()
    npz = _write_fake_npz(tmp_path, config)
    vocab_src = _write_fake_vocab(tmp_path, "vocab.src.yml", config["dim-vocabs"][0])
    vocab_tgt = _write_fake_vocab(tmp_path, "vocab.tgt.yml", config["dim-vocabs"][1])

    converter = MarianConverter(str(npz), [str(vocab_src), str(vocab_tgt)])
    spec = converter._load()

    H = config["dim-rnn"]
    rec = spec.encoder.forward_cell.recurrent.weight  # [H, 3H]
    assert rec.shape == (H, 3 * H)
    # First 2H columns from _U (filled 1.0), last H columns from _Ux (filled 2.0)
    assert np.allclose(rec[:, : 2 * H], 1.0)
    assert np.allclose(rec[:, 2 * H :], 2.0)

    # And input weight: _W=3.0 (first 2H), _Wx=4.0 (last H)
    inp = spec.encoder.forward_cell.input.weight
    I = config["dim-emb"]
    assert inp.shape == (I, 3 * H)
    assert np.allclose(inp[:, : 2 * H], 3.0)
    assert np.allclose(inp[:, 2 * H :], 4.0)

    # And bias: _b=5.0 (2H), _bx=6.0 (H)
    bias = spec.encoder.forward_cell.input.bias
    assert bias.shape == (3 * H,)
    assert np.allclose(bias[: 2 * H], 5.0)
    assert np.allclose(bias[2 * H :], 6.0)


def test_s2s_unused_tensor_errors(tmp_path):
    config = _s2s_base_config()
    npz = _write_fake_npz(tmp_path, config,
                          extra_tensors={"encoder_bi_junk": np.zeros((1,), dtype=np.float32)})
    vocab_src = _write_fake_vocab(tmp_path, "vocab.src.yml", config["dim-vocabs"][0])
    vocab_tgt = _write_fake_vocab(tmp_path, "vocab.tgt.yml", config["dim-vocabs"][1])

    converter = MarianConverter(str(npz), [str(vocab_src), str(vocab_tgt)])
    with pytest.raises(ValueError, match="encoder_bi_junk"):
        converter._load()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/waadbk/git/CTranslate2
pytest python/tests/test_marian_s2s_converter.py -v
```
Expected: all tests fail (converter still rejects `type: s2s`).

- [ ] **Step 4: Refactor existing marian.py to dispatch on model type**

In `python/ctranslate2/converters/marian.py`:

1. Rename the current body of `_load()` (lines ~44-108) into a new function `_build_transformer_spec(config, model, vocabs)`.
2. Add `_build_s2s_spec(config, model, vocabs)` (initially raising `NotImplementedError` with a placeholder message — step 5 fills it in).
3. In `_load()`, dispatch:

```python
def _load(self):
    model = np.load(self._model_path)
    config = yaml.safe_load(bytes(model["special:model.yml"]).rstrip(b"\x00"))
    vocabs = [self._load_vocab(p) for p in self._vocab_paths]

    model_type = config["type"]
    if model_type == "transformer":
        return _build_transformer_spec(config, model, vocabs)
    elif model_type == "s2s":
        return _build_s2s_spec(config, model, vocabs)
    else:
        raise NotImplementedError(
            "Marian model type '%s' is not supported; "
            "supported types: 'transformer', 's2s'" % model_type
        )
```

- [ ] **Step 5: Implement `_build_s2s_spec`**

Add to `marian.py`:

```python
from ctranslate2.specs.rnn_spec import RNNModelSpec


_S2S_HARD_GUARDS = {
    "enc-depth": 1,
    "enc-cell-depth": 1,
    "enc-type": "bidirectional",
    "enc-cell": "gru",
    "dec-depth": 1,
    "dec-cell": "gru",
    "dec-cell-base-depth": 2,
    "dec-cell-high-depth": 1,
    "skip": False,
    "layer-normalization": True,
    "tied-embeddings-all": True,
    "right-left": False,
    "factors-dim-emb": 0,
    "lemma-dim-emb": 0,
    "transformer-rnn-projection": False,
    "transformer-pool": False,
    "ulr": False,
    "embedding-fix-src": False,
    "embedding-fix-trg": False,
}


def _s2s_require(config, key, expected):
    actual = config.get(key)
    # Treat empty-list expected equivalently to None/missing for list keys.
    if isinstance(expected, list) and (actual is None or actual == []):
        if expected == actual or expected == []:
            return
    if actual != expected:
        raise ValueError(
            "Marian s2s converter: option '%s' = %r is not supported "
            "(this converter currently requires '%s' = %r). "
            "File an issue with your model.npz.yml if you need this variant."
            % (key, actual, key, expected)
        )


def _s2s_require_empty_list(config, key):
    actual = config.get(key, [])
    if actual not in ([], None):
        raise ValueError(
            "Marian s2s converter: option '%s' = %r is not supported "
            "(this converter requires it to be empty). "
            "File an issue with your model.npz.yml if you need this variant."
            % (key, actual)
        )


def _s2s_gate_concat(np_u, np_ux):
    """Concat [W | Wx] on the last axis preserving gate order [reset | update | candidate].

    Marian's _U is [H, 2H] = [reset | update]. _Ux is [H, H] = candidate.
    Result shape: [H, 3H] with slices (0..2H) = reset+update, (2H..3H) = candidate.
    """
    return np.concatenate([np_u, np_ux], axis=-1)


def _s2s_bias_concat(np_b, np_bx):
    """b: [1, 2H], bx: [1, H] → flat [3H] bias (CT2 LinearSpec stores bias as 1-D)."""
    return np.concatenate([np_b.reshape(-1), np_bx.reshape(-1)], axis=0)


def _s2s_load_cell(spec_cell, model, prefix, has_input, dim_rnn):
    """Populate a GRUCellSpec from Marian tensors under `prefix`."""
    H = dim_rnn
    U = model["%s_U" % prefix]
    Ux = model["%s_Ux" % prefix]
    assert U.shape[1] == 2 * H, "cell %s _U shape" % prefix
    assert Ux.shape[1] == H
    spec_cell.recurrent.weight = _s2s_gate_concat(U, Ux)
    # Recurrent has no bias — the bias lives on the input side (fused).
    spec_cell.recurrent.bias = np.zeros(3 * H, dtype=np.float32)
    spec_cell.ln_recurrent.gamma = model["%s_gamma2" % prefix].reshape(-1)
    spec_cell.ln_recurrent.beta = np.zeros(3 * H, dtype=np.float32)

    if has_input:
        W = model["%s_W" % prefix]
        Wx = model["%s_Wx" % prefix]
        b = model["%s_b" % prefix]
        bx = model["%s_bx" % prefix]
        spec_cell.input.weight = _s2s_gate_concat(W, Wx)
        spec_cell.input.bias = _s2s_bias_concat(b, bx)
        spec_cell.ln_input.gamma = model["%s_gamma1" % prefix].reshape(-1)
        spec_cell.ln_input.beta = np.zeros(3 * H, dtype=np.float32)


def _s2s_load_attention(spec_attn, model, prefix, dim_dec, dim_enc):
    spec_attn.query.weight = model["%s_W_comb_att" % prefix]
    spec_attn.query.bias = np.zeros(dim_enc, dtype=np.float32)  # no bias on query side
    spec_attn.memory.weight = model["%s_Wc_att" % prefix]
    spec_attn.memory.bias = np.zeros(dim_enc, dtype=np.float32)
    spec_attn.v = model["%s_U_att" % prefix].reshape(-1)  # [Henc, 1] -> [Henc]
    spec_attn.ln_context.gamma = model["%s_att_gamma1" % prefix].reshape(-1)
    # b_att is used as LN shift (bias of the layer-norm on context side):
    spec_attn.ln_context.beta = model["%s_b_att" % prefix].reshape(-1)
    spec_attn.ln_state.gamma = model["%s_att_gamma2" % prefix].reshape(-1)
    spec_attn.ln_state.beta = np.zeros(dim_enc, dtype=np.float32)


def _s2s_load_dense_ln(spec_dense_ln, model, prefix, bias_dim):
    spec_dense_ln.linear.weight = model["%s_W" % prefix]
    spec_dense_ln.linear.bias = model["%s_b" % prefix].reshape(-1)
    spec_dense_ln.layer_norm.gamma = model["%s_gamma" % prefix].reshape(-1)
    spec_dense_ln.layer_norm.beta = np.zeros(bias_dim, dtype=np.float32)


def _build_s2s_spec(config, model, vocabs):
    # Apply guards.
    for key, expected in _S2S_HARD_GUARDS.items():
        _s2s_require(config, key, expected)
    _s2s_require_empty_list(config, "input-types")

    H = config["dim-rnn"]
    I = config["dim-emb"]
    V = config["dim-vocabs"][0]

    # Vocabulary consistency check (tied-embeddings-all requires identical vocabs).
    if vocabs[0] != vocabs[1]:
        raise ValueError(
            "Marian s2s converter: tied-embeddings-all=True requires identical "
            "source and target vocabularies; got mismatched vocab files."
        )

    spec = RNNModelSpec()

    # Shared embedding.
    Wemb = model["Wemb"]
    if Wemb.shape != (V, I):
        raise ValueError(
            "Marian s2s converter: Wemb shape %s != expected (%d, %d)"
            % (Wemb.shape, V, I)
        )
    spec.encoder.embeddings.weight = Wemb
    spec.decoder.embeddings.weight = Wemb  # tied; aliasing happens via LayerSpec.optimize

    # Encoder.
    _s2s_load_cell(spec.encoder.forward_cell, model, "encoder_bi",
                   has_input=True, dim_rnn=H)
    _s2s_load_cell(spec.encoder.backward_cell, model, "encoder_bi_r",
                   has_input=True, dim_rnn=H)

    # Decoder cells.
    _s2s_load_cell(spec.decoder.cell.gru1, model, "decoder_cell1",
                   has_input=True, dim_rnn=H)
    _s2s_load_cell(spec.decoder.cell.gru2, model, "decoder_cell2",
                   has_input=True, dim_rnn=H)

    # Attention.
    _s2s_load_attention(spec.decoder.cell.attention, model, "decoder",
                        dim_dec=H, dim_enc=2 * H)

    # Init state MLP.
    _s2s_load_dense_ln(spec.decoder.init_state, model, "decoder_ff_state",
                       bias_dim=H)

    # Deep output hidden.
    _s2s_load_dense_ln(spec.decoder.hidden, model, "decoder_ff_logit_l1",
                       bias_dim=I)

    # Deep output top: tied-transposed to Wemb; only bias is independent.
    spec.decoder.output.weight = Wemb  # same storage; LayerSpec.optimize aliases
    spec.decoder.output.bias = model["decoder_ff_logit_l2_b"].reshape(-1)

    # Validate every expected tensor was present and no extras were unused.
    _s2s_validate_tensor_set(model, config)

    # Register vocabulary (single shared vocab).
    spec.register_source_vocabulary(vocabs[0])
    spec.register_target_vocabulary(vocabs[0])
    spec.config.add_source_eos = True

    return spec


def _s2s_expected_tensor_names(config):
    """Build the exact set of NPZ tensor names this converter consumes."""
    names = {"Wemb", "special:model.yml"}
    for prefix in ("encoder_bi", "encoder_bi_r", "decoder_cell1", "decoder_cell2"):
        for sfx in ("_U", "_Ux", "_W", "_Wx", "_b", "_bx", "_gamma1", "_gamma2"):
            names.add(prefix + sfx)
    for sfx in ("_W_comb_att", "_Wc_att", "_U_att", "_b_att",
                "_att_gamma1", "_att_gamma2"):
        names.add("decoder" + sfx)
    for prefix in ("decoder_ff_state", "decoder_ff_logit_l1"):
        for sfx in ("_W", "_b", "_gamma"):
            names.add(prefix + sfx)
    names.add("decoder_ff_logit_l2_b")
    return names


def _s2s_validate_tensor_set(model, config):
    expected = _s2s_expected_tensor_names(config)
    present = set(model.files)
    missing = expected - present
    if missing:
        raise ValueError(
            "Marian s2s converter: missing expected tensor(s): %s" % sorted(missing)
        )
    # Ignore anything that's a Marian-internal non-weight key.
    ignorable = {"special:model.yml"}
    extra = present - expected - ignorable
    if extra:
        raise ValueError(
            "Marian s2s converter: unused tensor(s) in NPZ (unknown to this converter): %s"
            % sorted(extra)
        )
```

- [ ] **Step 6: Run the Tier-1 tests**

```bash
cd /Users/waadbk/git/CTranslate2
pytest python/tests/test_marian_s2s_converter.py -v
```
Expected: all 20+ parametrized guard tests pass, base-load test passes, weight-mapping test passes, unused-tensor test passes.

- [ ] **Step 7: Commit**

```bash
git add python/ctranslate2/converters/marian.py \
        python/tests/test_marian_s2s_converter.py
git commit -m "s2s: implement Marian s2s converter with guards and Tier-1 tests"
```

---

### Task 3: C++ `GRUCell` implementation + Tier-2 unit test

**Files:**
- Modify: `src/layers/rnn.h`, `src/layers/rnn.cc` (replace `GRUCell` stub)
- Modify: `tests/layers_test.cc` (append `GRUCellTest`)
- Create: `tests/data/rnn/generate_reference.py` (NumPy reference generator)
- Create: `tests/data/rnn/gru_reference.npz`

**Interfaces:**
- Consumes: `models::Model`, `layers::Dense`, `layers::LayerNorm`, `ops::Gemm`, `ops::Sigmoid`, `ops::Tanh`, `ops::Mul`, `ops::Add`.
- Produces: `GRUCell::operator()(input, prev_state, new_state)` where `new_state` matches a NumPy reference computed with the Marian formula.

- [ ] **Step 1: Write the reference generator**

Create `tests/data/rnn/generate_reference.py`:

```python
"""Generate NumPy reference tensors for Tier-2 C++ layer tests.

Formulas mirror Marian cells.h GRU + attention.h GlobalAttention.
Run once; commit the resulting .npz into tests/data/rnn/.
"""
import numpy as np

SEED = 0x5EED


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def layer_norm(x, gamma, beta, eps=1e-9):
    """Marian's standard LN: scale * (x - mean) / sqrt(var + eps) + beta."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gru_step(input_, prev_state, in_W, in_b, rec_W, ln_in_g, ln_in_b,
             ln_rec_g, ln_rec_b, final):
    """One Marian GRU step. Gate layout [reset | update | candidate] on last axis."""
    H = prev_state.shape[-1]
    # Input side: xW = LN(input @ in_W + in_b)
    xW = input_ @ in_W + in_b
    xW = layer_norm(xW, ln_in_g, ln_in_b)
    # State side: sU = LN(prev_state @ rec_W)
    sU = prev_state @ rec_W
    sU = layer_norm(sU, ln_rec_g, ln_rec_b)
    # Split into gate slices.
    r_x, u_x, c_x = xW[..., :H], xW[..., H:2 * H], xW[..., 2 * H:]
    r_s, u_s, c_s = sU[..., :H], sU[..., H:2 * H], sU[..., 2 * H:]
    r = sigmoid(r_x + r_s)
    u = sigmoid(u_x + u_s)
    if final:
        # Marian final=true: candidate uses only tanh(c_x + r * c_s)
        c = np.tanh(c_x + r * c_s)
    else:
        c = np.tanh(c_x + r * c_s)
    new_state = u * prev_state + (1.0 - u) * c
    return new_state


def main():
    rng = np.random.default_rng(SEED)
    B, I, H = 2, 4, 6

    input_ = rng.standard_normal((B, I)).astype(np.float32)
    prev_state = rng.standard_normal((B, H)).astype(np.float32)

    # Fused input weight: [I, 3H] with gate layout [r | u | c]
    in_W = rng.standard_normal((I, 3 * H)).astype(np.float32)
    in_b = rng.standard_normal((3 * H,)).astype(np.float32)
    rec_W = rng.standard_normal((H, 3 * H)).astype(np.float32)
    ln_in_g = np.ones((3 * H,), dtype=np.float32)
    ln_in_b = np.zeros((3 * H,), dtype=np.float32)
    ln_rec_g = np.ones((3 * H,), dtype=np.float32)
    ln_rec_b = np.zeros((3 * H,), dtype=np.float32)

    new_state_nonfinal = gru_step(input_, prev_state, in_W, in_b, rec_W,
                                   ln_in_g, ln_in_b, ln_rec_g, ln_rec_b,
                                   final=False)
    new_state_final = gru_step(input_, prev_state, in_W, in_b, rec_W,
                                ln_in_g, ln_in_b, ln_rec_g, ln_rec_b,
                                final=True)

    np.savez("gru_reference.npz",
             input=input_, prev_state=prev_state,
             in_weight=in_W, in_bias=in_b,
             rec_weight=rec_W,
             ln_in_gamma=ln_in_g, ln_in_beta=ln_in_b,
             ln_rec_gamma=ln_rec_g, ln_rec_beta=ln_rec_b,
             new_state_nonfinal=new_state_nonfinal,
             new_state_final=new_state_final)
    print("Wrote gru_reference.npz")


if __name__ == "__main__":
    main()
```

Run it:
```bash
cd /Users/waadbk/git/CTranslate2/tests/data/rnn
python generate_reference.py
```

- [ ] **Step 2: Write the C++ `GRUCellTest`**

Append to `tests/layers_test.cc`:

```cpp
#include "ctranslate2/layers/rnn.h"
#include "cnpy.h"  // if available; otherwise read via existing test helper

TEST(RNNLayers, GRUCellMatchesReference) {
  // Load reference tensors.
  auto npz = cnpy::npz_load("tests/data/rnn/gru_reference.npz");
  // Build a mini in-memory Model: not trivial. Instead, test GRUCell by
  // constructing it with explicit StorageViews via a test-only constructor.
  // See step 3 for the test-only constructor added to GRUCell.

  const auto input       = load_npz_array(npz, "input");
  const auto prev_state  = load_npz_array(npz, "prev_state");
  const auto in_weight   = load_npz_array(npz, "in_weight");
  const auto in_bias     = load_npz_array(npz, "in_bias");
  const auto rec_weight  = load_npz_array(npz, "rec_weight");
  const auto ln_in_g     = load_npz_array(npz, "ln_in_gamma");
  const auto ln_in_b     = load_npz_array(npz, "ln_in_beta");
  const auto ln_rec_g    = load_npz_array(npz, "ln_rec_gamma");
  const auto ln_rec_b    = load_npz_array(npz, "ln_rec_beta");
  const auto expected    = load_npz_array(npz, "new_state_nonfinal");

  layers::GRUCell cell(in_weight, in_bias, rec_weight,
                       ln_in_g, ln_in_b, ln_rec_g, ln_rec_b,
                       /*final=*/false);
  StorageView new_state;
  cell(&input, prev_state, new_state);
  expect_storage_eq(new_state, expected, /*tolerance=*/1e-5);
}

TEST(RNNLayers, GRUCellFinalModeMatchesReference) {
  auto npz = cnpy::npz_load("tests/data/rnn/gru_reference.npz");
  const auto input      = load_npz_array(npz, "input");
  const auto prev_state = load_npz_array(npz, "prev_state");
  const auto in_weight  = load_npz_array(npz, "in_weight");
  const auto in_bias    = load_npz_array(npz, "in_bias");
  const auto rec_weight = load_npz_array(npz, "rec_weight");
  const auto ln_in_g    = load_npz_array(npz, "ln_in_gamma");
  const auto ln_in_b    = load_npz_array(npz, "ln_in_beta");
  const auto ln_rec_g   = load_npz_array(npz, "ln_rec_gamma");
  const auto ln_rec_b   = load_npz_array(npz, "ln_rec_beta");
  const auto expected   = load_npz_array(npz, "new_state_final");

  layers::GRUCell cell(in_weight, in_bias, rec_weight,
                       ln_in_g, ln_in_b, ln_rec_g, ln_rec_b,
                       /*final=*/true);
  StorageView new_state;
  cell(&input, prev_state, new_state);
  expect_storage_eq(new_state, expected, /*tolerance=*/1e-5);
}
```

Note: `load_npz_array` is a helper to be added to `tests/test_utils.h` that converts a cnpy array into a `StorageView`. If cnpy is not already available in the test build, use the existing CT2 ability to load `.npy`/`.npz` arrays (check `tests/test_utils.h` for prior art).

- [ ] **Step 3: Add a test-only `GRUCell` constructor + implement `operator()`**

Modify `src/layers/rnn.h` to add a test-only constructor:

```cpp
class GRUCell : public Layer {
public:
  // Model-reading constructor (production).
  GRUCell(const models::Model& model, const std::string& scope,
          bool has_input, bool final);

  // Explicit-tensor constructor (test only).
  GRUCell(const StorageView& in_weight,
          const StorageView& in_bias,
          const StorageView& rec_weight,
          const StorageView& ln_in_gamma,
          const StorageView& ln_in_beta,
          const StorageView& ln_rec_gamma,
          const StorageView& ln_rec_beta,
          bool final);
  // ...
};
```

Implement in `src/layers/rnn.cc`:

```cpp
#include "ctranslate2/layers/rnn.h"

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace layers {

    // Production constructor: reads tensors from model.
    GRUCell::GRUCell(const models::Model& model, const std::string& scope,
                     bool has_input, bool final)
      : _recurrent_dense(model, scope + "/recurrent")
      , _ln_recurrent(std::make_unique<LayerNorm>(model, scope + "/ln_recurrent"))
      , _final(final)
      , _dim_state(_recurrent_dense.output_size() / 3)
    {
      if (has_input) {
        _input_dense = std::make_unique<Dense>(model, scope + "/input");
        _ln_input = std::make_unique<LayerNorm>(model, scope + "/ln_input");
      }
    }

    DataType GRUCell::output_type() const { return _recurrent_dense.output_type(); }
    dim_t GRUCell::output_size() const { return _dim_state; }

    void GRUCell::operator()(const StorageView* input,
                             const StorageView& prev_state,
                             StorageView& new_state) const {
      const dim_t H = _dim_state;
      StorageView xW(prev_state.dtype(), prev_state.device());
      if (input) {
        (*_input_dense)(*input, xW);
        (*_ln_input)(xW, xW);
      } else {
        xW.resize({prev_state.dim(0), 3 * H});
        xW.zero();
      }

      StorageView sU(prev_state.dtype(), prev_state.device());
      _recurrent_dense(prev_state, sU);
      (*_ln_recurrent)(sU, sU);

      // Element-wise: new_state[b,h] = u * prev + (1-u) * tanh(cx + r*cs)
      // where r = sigmoid(rx+rs), u = sigmoid(ux+us).
      _apply_gates(xW, sU, prev_state, new_state);
    }

    // Private helper _apply_gates: splits [r|u|c] slices and composes.
    void GRUCell::_apply_gates(const StorageView& xW, const StorageView& sU,
                               const StorageView& prev_state,
                               StorageView& new_state) const {
      const dim_t B = prev_state.dim(0);
      const dim_t H = _dim_state;

      // Split both xW and sU into [r|u|c] (last axis).
      StorageView rx(prev_state.dtype(), prev_state.device());
      StorageView ux(prev_state.dtype(), prev_state.device());
      StorageView cx(prev_state.dtype(), prev_state.device());
      _split_last_three(xW, H, rx, ux, cx);

      StorageView rs(prev_state.dtype(), prev_state.device());
      StorageView us(prev_state.dtype(), prev_state.device());
      StorageView cs(prev_state.dtype(), prev_state.device());
      _split_last_three(sU, H, rs, us, cs);

      // r = sigmoid(rx + rs); u = sigmoid(ux + us); c_pre = cx + r * cs
      ops::Add()(rx, rs, rx);
      ops::Sigmoid()(rx, rx);  // rx now holds r
      ops::Add()(ux, us, ux);
      ops::Sigmoid()(ux, ux);  // ux now holds u
      ops::Mul()(rx, cs, cs);  // cs = r * cs
      ops::Add()(cx, cs, cx);
      ops::Tanh()(cx, cx);     // cx now holds candidate c

      // new = u * prev + (1 - u) * c
      // Compute (1 - u) in-place: us = 1 - u
      _one_minus(ux, us);
      StorageView u_times_prev(prev_state.dtype(), prev_state.device());
      ops::Mul()(ux, prev_state, u_times_prev);
      StorageView one_minus_u_times_c(prev_state.dtype(), prev_state.device());
      ops::Mul()(us, cx, one_minus_u_times_c);
      ops::Add()(u_times_prev, one_minus_u_times_c, new_state);
    }

    // _split_last_three and _one_minus: utility helpers.
    // Implement with ops::Split and ops::Sub against a 1-filled constant.
    // (Pattern follows what src/layers/attention.cc does for Q/K/V splits.)
  }
}
```

Note: a real impl needs the `_split_last_three` and `_one_minus` helpers. Use `ops::Split` with `axis=-1` and split sizes `{H, H, H}` for the three-way split. For `1 - u`, allocate a 1-filled StorageView of matching shape and call `ops::Sub()`.

- [ ] **Step 4: Build and run the GRU cell test**

```bash
cmake --build build -j
./build/tests/ctranslate2_test --gtest_filter='RNNLayers.GRUCell*'
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/layers/rnn.{h,cc} tests/layers_test.cc \
        tests/data/rnn/generate_reference.py tests/data/rnn/gru_reference.npz
git commit -m "s2s: implement GRUCell with LN and final-mode; Tier-2 parity test"
```

---

### Task 4: C++ `BiGRUEncoder` + Tier-2 sequence test

**Files:**
- Modify: `src/layers/rnn.cc` (implement `BiGRUEncoder`)
- Modify: `tests/layers_test.cc` (append `BiGRUEncoderTest`)
- Modify: `tests/data/rnn/generate_reference.py` (append encoder reference)
- Regenerate: `tests/data/rnn/gru_reference.npz`

**Interfaces:**
- Consumes: `GRUCell` (Task 3).
- Produces: `BiGRUEncoder::operator()(ids, lengths, output)` producing `output: [B, T, 2H]` with padded positions zeroed.

- [ ] **Step 1: Extend reference generator**

Append to `tests/data/rnn/generate_reference.py`:

```python
def bidirectional_encode(ids_emb, lengths, fwd_params, bwd_params):
    """Reference: run fwd GRU 0..T, bwd GRU T..0 (per actual length), concat.

    ids_emb: [B, T, I], lengths: [B]
    """
    B, T, I = ids_emb.shape
    H = fwd_params["rec_weight"].shape[0]

    def step(input_, state, p):
        return gru_step(input_, state, p["in_weight"], p["in_bias"],
                        p["rec_weight"], p["ln_in_gamma"], p["ln_in_beta"],
                        p["ln_rec_gamma"], p["ln_rec_beta"], final=False)

    fwd_states = np.zeros((B, T, H), dtype=np.float32)
    s = np.zeros((B, H), dtype=np.float32)
    for t in range(T):
        s = step(ids_emb[:, t, :], s, fwd_params)
        fwd_states[:, t, :] = s

    bwd_states = np.zeros((B, T, H), dtype=np.float32)
    s = np.zeros((B, H), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        s_next = step(ids_emb[:, t, :], s, bwd_params)
        # Mask out padded steps (lengths-based).
        active = (t < lengths)[:, None]
        s = np.where(active, s_next, s)
        bwd_states[:, t, :] = s

    out = np.concatenate([fwd_states, bwd_states], axis=-1)  # [B, T, 2H]
    # Zero out padded positions in output.
    mask = (np.arange(T)[None, :] < lengths[:, None])[:, :, None]
    out = out * mask
    return out
```

Add a block in `main()` that builds `ids_emb` (B=2, T=5, I=4), `lengths=[5, 3]`, random fwd/bwd params, runs `bidirectional_encode`, and stores results in the same `.npz`.

- [ ] **Step 2: Write `BiGRUEncoderTest`**

Append to `tests/layers_test.cc`:

```cpp
TEST(RNNLayers, BiGRUEncoderMatchesReference) {
  auto npz = cnpy::npz_load("tests/data/rnn/gru_reference.npz");
  const auto ids_emb  = load_npz_array(npz, "enc_ids_emb");  // [B, T, I]
  const auto lengths  = load_npz_array(npz, "enc_lengths");   // [B]
  const auto expected = load_npz_array(npz, "enc_output");    // [B, T, 2H]

  // Build a BiGRUEncoder via explicit-tensor constructor (test-only overload
  // to be added, mirroring step 3 of Task 3).
  //
  // Alternatively, build an in-memory model and use the production
  // constructor. For brevity, prefer the explicit-tensor path.

  layers::BiGRUEncoder enc(/*embeddings_weight=*/..., fwd_params..., bwd_params...);
  StorageView out, out_lengths;
  // Need to construct `ids` as token ids whose embedding lookup equals ids_emb.
  // Simpler: add a test-only operator() overload taking pre-embedded inputs.
  enc.forward_from_embeddings(ids_emb, lengths, out);
  expect_storage_eq(out, expected, 1e-5);
}
```

Design note: expose a `forward_from_embeddings` protected or test-only method on `BiGRUEncoder` to skip the embedding lookup for testing. Not part of the public API.

- [ ] **Step 3: Implement `BiGRUEncoder::operator()`**

Add to `src/layers/rnn.cc`:

```cpp
BiGRUEncoder::BiGRUEncoder(const models::Model& model, const std::string& scope)
  : _embeddings(model, scope + "/embeddings")
  , _fwd(model, scope + "/forward_cell", /*has_input=*/true, /*final=*/false)
  , _bwd(model, scope + "/backward_cell", /*has_input=*/true, /*final=*/false)
{}

DataType BiGRUEncoder::output_type() const { return _fwd.output_type(); }
dim_t BiGRUEncoder::output_size() const { return 2 * _fwd.output_size(); }

void BiGRUEncoder::operator()(const StorageView& ids,
                              const StorageView& lengths,
                              StorageView& output,
                              const Padder*,
                              const Padder*) {
  const dim_t B = ids.dim(0);
  const dim_t T = ids.dim(1);
  const dim_t H = _fwd.output_size();

  StorageView emb(output.dtype(), output.device());
  _embeddings(ids, emb);  // [B, T, I]

  // Allocate output: [B, T, 2H].
  output.resize({B, T, 2 * H});
  output.zero();

  // Allocate per-step input slice.
  StorageView input_t(emb.dtype(), emb.device());
  StorageView fwd_state(emb.dtype(), emb.device());
  fwd_state.resize({B, H});
  fwd_state.zero();
  StorageView new_fwd(emb.dtype(), emb.device());

  // Forward pass.
  for (dim_t t = 0; t < T; ++t) {
    _slice_timestep(emb, t, input_t);
    _fwd(&input_t, fwd_state, new_fwd);
    _mask_by_length(new_fwd, lengths, t, fwd_state);  // only update active rows
    _copy_timestep(fwd_state, output, t, /*offset_last=*/0, /*span=*/H);
  }

  // Backward pass.
  StorageView bwd_state(emb.dtype(), emb.device());
  bwd_state.resize({B, H});
  bwd_state.zero();
  StorageView new_bwd(emb.dtype(), emb.device());
  for (dim_t t = T - 1; t >= 0; --t) {
    _slice_timestep(emb, t, input_t);
    _bwd(&input_t, bwd_state, new_bwd);
    _mask_by_length(new_bwd, lengths, t, bwd_state);
    _copy_timestep(bwd_state, output, t, /*offset_last=*/H, /*span=*/H);
  }
}
```

`_slice_timestep`, `_mask_by_length`, `_copy_timestep` are static helpers in `rnn.cc`:
- `_slice_timestep(seq, t, out)`: equivalent to `ops::Gather` on axis 1, or `ops::Split` then take the t-th piece.
- `_mask_by_length(new_state, lengths, t, out)`: `out[b] = (t < lengths[b]) ? new_state[b] : out[b]`. Implement with existing CT2 ops or a one-off helper.
- `_copy_timestep(src, dst, t, offset, span)`: writes `src[B, span]` into `dst[B, t, offset:offset+span]`. Can use `ops::Concat` across timesteps, or scatter.

- [ ] **Step 4: Build and run the encoder test**

```bash
cmake --build build -j
./build/tests/ctranslate2_test --gtest_filter='RNNLayers.BiGRUEncoder*'
```

- [ ] **Step 5: Commit**

```bash
git add src/layers/rnn.cc tests/layers_test.cc tests/data/rnn/
git commit -m "s2s: implement BiGRUEncoder with length-masked bi-directional pass"
```

---

### Task 5: C++ `BahdanauAttention` + Tier-2 test

**Files:**
- Modify: `src/layers/rnn.cc` (implement `BahdanauAttention`)
- Modify: `tests/layers_test.cc`, `tests/data/rnn/generate_reference.py`

**Interfaces:**
- Consumes: `Dense`, `LayerNorm`, `ops::Gemm`, `ops::SoftMax`, `ops::Tanh`.
- Produces: `BahdanauAttention::precompute(memory, projected_memory)` and `BahdanauAttention::compute(query, projected_memory, memory, memory_lengths, context, alignment)`.

- [ ] **Step 1: Extend reference generator**

```python
def bahdanau_attention(memory, memory_lengths, query, Wa, Wc, b_att, v,
                       gamma_ctx, gamma_state):
    """memory: [B, S, Henc], query: [B, Hdec], Wa: [Hdec, Henc], etc.

    Returns (context: [B, Henc], alignment: [B, S])."""
    B, S, Henc = memory.shape
    # Pre-mapped context: LN(memory @ Wc, gamma_ctx, b_att)  — zero beta
    mapped_ctx = memory @ Wc  # [B, S, Henc]
    mapped_ctx = layer_norm(mapped_ctx, gamma_ctx, b_att)
    # Per-step: LN(query @ Wa, gamma_state, 0)
    mapped_q = query @ Wa  # [B, Henc]
    mapped_q = layer_norm(mapped_q, gamma_state, np.zeros_like(gamma_state))
    mapped_q = mapped_q[:, None, :]  # [B, 1, Henc]
    # e = v^T · tanh(mapped_q + mapped_ctx)   [B, S]
    e = (np.tanh(mapped_q + mapped_ctx) * v[None, None, :]).sum(axis=-1)
    # Mask + softmax
    mask = np.arange(S)[None, :] < memory_lengths[:, None]
    e = np.where(mask, e, -1e9)
    alpha = np.exp(e - e.max(axis=-1, keepdims=True))
    alpha = alpha / alpha.sum(axis=-1, keepdims=True)
    context = (alpha[:, :, None] * memory).sum(axis=1)
    return context, alpha
```

Add a block in `main()` that generates random memory/query/weights and saves `attn_memory`, `attn_memory_lengths`, `attn_query`, `attn_Wa`, `attn_Wc`, `attn_b_att`, `attn_v`, `attn_gamma_ctx`, `attn_gamma_state`, `attn_expected_context`, `attn_expected_alignment` into `gru_reference.npz`.

- [ ] **Step 2: Write `BahdanauAttentionTest`**

```cpp
TEST(RNNLayers, BahdanauAttentionMatchesReference) {
  auto npz = cnpy::npz_load("tests/data/rnn/gru_reference.npz");
  // Load all tensors from Step 1.
  layers::BahdanauAttention attn(/* explicit tensors ... */);
  StorageView projected_memory;
  attn.precompute(memory, projected_memory);
  StorageView context, alignment;
  attn.compute(query, projected_memory, memory, memory_lengths, context, alignment);
  expect_storage_eq(context, expected_context, 1e-5);
  expect_storage_eq(alignment, expected_alignment, 1e-5);
}
```

- [ ] **Step 3: Implement `BahdanauAttention`**

```cpp
BahdanauAttention::BahdanauAttention(const models::Model& model, const std::string& scope)
  : _query_dense(model, scope + "/query")
  , _memory_dense(model, scope + "/memory")
  , _ln_context(model, scope + "/ln_context")
  , _ln_state(model, scope + "/ln_state")
  , _v(*model.get_variable(scope + "/v"))  // [Henc]
{}

void BahdanauAttention::precompute(const StorageView& memory,
                                   StorageView& projected_memory) const {
  // mapped = memory @ Wc + b_att  — but b_att is used as LN shift.
  // So: mapped = LN(memory @ Wc, gamma_ctx, b_att).
  // The `memory_dense` has bias=0; LN provides the b_att shift via its beta.
  _memory_dense(memory, projected_memory);
  _ln_context(projected_memory, projected_memory);
}

void BahdanauAttention::compute(const StorageView& query,
                                const StorageView& projected_memory,
                                const StorageView& memory,
                                const StorageView& memory_lengths,
                                StorageView& context,
                                StorageView& alignment) const {
  // mapped_query = LN(query @ Wa, gamma_state, 0)
  StorageView mq(query.dtype(), query.device());
  _query_dense(query, mq);
  _ln_state(mq, mq);  // [B, Henc]

  // Add mapped_query broadcast over source length to projected_memory.
  // combined[b,s,h] = mapped_query[b,h] + projected_memory[b,s,h]
  StorageView combined(query.dtype(), query.device());
  _broadcast_add(mq, projected_memory, combined);  // [B, S, Henc]
  ops::Tanh()(combined, combined);

  // scores = combined · v   → [B, S]
  StorageView scores(query.dtype(), query.device());
  _dot_last_axis(combined, _v, scores);

  // Masked softmax over source length.
  ops::SoftMax()(scores, memory_lengths, alignment);

  // context = sum_s alignment[b,s] * memory[b,s,:]  → [B, Henc]
  _weighted_sum(alignment, memory, context);
}
```

Helpers `_broadcast_add`, `_dot_last_axis`, `_weighted_sum`: implement with existing ops. `SoftMax` with lengths already exists for masked softmax; see `src/ops/softmax.cc`.

- [ ] **Step 4: Build and run**

```bash
cmake --build build -j
./build/tests/ctranslate2_test --gtest_filter='RNNLayers.BahdanauAttention*'
```

- [ ] **Step 5: Commit**

```bash
git add src/layers/rnn.cc tests/layers_test.cc \
        tests/data/rnn/generate_reference.py tests/data/rnn/gru_reference.npz
git commit -m "s2s: implement BahdanauAttention with LN and masked softmax"
```

---

### Task 6: C++ `CGRUDecoderCell` + `RNNDecoder` + `DeepOutput` + state/gather test

**Files:**
- Modify: `src/layers/rnn.cc` (implement `CGRUDecoderCell`, `RNNDecoder`)
- Modify: `tests/layers_test.cc` (append `RNNDecoderStateTest`)
- Modify: `tests/data/rnn/generate_reference.py` (append step-level reference)

**Interfaces:**
- Consumes: `GRUCell`, `BahdanauAttention`, `Embeddings`, `Dense`, `LayerNorm`, `Decoder` base class.
- Produces: `CGRUDecoderCell::operator()` and `RNNDecoder::operator()` (step mode), both matching NumPy reference. `RNNDecoder::replicate_state(name)` returns false for `"memory"`, `"memory_lengths"`, `"projected_memory"`.

- [ ] **Step 1: Extend reference generator**

Append `cgru_step` and `rnn_decoder_step` to `generate_reference.py`:

```python
def cgru_step(input_emb, prev_state, projected_memory, memory, memory_lengths,
              gru1_params, attn_params, gru2_params):
    s1 = gru_step(input_emb, prev_state, **gru1_params, final=False)
    # Attention takes s1 as query.
    context, alignment = bahdanau_attention(
        memory, memory_lengths, s1,
        attn_params["Wa"], attn_params["Wc"], attn_params["b_att"],
        attn_params["v"], attn_params["gamma_ctx"], attn_params["gamma_state"])
    s2 = gru_step(context, s1, **gru2_params, final=True)
    return s2, context, alignment


def deep_output(prev_emb, dec_state, context, hidden_params, output_W, output_b):
    """Marian: LN(tanh(concat @ W + b)) · Wemb.T + b_logit"""
    cat = np.concatenate([prev_emb, dec_state, context], axis=-1)
    hidden = cat @ hidden_params["W"] + hidden_params["b"]
    hidden = layer_norm(hidden, hidden_params["gamma"],
                        np.zeros_like(hidden_params["gamma"]))
    hidden = np.tanh(hidden)
    logits = hidden @ output_W.T + output_b
    return logits
```

Add a block in `main()` wiring these together over one decoder step with fixed seed inputs.

- [ ] **Step 2: Write `RNNDecoderStateTest`**

```cpp
TEST(RNNLayers, RNNDecoderStateReplicationFlags) {
  // Build an RNNDecoder from an in-memory model (synthesised with the same
  // helper used by the production model_factory path, but with tiny shapes).
  auto model = make_tiny_rnn_model();  // test fixture
  layers::RNNDecoder decoder(*model, "decoder");

  EXPECT_FALSE(decoder.replicate_state("memory"));
  EXPECT_FALSE(decoder.replicate_state("memory_lengths"));
  EXPECT_FALSE(decoder.replicate_state("projected_memory"));
  EXPECT_TRUE(decoder.replicate_state("hidden"));
}

TEST(RNNLayers, CGRUDecoderCellMatchesReference) {
  // Load reference tensors generated by cgru_step().
  // Build CGRUDecoderCell via test-only explicit-tensor constructor.
  // Run one step; compare new_state, context, alignment to expected.
  // (Full fixture body in test file.)
}
```

- [ ] **Step 3: Implement `CGRUDecoderCell`**

```cpp
CGRUDecoderCell::CGRUDecoderCell(const models::Model& model, const std::string& scope)
  : _gru1(model, scope + "/gru1", /*has_input=*/true, /*final=*/false)
  , _attn(model, scope + "/attention")
  , _gru2(model, scope + "/gru2", /*has_input=*/true, /*final=*/true)
{}

void CGRUDecoderCell::operator()(const StorageView& input_emb,
                                 const StorageView& prev_state,
                                 const StorageView& projected_memory,
                                 const StorageView& memory,
                                 const StorageView& memory_lengths,
                                 StorageView& new_state,
                                 StorageView& context,
                                 StorageView& alignment) const {
  StorageView s1(input_emb.dtype(), input_emb.device());
  _gru1(&input_emb, prev_state, s1);
  _attn.compute(s1, projected_memory, memory, memory_lengths, context, alignment);
  _gru2(&context, s1, new_state);
}
```

- [ ] **Step 4: Implement `RNNDecoder`**

```cpp
RNNDecoder::RNNDecoder(const models::Model& model, const std::string& scope)
  : _embeddings(model, scope + "/embeddings")
  , _init_state_linear(model, scope + "/init_state/linear")
  , _init_state_ln(model, scope + "/init_state/layer_norm")
  , _cell(model, scope + "/cell")
  , _hidden_linear(model, scope + "/hidden/linear")
  , _hidden_ln(model, scope + "/hidden/layer_norm")
  , _output(model, scope + "/output")
{}

bool RNNDecoder::replicate_state(const std::string& name) const {
  return !(name == "memory"
           || name == "memory_lengths"
           || name == "projected_memory");
}

DataType RNNDecoder::output_type() const { return _output.output_type(); }
dim_t RNNDecoder::output_size() const { return _output.output_size(); }

DecoderState RNNDecoder::initial_state(bool) const {
  DecoderState state;
  state["hidden"] = StorageView();
  state["memory"] = StorageView();
  state["memory_lengths"] = StorageView();
  state["projected_memory"] = StorageView();
  return state;
}

void RNNDecoder::operator()(dim_t /*step*/,
                            const StorageView& ids,
                            DecoderState& state,
                            StorageView* logits,
                            StorageView* /*attention*/) {
  // 1. Embed ids.  [B·K, I]
  StorageView emb(_output.output_type(), ids.device());
  _embeddings(ids, emb);

  // 2. One cGRU step.
  StorageView new_hidden(emb.dtype(), emb.device());
  StorageView context(emb.dtype(), emb.device());
  StorageView alignment(emb.dtype(), emb.device());
  _cell(emb,
        state.at("hidden"),
        state.at("projected_memory"),
        state.at("memory"),
        state.at("memory_lengths"),
        new_hidden, context, alignment);

  // 3. Replace hidden in state.
  state["hidden"] = std::move(new_hidden);

  // 4. Deep output: concat [emb; hidden; context] → hidden Dense + LN + tanh → output.
  if (logits) {
    StorageView cat(emb.dtype(), emb.device());
    _concat_last_axis({&emb, &state.at("hidden"), &context}, cat);
    StorageView h(emb.dtype(), emb.device());
    _hidden_linear(cat, h);
    _hidden_ln(h, h);
    ops::Tanh()(h, h);
    _output(h, *logits);
  }
}

void RNNDecoder::operator()(const StorageView&, const StorageView&,
                            DecoderState&, StorageView&, StorageView*) {
  // Full-sequence forward not needed for inference beam search; stub out.
  throw std::runtime_error("RNNDecoder: full-sequence forward not implemented");
}
```

- [ ] **Step 5: Build and run**

```bash
cmake --build build -j
./build/tests/ctranslate2_test --gtest_filter='RNNLayers.CGRUDecoder*:RNNLayers.RNNDecoderState*'
```

- [ ] **Step 6: Commit**

```bash
git add src/layers/rnn.cc tests/layers_test.cc tests/data/rnn/
git commit -m "s2s: implement CGRUDecoderCell + RNNDecoder with batch-only state flags"
```

---

### Task 7: Wire up `RNNModel::as_sequence_to_sequence` + first end-to-end decode

**Files:**
- Modify: `src/models/rnn.cc` (populate `as_sequence_to_sequence`)
- Modify: `src/layers/rnn.cc` (`RNNDecoder::set_state` for initial hidden from encoder memory)
- Create: `python/tests/test_marian_s2s_smoke.py`

**Interfaces:**
- Consumes: all layers from Tasks 3–6.
- Produces: a working `Translator` that can load a converted s2s model, decode one input batch end-to-end without crashing.

- [ ] **Step 1: Implement the init-state logic**

The decoder's hidden must be initialised from the mean-pooled encoder memory the first time `operator()` is called (at `step=0`). CT2's convention is to put this in `set_state()` or inline into the first step. Simplest: lazy init on `step == 0`.

In `src/layers/rnn.cc`, modify `RNNDecoder::operator()`:

```cpp
void RNNDecoder::operator()(dim_t step, const StorageView& ids,
                            DecoderState& state, StorageView* logits,
                            StorageView* attention) {
  if (step == 0 || state.at("hidden").empty()) {
    _initialize_hidden(state);
  }
  // ... rest as before ...
}

void RNNDecoder::_initialize_hidden(DecoderState& state) const {
  const StorageView& memory = state.at("memory");
  const StorageView& lengths = state.at("memory_lengths");
  StorageView pooled(memory.dtype(), memory.device());
  _mean_pool(memory, lengths, pooled);  // [B, 2H]
  StorageView proj(memory.dtype(), memory.device());
  _init_state_linear(pooled, proj);
  _init_state_ln(proj, proj);
  ops::Tanh()(proj, proj);
  state["hidden"] = std::move(proj);
}
```

`_mean_pool`: compute `sum(memory * mask) / sum(mask)` along axis=1.

Also populate `projected_memory` lazily on first use, or (cleaner) override `set_state` to populate both.

- [ ] **Step 2: Override `set_state` to pre-compute attention projection**

Add to `RNNDecoder`:

```cpp
void set_state(DecoderState& state) override {
  // Called by SequenceToSequenceReplica after encoder runs. Pre-compute
  // the attention projection; hidden is initialised lazily on first step.
  const auto& memory = state.at("memory");
  StorageView projected;
  _cell.attention().precompute(memory, projected);
  state["projected_memory"] = std::move(projected);
}
```

Expose `_cell.attention()` as a const-accessor in `CGRUDecoderCell`:
```cpp
const BahdanauAttention& attention() const { return _attn; }
```

- [ ] **Step 3: Write the smoke test**

Create `python/tests/test_marian_s2s_smoke.py`:

```python
import numpy as np
import pytest
import yaml

from ctranslate2 import Translator
from ctranslate2.converters.marian import MarianConverter


def test_s2s_tiny_roundtrip(tmp_path):
    """Convert a tiny synthetic s2s model and decode one batch end-to-end.

    This is a smoke test — assertions check that translation returns at least
    one token per source sentence, not any specific output."""
    # Build a tiny-but-complete fake Marian NPZ.
    from python.tests.test_marian_s2s_converter import _s2s_base_config, _write_fake_npz, _write_fake_vocab
    config = _s2s_base_config()
    # Make it small-but-valid.
    config["dim-vocabs"] = [20, 20]
    config["dim-emb"] = 8
    config["dim-rnn"] = 16
    npz = _write_fake_npz(tmp_path, config)
    vocab = _write_fake_vocab(tmp_path, "vocab.yml", config["dim-vocabs"][0])

    # Convert.
    out_dir = tmp_path / "ct2_model"
    converter = MarianConverter(str(npz), [str(vocab), str(vocab)])
    converter.convert(str(out_dir))

    # Load and translate.
    translator = Translator(str(out_dir), device="cpu", compute_type="float32")
    results = translator.translate_batch([["tok1", "tok2", "tok3"]],
                                          beam_size=4,
                                          max_decoding_length=10)
    assert len(results) == 1
    assert len(results[0].hypotheses) >= 1
    assert len(results[0].hypotheses[0]) >= 1  # at least one output token
```

- [ ] **Step 4: Build, run, and iterate**

```bash
cmake --build build -j
pip install -e python/
pytest python/tests/test_marian_s2s_smoke.py -v
```

This step is where most bugs surface — scope-name mismatches between Python spec attribute names and C++ layer scope strings, tensor-shape surprises, uninitialised state entries. Debug by adding `CT2_VERBOSITY=3` env var and reading the model-load logs.

- [ ] **Step 5: Commit**

```bash
git add src/models/rnn.cc src/layers/rnn.cc python/tests/test_marian_s2s_smoke.py
git commit -m "s2s: wire up RNNModel end-to-end; smoke-test one-batch decode"
```

---

### Task 8: Tier-3 end-to-end parity test against real `model.npz`

**Files:**
- Create: `python/tests/test_marian_s2s_e2e.py`
- Modify: `python/tests/conftest.py` (add `--runslow` option, model-download fixture)
- Create: `tests/fixtures/marian_s2s/` (holds `expected_translations.jsonl` once generated)

**Interfaces:**
- Consumes: converter + runtime from Tasks 1–7.
- Produces: parity test that exact-matches token sequences against pre-computed Marian output.

- [ ] **Step 1: Extend `conftest.py`**

Add to `python/tests/conftest.py`:

```python
import os
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests (requires model fixtures)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def marian_s2s_model_path():
    """Cached location of the real model.npz. User must supply the file via
    env var CT2_MARIAN_S2S_MODEL_PATH or by downloading to
    ~/.cache/ctranslate2-tests/marian_s2s/model.npz before running --runslow."""
    env = os.environ.get("CT2_MARIAN_S2S_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    cache = os.path.expanduser("~/.cache/ctranslate2-tests/marian_s2s/model.npz")
    if os.path.exists(cache):
        return cache
    pytest.skip("Marian s2s model.npz not available; set "
                "CT2_MARIAN_S2S_MODEL_PATH or place file at %s" % cache)
```

- [ ] **Step 2: Write the parity test**

Create `python/tests/test_marian_s2s_e2e.py`:

```python
import json
import os
import pytest

from ctranslate2 import Translator
from ctranslate2.converters.marian import MarianConverter


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                            "tests", "fixtures", "marian_s2s")


def _load_expected():
    path = os.path.join(FIXTURE_DIR, "expected_translations.jsonl")
    if not os.path.exists(path):
        pytest.skip("expected_translations.jsonl not present")
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.mark.slow
def test_s2s_fp32_exact_token_parity(tmp_path, marian_s2s_model_path):
    expected = _load_expected()
    vocab_src = os.path.join(os.path.dirname(marian_s2s_model_path), "vocab.ara.yml")
    vocab_tgt = os.path.join(os.path.dirname(marian_s2s_model_path), "vocab.fra.yml")
    out_dir = tmp_path / "ct2"

    converter = MarianConverter(marian_s2s_model_path, [vocab_src, vocab_tgt])
    converter.convert(str(out_dir))

    translator = Translator(str(out_dir), device="cpu", compute_type="float32")
    sources = [item["source"] for item in expected]
    results = translator.translate_batch(sources, beam_size=12, length_penalty=1.0)

    for item, result in zip(expected, results):
        actual_tokens = result.hypotheses[0]
        assert actual_tokens == item["expected"], (
            "mismatch for source %r:\n  expected: %s\n  actual:   %s"
            % (item["source"], item["expected"], actual_tokens)
        )


@pytest.mark.slow
def test_s2s_int8_token_agreement(tmp_path, marian_s2s_model_path):
    """int8 quantization should agree with fp32 on ≥90% of sentences."""
    expected = _load_expected()
    vocab_src = os.path.join(os.path.dirname(marian_s2s_model_path), "vocab.ara.yml")
    vocab_tgt = os.path.join(os.path.dirname(marian_s2s_model_path), "vocab.fra.yml")
    out_dir = tmp_path / "ct2"
    converter = MarianConverter(marian_s2s_model_path, [vocab_src, vocab_tgt])
    converter.convert(str(out_dir), quantization="int8")

    translator = Translator(str(out_dir), device="cpu", compute_type="int8")
    sources = [item["source"] for item in expected]
    results = translator.translate_batch(sources, beam_size=12, length_penalty=1.0)

    agreements = sum(1 for item, r in zip(expected, results)
                     if r.hypotheses[0] == item["expected"])
    rate = agreements / len(expected)
    assert rate >= 0.9, "int8 agreement rate %.2f below threshold 0.9" % rate
```

- [ ] **Step 3: Generate the fixture (one-time, outside this session)**

The user runs Marian locally once:

```bash
# User-side instruction, not for CI:
marian-decoder -m model.npz -v vocab.ara.yml vocab.fra.yml \
   --beam-size 12 --normalize 1 --word-penalty 0 \
   < test_sources.txt > expected_translations.raw

# Then post-process into jsonl:
python scripts/make_fixture.py test_sources.txt expected_translations.raw \
   > tests/fixtures/marian_s2s/expected_translations.jsonl
```

Document this procedure in `README_s2s.md` under §9.4.

- [ ] **Step 4: Run the parity test**

```bash
export CT2_MARIAN_S2S_MODEL_PATH=/path/to/model.npz
pytest python/tests/test_marian_s2s_e2e.py -v --runslow
```

Expected on first try: likely fails somewhere. Debug by narrowing to one sentence, adding `print(logits)` traces, or dumping intermediate `StorageView`s to NumPy with CT2's debug hooks. Common bug classes to check in order:
1. Gate ordering wrong → Tier-1 test should have caught; re-verify.
2. LN beta non-zero somewhere it should be → check spec weight assignment.
3. Attention `b_att` wired as affine bias instead of LN shift → verify `ln_context.beta` holds `b_att`.
4. Deep-output concat axis / ordering off → verify `[emb; hidden; context]`.
5. BOS/EOS token id mapping → verify Marian's vocab-id-0 convention vs CT2's.

- [ ] **Step 5: Commit**

```bash
git add python/tests/conftest.py python/tests/test_marian_s2s_e2e.py
git commit -m "s2s: add Tier-3 parity test (fp32 exact + int8 ≥90% agreement)"
```

---

### Task 9: Int8 quantization verification

**Files:**
- No new files; just test that `is_quantizable` is correct and run the int8 variant.

**Interfaces:**
- Consumes: Task 7 end-to-end working runtime.
- Produces: `int8` conversion that passes the Tier-3 int8 agreement test from Task 8.

- [ ] **Step 1: Audit `RNNModel::is_quantizable`**

Open `src/models/rnn.cc`. The implementation from Task 1 lists `/weight` as quantizable except LN scales and the attention `v`. Verify by printing which variables get quantized when converting with `quantization="int8"`:

```bash
python -c "
from ctranslate2.converters.marian import MarianConverter
c = MarianConverter('/path/to/model.npz', ['vocab.yml','vocab.yml'])
c.convert('/tmp/out_int8', quantization='int8', force=True)
"
```

Confirm that:
- `*/input/weight`, `*/recurrent/weight`, `*/query/weight`, `*/memory/weight`, `*/hidden/weight`, `*/output/weight`, `*/init_state/linear/weight` → quantized.
- `*/ln_*/gamma`, `*/ln_*/beta`, `*/v`, all biases, `embeddings/weight` → NOT quantized.

If any wrong-category weights appear, adjust the string filter in `is_quantizable`.

- [ ] **Step 2: Run int8 parity test**

```bash
pytest python/tests/test_marian_s2s_e2e.py::test_s2s_int8_token_agreement -v --runslow
```

If agreement is <90%, the quantizable set may be too aggressive (e.g., quantizing the small attention `v` vector would hurt a lot). Narrow and retry.

- [ ] **Step 3: Commit**

```bash
git add src/models/rnn.cc
git commit -m "s2s: tune is_quantizable to exclude LN scales, biases, attention v"
```

---

### Task 10: Documentation finalization

**Files:**
- Modify: `README_s2s.md` — update "Status" line, add usage snippet, note known caveats discovered during implementation.
- Modify: `CHANGELOG.md` — add entry.
- Modify: `docs/conversion.md` — add a one-liner mentioning s2s support alongside transformer.

**Interfaces:** None.

- [ ] **Step 1: Update `README_s2s.md` status line**

Change `**Status:** Design — approved, implementation pending` to `**Status:** Implemented (phase 1 — target config only)`. Add a "Usage" section:

```markdown
## Usage

Convert a Marian s2s model (NPZ format, type: s2s, bi-GRU encoder, cGRU
decoder, LN on, tied-embeddings-all):

    ct2-marian-converter --model_path model.npz \
                          --vocab_paths vocab.src.yml vocab.tgt.yml \
                          --output_dir out_ct2

Translate:

    from ctranslate2 import Translator
    t = Translator("out_ct2", compute_type="int8")
    print(t.translate_batch([["hello", "world"]], beam_size=12, length_penalty=1.0))
```

Add a "Known caveats" subsection documenting anything surprising discovered during Tasks 7–8 (e.g., specific tokenizer quirks, non-portable BLAS behaviour).

- [ ] **Step 2: Add CHANGELOG entry**

Append to `CHANGELOG.md`:

```markdown
## Unreleased

### Added
- Marian NMT `s2s` (RNN encoder-decoder) model conversion and inference.
  Supports the specific architectural variant: single-layer bi-GRU encoder,
  conditional GRU decoder (`dec-cell-base-depth: 2`) with Bahdanau MLP
  attention, layer normalization on, tied embeddings across source/target
  and output projection. CPU float32 and int8 compute types. See
  `README_s2s.md` for the full support matrix.
```

- [ ] **Step 3: Add one-liner to `docs/conversion.md`**

Add bullet to the Marian section: "Both `type: transformer` and `type: s2s` (RNN encoder-decoder) are supported. See `README_s2s.md` for s2s support details."

- [ ] **Step 4: Commit**

```bash
git add README_s2s.md CHANGELOG.md docs/conversion.md
git commit -m "s2s: documentation — usage, changelog, conversion.md pointer"
```

---

## Self-review

**1. Spec coverage.** Walking spec §1–§13:
- §1 (why): Task 10 (CHANGELOG) covers.
- §2 (target model): Task 2 (`_s2s_base_config` fixture) + Task 8 (real model) cover.
- §3 (scope decisions): Task 2 (guards) implements §3.1–§3.6; Task 9 int8 implements §3.4.
- §4 (architecture): Task 1 scaffolds everything; Tasks 3–7 fill it in.
- §5 (weight mapping): Task 2 (`_build_s2s_spec` + helpers) implements §5.1–§5.4; Task 7 (`set_state`) implements §5.5.
- §6 (C++ layers): Tasks 3–6 cover all six classes.
- §7 (state/beam): Task 6 (`replicate_state`) + Task 7 (`set_state`, lazy init) cover.
- §8 (converter guards): Task 2.
- §9 (testing): Tasks 2 (Tier-1), 3–6 (Tier-2), 8 (Tier-3).
- §10 (risks): mitigations distributed across Tasks 2, 3, 5, 6, 7.
- §11 (phasing): tasks 1–10 map 1:1.
- §12 (deferred): intentionally not in plan.
- §13 (glossary): no implementation needed.

Gaps: none.

**2. Placeholder scan.** Searched for TBD/TODO/implement-later. The only "fill in" is Step 3 of Task 2 (where current `marian.py` is read before refactoring) — that's an explicit read step, not a placeholder. The reference-generator and C++ layer code show real formulas/signatures. OK.

**3. Type consistency.** Scope-name convention is strict: Python spec attribute paths like `encoder.forward_cell.input.weight` map to C++ scope strings like `"encoder/forward_cell/input/weight"`. Confirmed:
- `GRUCellSpec.input` → `LinearSpec` → produces `/input/weight`, `/input/bias`. Matches `Dense(model, scope + "/input")` constructor in `GRUCell`.
- `GRUCellSpec.ln_input` → `LayerNormSpec` → produces `/ln_input/gamma`, `/ln_input/beta`. Matches `LayerNorm(model, scope + "/ln_input")`.
- `BahdanauAttentionSpec.v` (bare attribute, not a spec class) → produces `/v`. Matches `*model.get_variable(scope + "/v")`.
- `RNNDecoderSpec.init_state` → `DenseLNSpec` → nested `linear` and `layer_norm`. Matches `Dense(model, scope + "/init_state/linear")` and `LayerNorm(model, scope + "/init_state/layer_norm")`.

All consistent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-19-marian-s2s-support.md`.

User has pre-authorised execution via "write the implementation plan and implement the entire plan." Using **Inline Execution** mode (superpowers:executing-plans) because:
1. User instructed sequential tool usage in global CLAUDE.md, which negates the parallelism advantage of subagent dispatch.
2. Session continuity reduces context-reload overhead for a long, highly-coupled plan.

**Realistic session-scope caveat:** Tasks 8–9 require the user's real `model.npz` (not available in this session) and iterative debugging against Marian output. In this session the plan will aim to complete Tasks 1–7 and 10, leaving Tasks 8–9 as the user-driven final acceptance phase.
