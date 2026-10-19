"""Model specification for Marian s2s (RNN encoder-decoder) models.

Target configuration: single bi-directional GRU encoder, conditional-GRU decoder
(two GRU transitions per step with Bahdanau attention between them), layer-norm
enabled, shared input/output embeddings.

See `README_s2s.md` at the repo root for the end-to-end design.
"""

from ctranslate2.specs import common_spec, model_spec


class GRUCellSpec(model_spec.LayerSpec):
    """A single Marian GRU cell with scale-only layer normalization.

    Weight layout after conversion from Marian's `[W | Wx]`, `[U | Ux]`,
    `[b | bx]` tensors:
      - `input.weight`  : [I, 3H] = concat(W, Wx)  on gate axis [reset|update|candidate]
      - `input.bias`    : [3H]    = concat(b, bx)
      - `recurrent.weight`: [H, 3H] = concat(U, Ux)
      - `recurrent.bias`: [3H] zeros (Marian stores bias only on the input side)
      - `ln_input.gamma`:  Marian `_gamma1` reshaped to 1-D
      - `ln_recurrent.gamma`: Marian `_gamma2` reshaped to 1-D
      - `ln_*.beta`: zeros (Marian layer norm is scale-only)
    """

    def __init__(self, has_input=True, layer_norm=True):
        if has_input:
            self.input = common_spec.LinearSpec()
            if layer_norm:
                self.ln_input = common_spec.LayerNormSpec()
        self.recurrent = common_spec.LinearSpec()
        if layer_norm:
            self.ln_recurrent = common_spec.LayerNormSpec()


class BahdanauAttentionSpec(model_spec.LayerSpec):
    """Marian additive (MLP) attention.

    The computation is `v^T · tanh(LN(Wa·s + 0) + LN(Wc·c + b_att))`.
    - `query`/`memory` are Dense layers (no affine bias; `b_att` is used as the
      LN *shift* on the context side, hence `ln_context.beta`).
    - `v` is the attention projection vector (last dense; stored as 1-D).
    """

    def __init__(self):
        self.query = common_spec.LinearSpec()
        self.memory = common_spec.LinearSpec()
        self.v = None
        self.ln_context = common_spec.LayerNormSpec()
        self.ln_state = common_spec.LayerNormSpec()


class CGRUDecoderCellSpec(model_spec.LayerSpec):
    """Conditional GRU: GRU1 (prev → s') → attention(s', memory) → GRU2(context → s)."""

    def __init__(self):
        self.gru1 = GRUCellSpec(has_input=True, layer_norm=True)
        self.attention = BahdanauAttentionSpec()
        self.gru2 = GRUCellSpec(has_input=True, layer_norm=True)


class DenseLNSpec(model_spec.LayerSpec):
    """Affine (`W`, `b`) followed by scale-only layer norm.

    Used by Marian's `ff_state` (init-state MLP) and `ff_logit_l1`
    (deep-output hidden layer).
    """

    def __init__(self):
        self.linear = common_spec.LinearSpec()
        self.layer_norm = common_spec.LayerNormSpec()


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

    def get_source_vocabulary_size(self):
        return [self.encoder.embeddings.weight.shape[0]]

    def get_target_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]
