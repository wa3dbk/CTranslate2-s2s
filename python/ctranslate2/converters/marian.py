import argparse
import re

from typing import List

import numpy as np
import yaml

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, rnn_spec, transformer_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELUSigmoid,
    "relu": common_spec.Activation.RELU,
    "swish": common_spec.Activation.SWISH,
}

_SUPPORTED_POSTPROCESS_EMB = {"", "d", "n", "nd"}


class MarianConverter(Converter):
    """Converts models trained with Marian."""

    def __init__(self, model_path: str, vocab_paths: List[str]):
        """Initializes the Marian converter.

        Arguments:
          model_path: Path to the Marian model (.npz file).
          vocab_paths: Paths to the vocabularies (.yml files).
        """
        self._model_path = model_path
        self._vocab_paths = vocab_paths

    def _load(self):
        model = np.load(self._model_path)
        config = _get_model_config(model)
        vocabs = list(map(load_vocab, self._vocab_paths))

        model_type = config.get("type")
        if model_type == "s2s":
            return _build_s2s_spec(config, model, vocabs)
        elif model_type != "transformer":
            raise NotImplementedError(
                "Marian model type %r is not supported; "
                "supported types: 'transformer', 's2s'" % model_type
            )
        return _build_transformer_spec(config, model, vocabs)


def _build_transformer_spec(config, model, vocabs):
    activation = config["transformer-ffn-activation"]
    pre_norm = "n" in config["transformer-preprocess"]
    postprocess_emb = config["transformer-postprocess-emb"]

    check = utils.ConfigurationChecker()
    check(config["type"] == "transformer", "Option --type must be 'transformer'")
    check(
        config["transformer-decoder-autoreg"] == "self-attention",
        "Option --transformer-decoder-autoreg must be 'self-attention'",
    )
    check(
        not config["transformer-no-projection"],
        "Option --transformer-no-projection is not supported",
    )
    check(
        activation in _SUPPORTED_ACTIVATIONS,
        "Option --transformer-ffn-activation %s is not supported "
        "(supported activations are: %s)"
        % (activation, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
    )
    check(
        postprocess_emb in _SUPPORTED_POSTPROCESS_EMB,
        "Option --transformer-postprocess-emb %s is not supported (supported values are: %s)"
        % (postprocess_emb, ", ".join(_SUPPORTED_POSTPROCESS_EMB)),
    )

    if pre_norm:
        check(
            config["transformer-preprocess"] == "n"
            and config["transformer-postprocess"] == "da"
            and config.get("transformer-postprocess-top", "") == "n",
            "Unsupported pre-norm Transformer architecture, expected the following "
            "combination of options: "
            "--transformer-preprocess n "
            "--transformer-postprocess da "
            "--transformer-postprocess-top n",
        )
    else:
        check(
            config["transformer-preprocess"] == ""
            and config["transformer-postprocess"] == "dan"
            and config.get("transformer-postprocess-top", "") == "",
            "Unsupported post-norm Transformer architecture, excepted the following "
            "combination of options: "
            "--transformer-preprocess '' "
            "--transformer-postprocess dan "
            "--transformer-postprocess-top ''",
        )

    check.validate()

    alignment_layer = config["transformer-guided-alignment-layer"]
    alignment_layer = -1 if alignment_layer == "last" else int(alignment_layer) - 1
    layernorm_embedding = "n" in postprocess_emb

    model_spec = transformer_spec.TransformerSpec.from_config(
        (config["enc-depth"], config["dec-depth"]),
        config["transformer-heads"],
        pre_norm=pre_norm,
        activation=_SUPPORTED_ACTIVATIONS[activation],
        alignment_layer=alignment_layer,
        alignment_heads=1,
        layernorm_embedding=layernorm_embedding,
    )
    set_transformer_spec(model_spec, model)
    model_spec.register_source_vocabulary(vocabs[0])
    model_spec.register_target_vocabulary(vocabs[-1])
    model_spec.config.add_source_eos = True
    return model_spec


# ----------------------------------------------------------------------------
# Marian s2s (RNN encoder-decoder) builder.
#
# This path targets a single configuration: single bi-GRU encoder, cGRU decoder
# (`dec-cell-base-depth: 2`), layer-norm on, `tied-embeddings-all: True`.
# Anything outside this set is rejected at convert time with a `ValueError`
# naming the offending option. See `README_s2s.md` for the full scope.
# ----------------------------------------------------------------------------


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
    # Marian treats absent boolean options as their default (False for every
    # key in _S2S_HARD_GUARDS whose expected value is False). Older yaml
    # headers omit these keys entirely, so `None` must match `False`.
    if actual is None and expected is False:
        return
    if actual != expected:
        raise ValueError(
            "Marian s2s converter: option %r = %r is not supported "
            "(this converter currently requires %r = %r). "
            "File an issue with your model.npz.yml if you need this variant."
            % (key, actual, key, expected)
        )


def _s2s_require_empty_list(config, key):
    actual = config.get(key, [])
    if actual not in ([], None):
        raise ValueError(
            "Marian s2s converter: option %r = %r is not supported "
            "(this converter requires it to be empty). "
            "File an issue with your model.npz.yml if you need this variant."
            % (key, actual)
        )


def _s2s_gate_concat(np_u, np_ux):
    """Concat `[W | Wx]` on the last axis preserving gate order
    `[reset | update | candidate]`, then transpose to CT2's `[out, in]`.

    Marian stores GRU weights as `[in, out]`: `_U` is `[H, 2H] = [reset | update]`
    and `_Ux` is `[H, H] = candidate`. CT2's `Dense` expects `[out, in]`, so the
    result is transposed to `[3H, in]`.
    """
    return np.concatenate([np_u, np_ux], axis=-1).T


def _s2s_bias_concat(np_b, np_bx):
    """Marian stores GRU bias as `_b` ([1, 2H]) and `_bx` ([1, H]). CTranslate2
    `LinearSpec` stores a 1-D bias. Returns a flat `[3H]` array."""
    return np.concatenate([np_b.reshape(-1), np_bx.reshape(-1)], axis=0)


def _s2s_load_cell(spec_cell, model, prefix, dim_rnn, final=False):
    """Populate a `GRUCellSpec` from Marian tensors under `<prefix>_...`.

    `final` selects between Marian's two cGRU kernel branches. Both branches
    share the parameter layout; they differ only in whether the candidate bias
    `bx` is reset-gated:

      non-final:  h = tanh(x@W_c + s@U_c * r + bx)
      final:      h = tanh(x@W_c + (s@U_c + bx) * r)

    Marian's vanilla `GRU` cell (which this converter targets) adds the whole
    bias *after* the LN inside the fused `gruOps` kernel, so we can reproduce
    the final formula without any kernel change by routing `bx` from the input
    side to the recurrent side: the candidate pre-activation then becomes
    `cx + r * cs = x@W_c_ln + r * (s@U_c_ln + bx)` — bit-exact Marian final.
    The r/z pre-activations are a plain sum of the two LN'd projections, so
    their bias routing is commutative.

    Marian wires the decoder's cGRU as GRU1 (non-final) → GRU2 (final).
    """
    H = dim_rnn
    U = model["%s_U" % prefix]
    Ux = model["%s_Ux" % prefix]
    if U.shape[-1] != 2 * H:
        raise ValueError(
            "Marian s2s converter: tensor %s_U has last-axis size %d, expected %d"
            % (prefix, U.shape[-1], 2 * H)
        )
    if Ux.shape[-1] != H:
        raise ValueError(
            "Marian s2s converter: tensor %s_Ux has last-axis size %d, expected %d"
            % (prefix, Ux.shape[-1], H)
        )
    spec_cell.recurrent.weight = _s2s_gate_concat(U, Ux).astype(np.float32)
    spec_cell.ln_recurrent.gamma = model["%s_gamma2" % prefix].reshape(-1).astype(np.float32)

    W = model["%s_W" % prefix]
    Wx = model["%s_Wx" % prefix]
    b = model["%s_b" % prefix].reshape(-1).astype(np.float32)
    bx = model["%s_bx" % prefix].reshape(-1).astype(np.float32)
    spec_cell.input.weight = _s2s_gate_concat(W, Wx).astype(np.float32)
    # Marian's layer-normalized Dense is `layerNorm(dot(x, W), gamma, b)` — the
    # bias is the LN *shift*, not an affine Dense bias. Store it on ln_input.beta
    # and leave input.bias as zeros so CT2 computes the exact Marian math.
    spec_cell.input.bias = np.zeros(3 * H, dtype=np.float32)
    spec_cell.ln_input.gamma = model["%s_gamma1" % prefix].reshape(-1).astype(np.float32)
    if final:
        # bias on input side carries only (b_r, b_z); candidate bias moves to
        # the recurrent side so the r*cs multiply gates it (final formula).
        spec_cell.ln_input.beta = np.concatenate(
            [b, np.zeros(H, dtype=np.float32)]).astype(np.float32)
        spec_cell.ln_recurrent.beta = np.concatenate(
            [np.zeros(2 * H, dtype=np.float32), bx]).astype(np.float32)
    else:
        spec_cell.ln_input.beta = np.concatenate([b, bx]).astype(np.float32)
        spec_cell.ln_recurrent.beta = np.zeros(3 * H, dtype=np.float32)


def _s2s_load_attention(spec_attn, model, prefix, dim_enc):
    """Populate a `BahdanauAttentionSpec`. `b_att` is stored as the LN *shift*
    on the context side (`ln_context.beta`), not as an affine bias — Marian
    applies it inside `layerNorm(dot(context, Ua_), gammaContext_, ba_)`."""
    # Marian stores these as `[in, out]`; CT2 Dense expects `[out, in]`.
    spec_attn.query.weight = model["%s_W_comb_att" % prefix].T.astype(np.float32)
    spec_attn.memory.weight = model["%s_Wc_att" % prefix].T.astype(np.float32)
    spec_attn.v = model["%s_U_att" % prefix].reshape(-1).astype(np.float32)
    spec_attn.ln_context.gamma = model["%s_att_gamma1" % prefix].reshape(-1).astype(np.float32)
    spec_attn.ln_context.beta = model["%s_b_att" % prefix].reshape(-1).astype(np.float32)
    spec_attn.ln_state.gamma = model["%s_att_gamma2" % prefix].reshape(-1).astype(np.float32)
    spec_attn.ln_state.beta = np.zeros(dim_enc, dtype=np.float32)


def _s2s_load_dense_ln(spec_dense_ln, model, prefix):
    """Populate a `DenseLNSpec` from `<prefix>_W/_b/_gamma`.

    Marian stores `_W` as `[in, out]`; CT2 Dense expects `[out, in]`.

    Marian's layer-normalized Dense is `layerNorm(dot(x, W), gamma, b)` — `b` is
    the LN *shift*, not an affine Dense bias. We store `b` on
    `layer_norm.beta` and leave `linear.bias` as zeros so CT2's
    `LayerNorm(gamma * (x - mean)/sigma + beta)` reproduces Marian's exact
    math; the alternative (b on linear.bias) would subtract `mean(b)` and
    inject the zero-mean projection of `b` into the pre-norm tensor, which is
    not equivalent.
    """
    weight = model["%s_W" % prefix].T.astype(np.float32)
    bias = model["%s_b" % prefix].reshape(-1).astype(np.float32)
    gamma = model["%s_gamma" % prefix].reshape(-1).astype(np.float32)
    spec_dense_ln.linear.weight = weight
    spec_dense_ln.linear.bias = np.zeros_like(bias)
    spec_dense_ln.layer_norm.gamma = gamma
    spec_dense_ln.layer_norm.beta = bias


def _s2s_load_dense_ln_indexed(spec_dense_ln, model, prefix, index):
    """Variant of `_s2s_load_dense_ln` for Marian's multi-input `mlp::dense`
    with layer-normalization enabled: tensors are stored under
    `<prefix>_W{i}` / `<prefix>_b{i}` / `<prefix>_gamma{i}` (one sub-affine
    per input of the parent `mlp::dense`)."""
    weight = model["%s_W%d" % (prefix, index)].T.astype(np.float32)
    bias = model["%s_b%d" % (prefix, index)].reshape(-1).astype(np.float32)
    gamma = model["%s_gamma%d" % (prefix, index)].reshape(-1).astype(np.float32)
    spec_dense_ln.linear.weight = weight
    spec_dense_ln.linear.bias = np.zeros_like(bias)
    spec_dense_ln.layer_norm.gamma = gamma
    spec_dense_ln.layer_norm.beta = bias


def _s2s_load_deep_output(spec_deep, model, prefix):
    """Populate a `DeepOutputSpec` from Marian's `ff_logit_l1` sub-affines.

    Marian builds the deep-output hidden layer as
    `tanh(sum_i LN(W_i @ x_i, gamma_i, b_i))` with inputs in order
    `(embeddings, decoderStates, alignedContext)`. Each `_i` is stored as
    `<prefix>_W{i}` / `<prefix>_b{i}` / `<prefix>_gamma{i}`.
    """
    for sub_name, i in (("proj_embed", 0), ("proj_state", 1), ("proj_context", 2)):
        _s2s_load_dense_ln_indexed(getattr(spec_deep, sub_name), model, prefix, i)


def _s2s_expected_tensor_names():
    """The exact set of NPZ tensor names this converter consumes."""
    names = {"Wemb", "special:model.yml"}
    for prefix in ("encoder_bi", "encoder_bi_r", "decoder_cell1", "decoder_cell2"):
        for sfx in ("_U", "_Ux", "_W", "_Wx", "_b", "_bx", "_gamma1", "_gamma2"):
            names.add(prefix + sfx)
    for sfx in ("_W_comb_att", "_Wc_att", "_U_att", "_b_att",
                "_att_gamma1", "_att_gamma2"):
        names.add("decoder" + sfx)
    for sfx in ("_W", "_b", "_gamma"):
        names.add("decoder_ff_state" + sfx)
    # ff_logit_l1 has one Dense sub-affine per input (embeddings, hidden,
    # context) when layer-normalization is enabled.
    for i in range(3):
        for sfx in ("_W%d" % i, "_b%d" % i, "_gamma%d" % i):
            names.add("decoder_ff_logit_l1" + sfx)
    names.add("decoder_ff_logit_l2_b")
    return names


_S2S_IGNORABLE_TENSOR_NAMES = {
    # Non-weight metadata keys that may appear in the NPZ.
    "special:model.yml",
    # Training-only state kept for backwards compatibility.
    "Wpos",
}


def _s2s_validate_tensor_set(model):
    expected = _s2s_expected_tensor_names()
    present = set(model.files)
    missing = expected - present
    if missing:
        raise ValueError(
            "Marian s2s converter: missing expected tensor(s): %s" % sorted(missing)
        )
    extra = present - expected - _S2S_IGNORABLE_TENSOR_NAMES
    if extra:
        raise ValueError(
            "Marian s2s converter: unused tensor(s) in NPZ (unknown to this "
            "converter): %s" % sorted(extra)
        )
    # Reject corrupted model files up front: NaN/Inf in any expected weight
    # propagates silently through conversion and only surfaces as garbage
    # output at inference time.
    bad = []
    for name in sorted(expected):
        arr = model[name]
        if arr.dtype.kind != "f":
            continue
        if np.isnan(arr).any() or np.isinf(arr).any():
            bad.append(name)
    if bad:
        raise ValueError(
            "Marian s2s converter: tensor(s) contain NaN or Inf values; "
            "the model file is corrupted or came from a diverged training "
            "run: %s" % bad
        )


def _build_s2s_spec(config, model, vocabs):
    for key, expected in _S2S_HARD_GUARDS.items():
        _s2s_require(config, key, expected)
    _s2s_require_empty_list(config, "input-types")

    H = config["dim-rnn"]
    I = config["dim-emb"]
    V_src, V_tgt = config["dim-vocabs"][0], config["dim-vocabs"][-1]

    # tied-embeddings-all ties the single Wemb across src/tgt/output in
    # id-space, so the two vocab files only need to agree on size. Their
    # token-to-id mappings may differ per side (common in bilingual setups).
    if V_src != V_tgt:
        raise ValueError(
            "Marian s2s converter: tied-embeddings-all=True requires equal "
            "source and target vocabulary sizes; got %d and %d." % (V_src, V_tgt)
        )

    spec = rnn_spec.RNNModelSpec()

    Wemb = model["Wemb"]
    if Wemb.shape != (V_src, I):
        raise ValueError(
            "Marian s2s converter: Wemb shape %s != expected (%d, %d)"
            % (Wemb.shape, V_src, I)
        )
    Wemb_f32 = Wemb.astype(np.float32)
    spec.encoder.embeddings.weight = Wemb_f32
    spec.decoder.embeddings.weight = Wemb_f32  # tied; aliased by LayerSpec.optimize

    _s2s_load_cell(spec.encoder.forward_cell,  model, "encoder_bi",   dim_rnn=H)
    _s2s_load_cell(spec.encoder.backward_cell, model, "encoder_bi_r", dim_rnn=H)

    _s2s_load_cell(spec.decoder.cell.gru1, model, "decoder_cell1", dim_rnn=H,
                   final=False)
    _s2s_load_cell(spec.decoder.cell.gru2, model, "decoder_cell2", dim_rnn=H,
                   final=True)

    _s2s_load_attention(spec.decoder.cell.attention, model, "decoder", dim_enc=2 * H)

    _s2s_load_dense_ln(spec.decoder.init_state, model, "decoder_ff_state")
    _s2s_load_deep_output(spec.decoder.hidden,  model, "decoder_ff_logit_l1")

    # Deep-output top layer: Wt is tied-transposed to Wemb; only the bias is
    # independent in Marian. The spec stores the tied weight here; LayerSpec
    # aliasing collapses it with the shared embedding at save time.
    spec.decoder.output.weight = Wemb_f32
    spec.decoder.output.bias = model["decoder_ff_logit_l2_b"].reshape(-1).astype(np.float32)

    _s2s_validate_tensor_set(model)

    spec.register_source_vocabulary(vocabs[0])
    spec.register_target_vocabulary(vocabs[-1])
    spec.config.add_source_eos = True
    # Marian s2s decoders are trained to start generation from EOS — the same
    # id used to terminate sequences. The CT2 default `<s>` is not in Marian
    # vocabularies and would be resolved to <unk>, poisoning the first step.
    spec.config.decoder_start_token = "</s>"
    # Marian's LayerNormalization uses eps=1e-9. The betas injected above make
    # CT2 pick its full LayerNorm path (gamma * (x - mean)/sqrt(var + eps))
    # instead of the RMSNorm fallback it uses when beta is absent.
    spec.config.add_attribute("layer_norm_epsilon", 1e-9)
    return spec


def _get_model_config(model):
    config = model["special:model.yml"]
    config = config[:-1].tobytes()
    config = yaml.safe_load(config)
    return config


def load_vocab(path):
    # pyyaml skips some entries so we manually parse the vocabulary file.
    with open(path, encoding="utf-8") as vocab:
        tokens = []
        token = None
        idx = None
        for i, line in enumerate(vocab):
            line = line.rstrip("\n\r")
            if not line:
                continue

            if line.startswith("? "):  # Complex key mapping (key)
                token = line[2:]
            elif token is not None:  # Complex key mapping (value)
                idx = line[2:]
            else:
                token, idx = line.rsplit(":", 1)

            if token is not None:
                if token.startswith('"') and token.endswith('"'):
                    # Unescape characters and remove quotes.
                    token = re.sub(r"\\([^x])", r"\1", token)
                    token = token[1:-1]
                    if token.startswith("\\x"):
                        # Convert the digraph \x to the actual escaped sequence.
                        token = chr(int(token[2:], base=16))
                elif token.startswith("'") and token.endswith("'"):
                    token = token[1:-1]
                    token = token.replace("''", "'")

            if idx is not None:
                try:
                    idx = int(idx.strip())
                except ValueError as e:
                    raise ValueError(
                        "Unexpected format at line %d: '%s'" % (i + 1, line)
                    ) from e

                tokens.append((idx, token))

                token = None
                idx = None

    return [token for _, token in sorted(tokens, key=lambda item: item[0])]


def set_transformer_spec(spec, weights):
    set_transformer_encoder(spec.encoder, weights, "encoder")
    set_transformer_decoder(spec.decoder, weights, "decoder")


def set_transformer_encoder(spec, weights, scope):
    set_common_layers(spec, weights, scope)
    for i, layer_spec in enumerate(spec.layer):
        set_transformer_encoder_layer(layer_spec, weights, "%s_l%d" % (scope, i + 1))


def set_transformer_decoder(spec, weights, scope):
    spec.start_from_zero_embedding = True
    set_common_layers(spec, weights, scope)
    for i, layer_spec in enumerate(spec.layer):
        set_transformer_decoder_layer(layer_spec, weights, "%s_l%d" % (scope, i + 1))

    set_linear(
        spec.projection,
        weights,
        "%s_ff_logit_out" % scope,
        reuse_weight=spec.embeddings.weight,
    )


def set_common_layers(spec, weights, scope):
    embeddings_specs = spec.embeddings
    if not isinstance(embeddings_specs, list):
        embeddings_specs = [embeddings_specs]

    set_embeddings(embeddings_specs[0], weights, scope)
    set_position_encodings(
        spec.position_encodings, weights, dim=embeddings_specs[0].weight.shape[1]
    )
    if hasattr(spec, "layernorm_embedding"):
        set_layer_norm(
            spec.layernorm_embedding,
            weights,
            "%s_emb" % scope,
            pre_norm=True,
        )
    if hasattr(spec, "layer_norm"):
        set_layer_norm(spec.layer_norm, weights, "%s_top" % scope)


def set_transformer_encoder_layer(spec, weights, scope):
    set_ffn(spec.ffn, weights, "%s_ffn" % scope)
    set_multi_head_attention(
        spec.self_attention, weights, "%s_self" % scope, self_attention=True
    )


def set_transformer_decoder_layer(spec, weights, scope):
    set_ffn(spec.ffn, weights, "%s_ffn" % scope)
    set_multi_head_attention(
        spec.self_attention, weights, "%s_self" % scope, self_attention=True
    )
    set_multi_head_attention(spec.attention, weights, "%s_context" % scope)


def set_multi_head_attention(spec, weights, scope, self_attention=False):
    split_layers = [common_spec.LinearSpec() for _ in range(3)]
    set_linear(split_layers[0], weights, scope, "q")
    set_linear(split_layers[1], weights, scope, "k")
    set_linear(split_layers[2], weights, scope, "v")

    if self_attention:
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        spec.linear[0].weight = split_layers[0].weight
        spec.linear[0].bias = split_layers[0].bias
        utils.fuse_linear(spec.linear[1], split_layers[1:])

    set_linear(spec.linear[-1], weights, scope, "o")
    set_layer_norm_auto(spec.layer_norm, weights, "%s_Wo" % scope)


def set_ffn(spec, weights, scope):
    set_layer_norm_auto(spec.layer_norm, weights, "%s_ffn" % scope)
    set_linear(spec.linear_0, weights, scope, "1")
    set_linear(spec.linear_1, weights, scope, "2")


def set_layer_norm_auto(spec, weights, scope):
    try:
        set_layer_norm(spec, weights, scope, pre_norm=True)
    except KeyError:
        set_layer_norm(spec, weights, scope)


def set_layer_norm(spec, weights, scope, pre_norm=False):
    suffix = "_pre" if pre_norm else ""
    spec.gamma = weights["%s_ln_scale%s" % (scope, suffix)].squeeze()
    spec.beta = weights["%s_ln_bias%s" % (scope, suffix)].squeeze()


def set_linear(spec, weights, scope, suffix="", reuse_weight=None):
    weight = weights.get("%s_W%s" % (scope, suffix))

    if weight is None:
        weight = weights.get("%s_Wt%s" % (scope, suffix), reuse_weight)
    else:
        weight = weight.transpose()

    spec.weight = weight

    bias = weights.get("%s_b%s" % (scope, suffix))
    if bias is not None:
        spec.bias = bias.squeeze()


def set_embeddings(spec, weights, scope):
    spec.weight = weights.get("%s_Wemb" % scope)
    if spec.weight is None:
        spec.weight = weights.get("Wemb")


def set_position_encodings(spec, weights, dim=None):
    spec.encodings = weights.get("Wpos", _make_sinusoidal_position_encodings(dim))


def _make_sinusoidal_position_encodings(dim, num_positions=2048):
    positions = np.arange(num_positions)
    timescales = np.power(10000, 2 * (np.arange(dim) // 2) / dim)
    position_enc = np.expand_dims(positions, 1) / np.expand_dims(timescales, 0)
    table = np.zeros_like(position_enc)
    table[:, : dim // 2] = np.sin(position_enc[:, 0::2])
    table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
    return table


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the model .npz file."
    )
    parser.add_argument(
        "--vocab_paths",
        required=True,
        nargs="+",
        help="List of paths to the YAML vocabularies.",
    )
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = MarianConverter(args.model_path, args.vocab_paths)
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
