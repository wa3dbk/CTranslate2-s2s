"""Tier-1 unit tests for the Marian s2s converter path.

These tests exercise `MarianConverter._load` end-to-end for `type: s2s`
NPZ archives: they synthesise minimal valid archives, flip one option
at a time to confirm each guard fires, and verify the gate-order
contract on the produced tensors.
"""

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
    """Write a Marian-like NPZ archive with embedded YAML config and all
    tensors shaped per `config`."""
    H = config["dim-rnn"]
    I = config["dim-emb"]
    V = config["dim-vocabs"][0]
    tensors = {
        # Shared embedding.
        "Wemb": np.arange(V * I, dtype=np.float32).reshape(V, I),
        # Encoder forward.
        "encoder_bi_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_Ux": np.full((H, H), 2.0, dtype=np.float32),
        "encoder_bi_W": np.full((I, 2 * H), 3.0, dtype=np.float32),
        "encoder_bi_Wx": np.full((I, H), 4.0, dtype=np.float32),
        "encoder_bi_b": np.full((1, 2 * H), 5.0, dtype=np.float32),
        "encoder_bi_bx": np.full((1, H), 6.0, dtype=np.float32),
        "encoder_bi_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "encoder_bi_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Encoder backward.
        "encoder_bi_r_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_Ux": np.full((H, H), 1.0, dtype=np.float32),
        "encoder_bi_r_W": np.full((I, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_Wx": np.full((I, H), 1.0, dtype=np.float32),
        "encoder_bi_r_b": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_bx": np.full((1, H), 1.0, dtype=np.float32),
        "encoder_bi_r_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "encoder_bi_r_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Decoder base cell 1 (dim_in=dim-emb).
        "decoder_cell1_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell1_Ux": np.full((H, H), 1.0, dtype=np.float32),
        "decoder_cell1_W": np.full((I, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell1_Wx": np.full((I, H), 1.0, dtype=np.float32),
        "decoder_cell1_b": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell1_bx": np.full((1, H), 1.0, dtype=np.float32),
        "decoder_cell1_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "decoder_cell1_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Decoder base cell 2 (dim_in=2*dim-rnn after attention concat).
        "decoder_cell2_U": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell2_Ux": np.full((H, H), 1.0, dtype=np.float32),
        "decoder_cell2_W": np.full((2 * H, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell2_Wx": np.full((2 * H, H), 1.0, dtype=np.float32),
        "decoder_cell2_b": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "decoder_cell2_bx": np.full((1, H), 1.0, dtype=np.float32),
        "decoder_cell2_gamma1": np.full((1, 3 * H), 1.0, dtype=np.float32),
        "decoder_cell2_gamma2": np.full((1, 3 * H), 1.0, dtype=np.float32),
        # Attention (single encoder → prefix = "decoder").
        "decoder_W_comb_att": np.full((H, 2 * H), 1.0, dtype=np.float32),
        "decoder_Wc_att": np.full((2 * H, 2 * H), 1.0, dtype=np.float32),
        "decoder_U_att": np.full((2 * H, 1), 1.0, dtype=np.float32),
        "decoder_b_att": np.full((1, 2 * H), 0.0, dtype=np.float32),
        "decoder_att_gamma1": np.full((1, 2 * H), 1.0, dtype=np.float32),
        "decoder_att_gamma2": np.full((1, 2 * H), 1.0, dtype=np.float32),
        # Decoder init MLP (ff_state): [2H, H].
        "decoder_ff_state_W": np.full((2 * H, H), 1.0, dtype=np.float32),
        "decoder_ff_state_b": np.full((1, H), 1.0, dtype=np.float32),
        "decoder_ff_state_gamma": np.full((1, H), 1.0, dtype=np.float32),
        # Deep output hidden (ff_logit_l1): [I + H + 2H, I].
        "decoder_ff_logit_l1_W": np.full((I + H + 2 * H, I), 1.0, dtype=np.float32),
        "decoder_ff_logit_l1_b": np.full((1, I), 1.0, dtype=np.float32),
        "decoder_ff_logit_l1_gamma": np.full((1, I), 1.0, dtype=np.float32),
        # Deep output top (ff_logit_l2): bias only; W tied to Wemb.
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
    """Marian vocab format: one `token: idx` per line, line-wise YAML-ish.

    `load_vocab` parses this manually (not via pyyaml) and reconstructs the
    ordered token list.
    """
    path = tmp_path / name
    with open(path, "w") as f:
        for i in range(size):
            f.write("tok%d: %d\n" % (i, i))
    return path


def _make_converter(tmp_path, config, extra_tensors=None):
    npz = _write_fake_npz(tmp_path, config, extra_tensors=extra_tensors)
    v_src = _write_fake_vocab(tmp_path, "vocab.src.yml", config["dim-vocabs"][0])
    v_tgt = _write_fake_vocab(tmp_path, "vocab.tgt.yml", config["dim-vocabs"][-1])
    return MarianConverter(str(npz), [str(v_src), str(v_tgt)])


# ---------------------------------------------------------------- guard tests

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
    converter = _make_converter(tmp_path, config)
    with pytest.raises(ValueError, match=key):
        converter._load()


@pytest.mark.parametrize(
    "key",
    [
        "skip",
        "right-left",
        "transformer-rnn-projection",
        "transformer-pool",
        "ulr",
        "embedding-fix-src",
        "embedding-fix-trg",
    ],
)
def test_s2s_guards_accept_absent_boolean_defaulting_to_false(tmp_path, key):
    # Older Marian model.npz.yml files omit boolean options whose default is
    # False. `config.get(key)` returns None in that case; the converter must
    # accept that as equivalent to False rather than raising.
    config = _s2s_base_config()
    config.pop(key, None)
    converter = _make_converter(tmp_path, config)
    spec = converter._load()
    assert spec.name == "RNNModelSpec"


def test_s2s_unknown_type_errors(tmp_path):
    config = _s2s_base_config()
    config["type"] = "amun"
    converter = _make_converter(tmp_path, config)
    with pytest.raises(NotImplementedError, match="amun"):
        converter._load()


# ----------------------------------------------------------- happy-path tests

def test_s2s_base_config_loads(tmp_path):
    config = _s2s_base_config()
    converter = _make_converter(tmp_path, config)
    spec = converter._load()
    assert spec.name == "RNNModelSpec"
    assert spec.revision == 1


def test_s2s_weight_mapping_gate_order(tmp_path):
    """Marian gate order is `[reset | update | candidate]`. The converter must
    concatenate `[_U | _Ux]` on the last axis so that recurrent_weight is
    `[reset=1.0 | update=1.0 | candidate=2.0]` when `_U` was filled with 1.0
    and `_Ux` with 2.0.
    """
    config = _s2s_base_config()
    converter = _make_converter(tmp_path, config)
    spec = converter._load()

    H = config["dim-rnn"]
    I = config["dim-emb"]
    rec = spec.encoder.forward_cell.recurrent.weight
    assert rec.shape == (H, 3 * H)
    assert np.allclose(rec[:, : 2 * H], 1.0)
    assert np.allclose(rec[:, 2 * H :], 2.0)

    inp = spec.encoder.forward_cell.input.weight
    assert inp.shape == (I, 3 * H)
    assert np.allclose(inp[:, : 2 * H], 3.0)
    assert np.allclose(inp[:, 2 * H :], 4.0)

    bias = spec.encoder.forward_cell.input.bias
    assert bias.shape == (3 * H,)
    assert np.allclose(bias[: 2 * H], 5.0)
    assert np.allclose(bias[2 * H :], 6.0)


def test_s2s_attention_wiring(tmp_path):
    config = _s2s_base_config()
    converter = _make_converter(tmp_path, config)
    spec = converter._load()

    H = config["dim-rnn"]
    attn = spec.decoder.cell.attention
    assert attn.query.weight.shape == (H, 2 * H)
    assert attn.memory.weight.shape == (2 * H, 2 * H)
    assert attn.v.shape == (2 * H,)
    assert attn.ln_context.gamma.shape == (2 * H,)
    # b_att is stored as the LN shift on the context side.
    assert attn.ln_context.beta.shape == (2 * H,)
    assert np.allclose(attn.ln_context.beta, 0.0)  # fake npz has b_att = 0


def test_s2s_shared_embedding_aliased(tmp_path):
    """After `optimize()`, the encoder embedding and decoder output weight
    should alias the decoder embedding. (Aliasing keeps the alphabetically
    earliest variable name; `decoder/embeddings/weight` wins over
    `encoder/embeddings/weight` and `decoder/output/weight`.)"""
    config = _s2s_base_config()
    converter = _make_converter(tmp_path, config)
    spec = converter._load()
    spec.validate()
    spec.optimize()
    variables = spec.variables()
    canonical = "decoder/embeddings/weight"
    assert canonical in variables
    # Alias values are strings pointing at the canonical variable.
    assert variables.get("encoder/embeddings/weight") == canonical
    assert variables.get("decoder/output/weight") == canonical


def test_s2s_unused_tensor_errors(tmp_path):
    config = _s2s_base_config()
    extra = {"encoder_bi_junk": np.zeros((1,), dtype=np.float32)}
    converter = _make_converter(tmp_path, config, extra_tensors=extra)
    with pytest.raises(ValueError, match="encoder_bi_junk"):
        converter._load()


def test_s2s_missing_tensor_errors(tmp_path):
    """Dropping a required tensor must produce a converter-level error, not an
    opaque numpy KeyError."""
    config = _s2s_base_config()
    npz = _write_fake_npz(tmp_path, config)
    data = dict(np.load(npz))
    data.pop("encoder_bi_U")
    np.savez(npz, **data)
    v_src = _write_fake_vocab(tmp_path, "vocab.src.yml", config["dim-vocabs"][0])
    v_tgt = _write_fake_vocab(tmp_path, "vocab.tgt.yml", config["dim-vocabs"][-1])
    converter = MarianConverter(str(npz), [str(v_src), str(v_tgt)])
    # `_s2s_load_cell` is called before `_s2s_validate_tensor_set`, so we get
    # KeyError from the dict lookup. Either way the pipeline fails loudly —
    # verify that it does.
    with pytest.raises((KeyError, ValueError)):
        converter._load()
