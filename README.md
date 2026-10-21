[![CI](https://github.com/OpenNMT/CTranslate2/workflows/CI/badge.svg)](https://github.com/OpenNMT/CTranslate2/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/CTranslate2/) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

# CTranslate2 — Marian `s2s` fork

> **Fork notice.** This repository extends upstream [`OpenNMT/CTranslate2`](https://github.com/OpenNMT/CTranslate2) with support for Marian NMT's RNN sequence-to-sequence models (`type: s2s`): a bi-directional GRU encoder with Bahdanau (additive) attention and a conditional GRU (cGRU) decoder. Upstream CTranslate2 only supports Marian's `transformer` type.

## s2s support status

| Component | Status |
| --- | --- |
| `RNNModelSpec` Python spec (`python/ctranslate2/specs/rnn_spec.py`) | **Done** |
| `RNNModel` factory registration (`src/models/rnn.cc`, `src/models/model_factory.cc`) | **Done** |
| Marian `s2s` converter path (`python/ctranslate2/converters/marian.py`) with hard guards + Tier-1 Python tests | **Done** |
| C++ `GRUCell` forward (scale-only LN, Marian `[r \| u \| c]` gate order) + Tier-2 unit test | **Done** |
| `BahdanauAttention`, `CGRUDecoderCell`, `BiGRUEncoder`, `RNNDecoder` runtime + Tier-2 unit tests | **Done** |

The converter and the full inference path (encoder → attention → decoder → beam search) run today.

## Marian export and inference

Install this fork from source. The Python package is a pybind11 extension that links against `libctranslate2.so`, so you build the C++ library first and point `CTRANSLATE2_ROOT` at it:

```bash
git clone https://github.com/wa3dbk/CTranslate2-s2s.git
cd CTranslate2-s2s
git submodule update --init --recursive

# 1) Build and install the C++ library to a prefix of your choice.
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/ct2" \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON   # pick the BLAS backend available on your box
cmake --build build -j
cmake --install build

# 2) Build the Python bindings against the installed library.
export CTRANSLATE2_ROOT="$HOME/.local/ct2"
export LD_LIBRARY_PATH="$CTRANSLATE2_ROOT/lib:$CTRANSLATE2_ROOT/lib64:$LD_LIBRARY_PATH"
pip install -e python/
```

You **must** enable exactly one CPU BLAS backend or `translate_batch` fails at runtime with `No SGEMM backend on CPU`. Pick one that your box has:

- `-DWITH_MKL=ON` — fastest on Intel; requires MKL to be discoverable.
- `-DWITH_OPENBLAS=ON` — Linux with `libopenblas-dev` installed.
- `-DWITH_DNNL=ON` — Linux with `libdnnl-dev` installed.
- `-DWITH_RUY=ON` — bundled, no system deps, slower than the above.
- `-DWITH_ACCELERATE=ON` — macOS, picked up automatically.

Always pair the chosen backend with `-DWITH_MKL=OFF` if MKL is not what you want (it defaults to `ON`). On CUDA machines also pass `-DWITH_CUDA=ON -DWITH_CUDNN=ON` and make sure `nvcc` is on `PATH`.

The converter CLI `ct2-marian-converter` and the Python entry point `ctranslate2.converters.MarianConverter` dispatch on the Marian config's `type:` field — `transformer` goes through the upstream path, `s2s` goes through this fork's new path. Both call `translate_batch` identically from Python.

Inputs expected by the converter:

- `model.npz` — Marian weights.
- One or two vocabularies as `.yml` (joint `--vocab_paths vocab.yml` or separate `--vocab_paths src.yml tgt.yml`). The vocabulary files must match what Marian used during training.

### Export — CLI

```bash
# Transformer-type Marian (upstream path, fully supported)
ct2-marian-converter \
  --model_path /path/to/model.npz \
  --vocab_paths /path/to/vocab.yml \
  --output_dir ct2_model \
  --quantization int8

# s2s-type Marian (this fork's new path)
ct2-marian-converter \
  --model_path /path/to/s2s_model.npz \
  --vocab_paths /path/to/src.yml /path/to/tgt.yml \
  --output_dir ct2_s2s_model
```

Common `--quantization` values: `int8`, `int8_float16`, `int16`, `float16`, `bfloat16`, `float32`. Add `--force` to overwrite an existing `--output_dir`.

### Export — Python

```python
from ctranslate2.converters import MarianConverter

converter = MarianConverter(
    model_path="/path/to/model.npz",
    vocab_paths=["/path/to/src.yml", "/path/to/tgt.yml"],
)
converter.convert(output_dir="ct2_model", quantization="int8", force=True)
```

### Inference (transformer-type)

Marian tokenization is **not** performed by CTranslate2 — pre-tokenize with the same SentencePiece/BPE model Marian was trained with, strip or keep BOS/EOS to match the config, then pass token strings to `translate_batch`:

```python
import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("ct2_model", device="cpu")  # or device="cuda"
sp = spm.SentencePieceProcessor(model_file="/path/to/sentencepiece.model")

sources = ["Hello world!", "This is a test."]
source_tokens = [sp.encode(s, out_type=str) for s in sources]

results = translator.translate_batch(
    source_tokens,
    beam_size=5,
    max_decoding_length=256,
)

for src, result in zip(sources, results):
    hypothesis = result.hypotheses[0]
    print(src, "=>", sp.decode(hypothesis))
```

### Inference (s2s-type)

The s2s path uses the same `ctranslate2.Translator` API as the transformer path. Pre-tokenize with the same tokenizer Marian was trained with — CTranslate2 does not tokenize. Marian recipes typically use **subword-nmt** BPE (text codes starting with `#version:`); some use **SentencePiece** (binary `.model` protobuf).

Match your `beam_size` and `length_penalty` to the Marian config's `beam-size` and `normalize` values for parity.

#### subword-nmt BPE (most Marian recipes)

```bash
pip install subword-nmt
```

```python
import ctranslate2
from subword_nmt.apply_bpe import BPE

translator = ctranslate2.Translator("ct2_s2s_model", device="cpu")  # or device="cuda"

with open("/path/to/arfr.bpe", encoding="utf-8") as f:
    bpe = BPE(f)

sources = ["source line 1", "source line 2"]
source_tokens = [bpe.process_line(s).split() for s in sources]

results = translator.translate_batch(
    source_tokens,
    beam_size=12,          # match Marian's beam-size
    length_penalty=1.0,    # match Marian's normalize
    max_decoding_length=256,
)

for src, result in zip(sources, results):
    hyp = result.hypotheses[0]
    # subword-nmt detokenization: strip trailing "@@" from each piece then join.
    detok = " ".join(hyp).replace("@@ ", "").replace("@@", "")
    print(src, "=>", detok)
```

#### SentencePiece

```python
import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("ct2_s2s_model", device="cpu")
sp = spm.SentencePieceProcessor(model_file="/path/to/sentencepiece.model")

sources = ["source line 1", "source line 2"]
source_tokens = [sp.encode(s, out_type=str) for s in sources]

results = translator.translate_batch(source_tokens, beam_size=12, length_penalty=1.0)
for src, result in zip(sources, results):
    print(src, "=>", sp.decode(result.hypotheses[0]))
```

To compare against the Marian reference decoder on the same inputs, run `marian-decoder` with the original `model.npz` and the same beam size, then diff the hypotheses:

```bash
# Marian reference
marian-decoder -m /path/to/s2s_model.npz \
  -v /path/to/src.yml /path/to/tgt.yml \
  --beam-size 5 \
  < sources.tok.txt > hyps.marian.txt

# CTranslate2
python -c "
import ctranslate2, sys
t = ctranslate2.Translator('ct2_s2s_model', device='cpu')
for line in sys.stdin:
    out = t.translate_batch([line.split()], beam_size=5)[0].hypotheses[0]
    print(' '.join(out))
" < sources.tok.txt > hyps.ct2.txt

diff hyps.marian.txt hyps.ct2.txt
```

### Benchmarking against `marian-decoder`

A fair CPU benchmark requires pinning both runtimes to the same core budget and matching numeric settings. Thread-matrix parity rule: `(parallel-batches) × (BLAS-threads-per-batch) ≤ physical cores`.

The two runtimes expose this differently:

| Setting | Marian | CTranslate2 |
|---|---|---|
| Parallel batches | `--cpu-threads N` (one worker thread per N; MKL stays single-threaded inside each) | `--inter_threads N` |
| BLAS threads per batch | *(not exposed; always 1 when `--cpu-threads > 0`)* | `--intra_threads N` |
| Compute precision | fp32 (default) | `--compute_type float32` |

On a 4-physical-core box, the apples-to-apples pair is:

- `marian-decoder --cpu-threads 4`
- `ct2-translator --inter_threads 4 --intra_threads 1 --compute_type float32`

CT2 can also run **one** batch at a time with BLAS across all cores (`--inter_threads 1 --intra_threads 4`). That configuration is not expressible in Marian and typically wins on small batches — report it as a second CT2 row, not as the apples-to-apples number.

Beam search, length penalty, and vocabulary handling must match or BLEU differences become uninterpretable:

| Knob | Marian | CTranslate2 |
|---|---|---|
| Beam size | `--beam-size 12` | `--beam_size 12` |
| Length normalisation | `--normalize 1.0` (= divide score by `len^1`) | `--length_penalty 1.0` (same convention) |
| Disallow `<unk>` | `--allow-unk false` (default) | `--disable_unk` |
| Mini-batch | `--mini-batch M --maxi-batch M --maxi-batch-sort src` | `--max_batch_size M` + `--batch_type examples` |

CT2's `--inter_threads` and `--intra_threads` default to the number of physical cores, which will double-count against Marian's `--cpu-threads`. Always set both explicitly in a benchmark.

**Environment hygiene.** Both runtimes link against MKL / OpenMP and will silently obey process-wide thread caps. Before every run:

```bash
unset MKL_NUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS
```

Otherwise `source /opt/intel/oneapi/mkl/latest/env/vars.sh` (needed to compile Marian with `-DUSE_STATIC_LIBS=on -DCOMPILE_CPU=on`) can leave `MKL_NUM_THREADS` set in the shell and fight the per-tool flags. The static MKL link affects compile-time only; at run-time MKL still obeys environment and per-call thread settings.

**Metrics worth reporting.** BLEU on a held-out test set with `sacrebleu` and identical detokenisation on both sides; wall-clock throughput (output tokens/s) at a few batch sizes (1, 32, 64) measured after a warm-up run; peak RSS (`/usr/bin/time -l` on macOS, `/usr/bin/time -v` on Linux). A clean report is a 3-row table — Marian baseline, CT2 apples-to-apples, CT2 best — plus an optional 4th `--compute_type int8` row labelled separately (Marian has no equivalent switch).

**Honest caveats.** CT2's beam search is not bit-identical to Marian's at tie-breaks, so a 0.1–0.5 BLEU residual between the two runtimes on the same model is expected and not a bug. Int8 scale calibration is embedding-table-wide; models with extreme embedding ranges may show slightly larger BLEU drops under `--compute_type int8` than Transformer models.

### Design spec

The architecture, weight mapping, converter guards, state-handling choices, and testing tiers for the s2s path live in [`README_s2s.md`](README_s2s.md). Read that if you're adding LSTM / `gru-nematus` support, relaxing a converter guard, extending alignment extraction, or wiring up CUDA fp16 for s2s.

Everything below this notice is upstream CTranslate2 documentation and applies unchanged (transformer paths are untouched).

---

CTranslate2 is a C++ and Python library for efficient inference with Transformer models.

The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to [accelerate and reduce the memory usage](#benchmarks) of Transformer models on CPU and GPU.

The following model types are currently supported:

* Encoder-decoder models: Transformer base/big, M2M-100, NLLB, BART, mBART, Pegasus, T5, UMT5, Whisper, T5Gemma, T5Gemma2, MADLAD-400
* Decoder-only models: GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, Gemma, CodeGen, GPTBigCode, Falcon, Qwen2
* Encoder-only models: BERT, DistilBERT, XLM-RoBERTa

Compatible models should be first converted into an optimized model format. The library includes converters for multiple frameworks:

* [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
* [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
* [Fairseq](https://opennmt.net/CTranslate2/guides/fairseq.html)
* [Marian](https://opennmt.net/CTranslate2/guides/marian.html)
* [OPUS-MT](https://opennmt.net/CTranslate2/guides/opus_mt.html)
* [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

The project is production-oriented and comes with [backward compatibility guarantees](https://opennmt.net/CTranslate2/versioning.html), but it also includes experimental features related to model compression and inference acceleration.

## Key features

* **Fast and efficient execution on CPU and GPU**<br/>The execution [is significantly faster and requires less resources](#benchmarks) than general-purpose deep learning frameworks on supported models and tasks thanks to many advanced optimizations: layer fusion, padding removal, batch reordering, in-place operations, caching mechanism, etc.
* **Quantization and reduced precision**<br/>The model serialization and computation support weights with [reduced precision](https://opennmt.net/CTranslate2/quantization.html): 16-bit floating points (FP16), 16-bit brain floating points (BF16), 16-bit integers (INT16), 8-bit integers (INT8) and AWQ quantization (INT4).
* **Multiple CPU architectures support**<br/>The project supports x86-64 and AArch64/ARM64 processors and integrates multiple backends that are optimized for these platforms: [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://github.com/oneapi-src/oneDNN), [OpenBLAS](https://www.openblas.net/), [Ruy](https://github.com/google/ruy), and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).
* **Automatic CPU detection and code dispatch**<br/>One binary can include multiple backends (e.g. Intel MKL and oneDNN) and instruction set architectures (e.g. AVX, AVX2) that are automatically selected at runtime based on the CPU information.
* **Parallel and asynchronous execution**<br/>Multiple batches can be processed in parallel and asynchronously using multiple GPUs or CPU cores.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Lightweight on disk**<br/>Quantization can make the models 4 times smaller on disk with minimal accuracy loss.
* **Simple integration**<br/>The project has few dependencies and exposes simple APIs in [Python](https://opennmt.net/CTranslate2/python/overview.html) and C++ to cover most integration needs.
* **Configurable and interactive decoding**<br/>[Advanced decoding features](https://opennmt.net/CTranslate2/decoding.html) allow autocompleting a partial sequence and returning alternatives at a specific location in the sequence.
* **Support tensor parallelism for distributed inference**<br/>Very large model can be split into multiple GPUs. Following this [documentation](docs/parallel.md#model-and-tensor-parallelism) to set up the required environment.

Some of these features are difficult to achieve with standard deep learning frameworks and are the motivation for this project.

## Installation and usage

CTranslate2 can be installed with pip:

```bash
pip install ctranslate2
```

The Python module is used to convert models and can translate or generate text with few lines of code:

```python
translator = ctranslate2.Translator(translation_model_path)
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
```

See the [documentation](https://opennmt.net/CTranslate2) for more information and examples.

If you have an AMD ROCm GPU, we provide specific Python wheels on the [releases page](https://github.com/OpenNMT/CTranslate2/releases/).

## Web Server

[ctranslate2-web-server](https://github.com/jordimas/ctranslate2-web-server) is a web server built on top of CTranslate2 that exposes an OpenAI-compatible REST API, making it easy to integrate CTranslate2 models into applications that already support the OpenAI API.

## Benchmarks

We translate the En->De test set *newstest2014* with multiple models:

* [OpenNMT-tf WMT14](https://opennmt.net/Models-tf/#translation): a base Transformer trained with OpenNMT-tf on the WMT14 dataset (4.5M lines)
* [OpenNMT-py WMT14](https://opennmt.net/Models-py/#translation): a base Transformer trained with OpenNMT-py on the WMT14 dataset (4.5M lines)
* [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/en-de#opus-2020-02-26zip): a base Transformer trained with Marian on all OPUS data available on 2020-02-26 (81.9M lines)

The benchmark reports the number of target tokens generated per second (higher is better). The results are aggregated over multiple runs. See the [benchmark scripts](tools/benchmark) for more details and reproduce these numbers.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

#### CPU

| | Tokens per second | Max. memory | BLEU |
| --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 209.2 | 2653MB | 26.93 |
| **OpenNMT-py WMT14 model** | | | |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1) | 275.8 | 2012MB | 26.77 |
| - int8 | 323.3 | 1359MB | 26.72 |
| CTranslate2 3.6.0 | 658.8 | 849MB | 26.77 |
| - int16 | 733.0 | 672MB | 26.82 |
| - int8 | 860.2 | 529MB | 26.78 |
| - int8 + vmap | 1126.2 | 598MB | 26.64 |
| **OPUS-MT model** | | | |
| Transformers 4.26.1 (with PyTorch 1.13.1) | 147.3 | 2332MB | 27.90 |
| Marian 1.11.0 | 344.5 | 7605MB | 27.93 |
| - int16 | 330.2 | 5901MB | 27.65 |
| - int8 | 355.8 | 4763MB | 27.27 |
| CTranslate2 3.6.0 | 525.0 | 721MB | 27.92 |
| - int16 | 596.1 | 660MB | 27.53 |
| - int8 | 696.1 | 516MB | 27.65 |

Executed with 4 threads on a [*c5.2xlarge*](https://aws.amazon.com/ec2/instance-types/c5/) Amazon EC2 instance equipped with an Intel(R) Xeon(R) Platinum 8275CL CPU.

#### GPU

| | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU |
| --- | --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | | |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 1483.5 | 3031MB | 3122MB | 26.94 |
| **OpenNMT-py WMT14 model** | | | | |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1) | 1795.2 | 2973MB | 3099MB | 26.77 |
| FasterTransformer 5.3 | 6979.0 | 2402MB | 1131MB | 26.77 |
| - float16 | 8592.5 | 1360MB | 1135MB | 26.80 |
| CTranslate2 3.6.0 | 6634.7 | 1261MB | 953MB | 26.77 |
| - int8 | 8567.2 | 1005MB | 807MB | 26.85 |
| - float16 | 10990.7 | 941MB | 807MB | 26.77 |
| - int8 + float16 | 8725.4 | 813MB | 800MB | 26.83 |
| **OPUS-MT model** | | | | |
| Transformers 4.26.1 (with PyTorch 1.13.1) | 1022.9 | 4097MB | 2109MB | 27.90 |
| Marian 1.11.0 | 3241.0 | 3381MB | 2156MB | 27.92 |
| - float16 | 3962.4 | 3239MB | 1976MB | 27.94 |
| CTranslate2 3.6.0 | 5876.4 | 1197MB | 754MB | 27.92 |
| - int8 | 7521.9 | 1005MB | 792MB | 27.79 |
| - float16 | 9296.7 | 909MB | 814MB | 27.90 |
| - int8 + float16 | 8362.7 | 813MB | 766MB | 27.90 |

Executed with CUDA 11 on a [*g5.xlarge*](https://aws.amazon.com/ec2/instance-types/g5/) Amazon EC2 instance equipped with a NVIDIA A10G GPU (driver version: 510.47.03).

## Contributing

CTranslate2 is a community-driven project. We welcome contributions of all kinds:
* **New Model Support:** Help us implement more Transformer architectures.
* **Performance:** Propose optimizations for CPU or GPU kernels.
* **Bug Reports:** Open an issue if you find something not working as expected.
* **Documentation:** Improve our guides or add new examples.

Check out our [Contributing Guide](CONTRIBUTING.md) to learn how to set up your development environment.

## Additional resources

* [Documentation](https://opennmt.net/CTranslate2)
* [Forum](https://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/CTranslate2)
