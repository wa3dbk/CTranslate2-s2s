#include "rnn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace layers {

    namespace {

      // Env-var-gated NaN/Inf detector. Enable with `CT2_S2S_DEBUG=1`.
      // Prints shape, finite min/max, NaN/Inf counts, and first-5 values.
      // CPU-only (moves to CPU if needed).
      static bool s2s_debug_enabled() {
        static const bool enabled = std::getenv("CT2_S2S_DEBUG") != nullptr;
        return enabled;
      }

      static void probe(const char* tag, const StorageView& t) {
        if (!s2s_debug_enabled()) return;
        if (t.empty()) {
          std::fprintf(stderr, "[s2s_probe] %s: <empty>\n", tag);
          return;
        }
        StorageView host = t.device() == Device::CPU ? t : t.to(Device::CPU);
        if (host.dtype() != DataType::FLOAT32) {
          std::fprintf(stderr, "[s2s_probe] %s: dtype=%d (skipped)\n",
                       tag, static_cast<int>(host.dtype()));
          return;
        }
        const float* d = host.data<float>();
        const dim_t n = host.size();
        dim_t nan_count = 0, inf_count = 0;
        float fmin = 0.0f, fmax = 0.0f;
        bool have_finite = false;
        for (dim_t i = 0; i < n; ++i) {
          const float v = d[i];
          if (std::isnan(v)) { ++nan_count; continue; }
          if (std::isinf(v)) { ++inf_count; continue; }
          if (!have_finite) { fmin = fmax = v; have_finite = true; }
          else { if (v < fmin) fmin = v; if (v > fmax) fmax = v; }
        }
        std::string shape_s;
        for (dim_t i = 0; i < host.rank(); ++i) {
          shape_s += (i ? "x" : "") + std::to_string(host.dim(i));
        }
        std::fprintf(stderr,
                     "[s2s_probe] %s shape=%s n=%lld nan=%lld inf=%lld "
                     "min=%g max=%g head=[%g %g %g %g %g]\n",
                     tag, shape_s.c_str(),
                     (long long)n, (long long)nan_count, (long long)inf_count,
                     have_finite ? fmin : 0.0f,
                     have_finite ? fmax : 0.0f,
                     n > 0 ? d[0] : 0.0f,
                     n > 1 ? d[1] : 0.0f,
                     n > 2 ? d[2] : 0.0f,
                     n > 3 ? d[3] : 0.0f,
                     n > 4 ? d[4] : 0.0f);
      }

      // One-shot dump of a 1-D [V] tensor (output bias) to a raw float32 file.
      // Gated by `CT2_S2S_DUMP_BIAS=<path>`; fires at most once per process.
      static void dump_bias_once(const StorageView* bias) {
        static const char* path = std::getenv("CT2_S2S_DUMP_BIAS");
        if (!path || !*path) return;
        static bool fired = false;
        if (fired) return;
        if (!bias || bias->empty()) {
          std::fprintf(stderr, "[s2s_dump] output bias is null/empty\n");
          fired = true;
          return;
        }
        StorageView host = bias->device() == Device::CPU ? *bias : bias->to(Device::CPU);
        if (host.dtype() != DataType::FLOAT32) {
          std::fprintf(stderr, "[s2s_dump] bias dtype=%d (skipped)\n",
                       static_cast<int>(host.dtype()));
          return;
        }
        std::ofstream out(path, std::ios::binary);
        if (!out) {
          std::fprintf(stderr, "[s2s_dump] failed to open %s\n", path);
          return;
        }
        out.write(reinterpret_cast<const char*>(host.data<float>()),
                  static_cast<std::streamsize>(host.size() * sizeof(float)));
        std::fprintf(stderr, "[s2s_dump] wrote %lld bias floats to %s "
                     "(bias ptr=%p data=%p)\n",
                     (long long)host.size(), path,
                     (const void*)bias,
                     (const void*)bias->data<float>());
        fired = true;
      }

      // One-shot dump of a 2-D [B, V] tensor's first row to a raw float32 file.
      // Gated by `CT2_S2S_DUMP_LOGITS=<path>`; fires at most once per process.
      static void dump_logits_once(const StorageView& t) {
        static const char* path = std::getenv("CT2_S2S_DUMP_LOGITS");
        if (!path || !*path) return;
        static bool fired = false;
        if (fired) return;
        if (t.rank() != 2) return;
        StorageView host = t.device() == Device::CPU ? t : t.to(Device::CPU);
        if (host.dtype() != DataType::FLOAT32) return;
        const dim_t V = host.dim(1);
        std::ofstream out(path, std::ios::binary);
        if (!out) {
          std::fprintf(stderr, "[s2s_dump] failed to open %s\n", path);
          return;
        }
        out.write(reinterpret_cast<const char*>(host.data<float>()),
                  static_cast<std::streamsize>(V * sizeof(float)));
        std::fprintf(stderr, "[s2s_dump] wrote %lld logits floats to %s "
                     "(logits ptr=%p data=%p)\n",
                     (long long)V, path,
                     (const void*)&t,
                     (const void*)t.data<float>());
        fired = true;
      }

      // Build a float mask `[B, 1]` where entry b is 1.0 if `t < lengths[b]`,
      // else 0.0. `lengths` is a 1-D int32 tensor on the same device as the
      // output. If `lengths` is null, returns a tensor filled with ones.
      StorageView make_timestep_mask(const StorageView* lengths,
                                     dim_t batch_size,
                                     dim_t t,
                                     DataType dtype,
                                     Device device) {
        StorageView mask(Shape{batch_size, 1}, dtype, device);
        if (!lengths) {
          mask.fill(1.0f);
          return mask;
        }
        const StorageView lengths_host = lengths->to(Device::CPU);
        std::vector<float> values(batch_size, 0.0f);
        for (dim_t b = 0; b < batch_size; ++b)
          values[b] = (t < lengths_host.at<int32_t>(b)) ? 1.0f : 0.0f;
        StorageView host_mask(Shape{batch_size, 1}, values);
        if (device == Device::CPU)
          mask = std::move(host_mask);
        else
          mask = host_mask.to(device);
        return mask;
      }

      // Expand a `[B, 1]` tensor to `[B, D]` by tiling along the last axis.
      void broadcast_to(const StorageView& mask_b1, dim_t D, StorageView& out) {
        ops::Tile(1, D)(mask_b1, out);
      }

      // Returns `mask * a + (1 - mask) * b`, where `mask` is `[B, 1]` and `a`,
      // `b` are `[B, D]`.
      void select_by_mask(const StorageView& mask_b1,
                          const StorageView& a,
                          const StorageView& b,
                          StorageView& out) {
        const DataType dtype = a.dtype();
        const Device device = a.device();
        const dim_t D = a.dim(-1);

        StorageView mask_bd(dtype, device);
        broadcast_to(mask_b1, D, mask_bd);

        // out = b + mask * (a - b)
        StorageView diff(dtype, device);
        ops::Sub()(a, b, diff);
        StorageView scaled(dtype, device);
        ops::Mul()(mask_bd, diff, scaled);
        ops::Add()(b, scaled, out);
      }

      // Slice `[B, S, D]` along axis 1 at index t -> `[B, D]` using
      // `Gather(axis=1, batch_dims=1)` with a per-batch constant index tensor.
      void slice_time(const StorageView& bsd, dim_t t, StorageView& bd) {
        const dim_t B = bsd.dim(0);
        std::vector<int32_t> ids(B, static_cast<int32_t>(t));
        StorageView index_host(Shape{B}, ids);
        StorageView index = bsd.device() == Device::CPU
                              ? std::move(index_host)
                              : index_host.to(bsd.device());
        StorageView tmp(bsd.dtype(), bsd.device());
        ops::Gather(1, 1)(bsd, index, tmp);  // [B, D]
        bd = std::move(tmp);
      }

    }


    // ---------------------------------------------------------------- GRUCell

    GRUCell::GRUCell(const models::Model& model, const std::string& scope,
                     bool has_input, bool final)
      : _recurrent_dense(model, scope + "/recurrent")
      , _final(final)
      , _dim_state(_recurrent_dense.output_size() / 3) {
      if (has_input) {
        _input_dense = std::make_unique<Dense>(model, scope + "/input");
        if (model.layer_exists(scope + "/ln_input"))
          _ln_input = std::make_unique<LayerNorm>(model, scope + "/ln_input");
      }
      if (model.layer_exists(scope + "/ln_recurrent"))
        _ln_recurrent = std::make_unique<LayerNorm>(model, scope + "/ln_recurrent");
    }

    DataType GRUCell::output_type() const {
      return _recurrent_dense.output_type();
    }

    dim_t GRUCell::output_size() const {
      return _dim_state;
    }

    void GRUCell::operator()(const StorageView* input,
                             const StorageView& prev_state,
                             StorageView& new_state) const {
      const Device device = prev_state.device();
      const DataType dtype = prev_state.dtype();
      const dim_t H = _dim_state;

      // Recurrent projection: [B, H] -> [B, 3H], then scale-only LN. Split
      // last axis into (rs, us, cs) by the Marian gate order [r | u | c].
      probe("gru:prev_state", prev_state);
      StorageView s_proj(dtype, device);
      _recurrent_dense(prev_state, s_proj);
      probe("gru:recurrent_proj", s_proj);
      if (_ln_recurrent) {
        (*_ln_recurrent)(s_proj, s_proj);
        probe("gru:recurrent_proj_ln", s_proj);
      }
      StorageView rs(dtype, device), us(dtype, device), cs(dtype, device);
      ops::Split(-1, std::vector<dim_t>{H, H, H})(s_proj, rs, us, cs);

      // Input projection (if any): [B, Dx] -> [B, 3H], then scale-only LN.
      StorageView rx(dtype, device), ux(dtype, device), cx(dtype, device);
      const bool has_input = static_cast<bool>(_input_dense);
      if (has_input) {
        if (!input)
          throw std::invalid_argument(
              "GRUCell with an input side requires a non-null input tensor");
        probe("gru:input", *input);
        StorageView x_proj(dtype, device);
        (*_input_dense)(*input, x_proj);
        probe("gru:input_proj", x_proj);
        if (_ln_input) {
          (*_ln_input)(x_proj, x_proj);
          probe("gru:input_proj_ln", x_proj);
        }
        ops::Split(-1, std::vector<dim_t>{H, H, H})(x_proj, rx, ux, cx);
      }

      // Pre-activations: pre_r = rx + rs; pre_u = ux + us. When has_input is
      // false the pre-activations are just the recurrent slices.
      StorageView pre_r(dtype, device), pre_u(dtype, device);
      if (has_input) {
        ops::Add()(rx, rs, pre_r);
        ops::Add()(ux, us, pre_u);
      } else {
        pre_r = std::move(rs);
        pre_u = std::move(us);
      }
      StorageView r(dtype, device), u(dtype, device);
      ops::Sigmoid()(pre_r, r);
      ops::Sigmoid()(pre_u, u);

      // Candidate: c = tanh(cx + r * cs). Without an input side the candidate
      // reduces to tanh(r * cs).
      StorageView r_cs(dtype, device);
      ops::Mul()(r, cs, r_cs);
      StorageView pre_c(dtype, device);
      if (has_input)
        ops::Add()(cx, r_cs, pre_c);
      else
        pre_c = std::move(r_cs);
      StorageView c(dtype, device);
      ops::Tanh()(pre_c, c);

      // new_state = u * prev + (1 - u) * c, computed as c + u * (prev - c)
      // because Sub's scalar-on-left overload is not available for tensors.
      StorageView diff(dtype, device);
      ops::Sub()(prev_state, c, diff);
      StorageView weighted(dtype, device);
      ops::Mul()(u, diff, weighted);
      ops::Add()(c, weighted, new_state);
      probe("gru:new_state", new_state);
    }


    // -------------------------------------------------------- BahdanauAttention

    BahdanauAttention::BahdanauAttention(const models::Model& model,
                                         const std::string& scope)
      : _query_dense(model, scope + "/query")
      , _memory_dense(model, scope + "/memory")
      , _ln_context(model, scope + "/ln_context")
      , _ln_state(model, scope + "/ln_state")
      , _v(model.get_variable(scope + "/v")) {
    }

    DataType BahdanauAttention::output_type() const {
      return _memory_dense.output_type();
    }

    dim_t BahdanauAttention::output_size() const {
      return _memory_dense.output_size();
    }

    void BahdanauAttention::precompute(const StorageView& memory,
                                       StorageView& projected_memory) const {
      // Marian: `layerNorm(dot(memory, Wc_att), gammaContext_, b_att)`.
      // Dense projects memory from `[B, S, 2H]` to `[B, S, A]`; LN applies
      // along the last axis with `b_att` as the shift (beta).
      const Device device = memory.device();
      const DataType dtype = _memory_dense.output_type();
      probe("attn:memory_in", memory);
      StorageView proj(dtype, device);
      _memory_dense(memory, proj);
      probe("attn:memory_proj", proj);
      _ln_context(proj, projected_memory);
      probe("attn:memory_proj_ln", projected_memory);
    }

    void BahdanauAttention::compute(const StorageView& query,
                                    const StorageView& projected_memory,
                                    const StorageView& memory,
                                    const StorageView& memory_lengths,
                                    StorageView& context,
                                    StorageView& alignment) const {
      const Device device = query.device();
      const DataType dtype = _query_dense.output_type();
      const dim_t Bq = query.dim(0);
      const dim_t B_mem = projected_memory.dim(0);
      const dim_t S = projected_memory.dim(1);
      const dim_t A = projected_memory.dim(2);
      const dim_t E_mem = memory.dim(2);

      // During beam search `query` is beam-expanded (B_batch * beam) while
      // `projected_memory`/`memory`/`memory_lengths` stay at B_batch (we keep
      // them off the auto-replication path in `replicate_state`). Tile the
      // memory-side tensors up to the query batch size for this call; the
      // decoder's state is left untouched.
      if (Bq % B_mem != 0)
        throw std::invalid_argument(
            "BahdanauAttention: query batch is not a multiple of memory batch");
      const dim_t beam = Bq / B_mem;

      // Fast path for beam == 1 (greedy): avoid an unnecessary deep copy by
      // forwarding the state tensors directly.
      StorageView pm_tiled(dtype, device);
      StorageView mem_tiled(memory.dtype(), device);
      StorageView lengths_tiled(memory_lengths.dtype(), device);
      const StorageView* pm_ptr      = &projected_memory;
      const StorageView* mem_ptr     = &memory;
      const StorageView* lengths_ptr = &memory_lengths;
      if (beam > 1) {
        pm_tiled      = projected_memory;  repeat_batch(pm_tiled,      beam);
        mem_tiled     = memory;            repeat_batch(mem_tiled,     beam);
        lengths_tiled = memory_lengths;    repeat_batch(lengths_tiled, beam);
        pm_ptr      = &pm_tiled;
        mem_ptr     = &mem_tiled;
        lengths_ptr = &lengths_tiled;
      }

      // Query side: q = ln_state(Wa * query) -> [Bq, A].
      StorageView q_proj(dtype, device);
      _query_dense(query, q_proj);
      StorageView q(dtype, device);
      _ln_state(q_proj, q);

      // Broadcast q over the time axis: [Bq, A] -> [Bq, 1, A] -> [Bq, S, A].
      q.expand_dims(1);
      StorageView q_bcast(dtype, device);
      ops::Tile(1, S)(q, q_bcast);

      // Pre-energy = tanh(q_bcast + projected_memory) -> [Bq, S, A].
      StorageView pre(dtype, device);
      ops::Add()(q_bcast, *pm_ptr, pre);
      StorageView act(dtype, device);
      ops::Tanh()(pre, act);

      // Energy = act · v -> [Bq, S]. Reshape to [Bq*S, A] × [A, 1].
      act.reshape(Shape{Bq * S, A});
      StorageView v_col(dtype, device);
      v_col.view(const_cast<float*>(_v.data<float>()), Shape{A, 1});
      StorageView energy_flat(dtype, device);
      ops::MatMul()(act, v_col, energy_flat);
      energy_flat.reshape(Shape{Bq, S});
      StorageView energy = std::move(energy_flat);

      probe("attn:query", query);
      probe("attn:q_proj_ln", q);
      probe("attn:pre_tanh", pre);
      probe("attn:act", act);
      probe("attn:energy", energy);
      // Masked softmax along time axis.
      ops::SoftMax()(energy, lengths_ptr, alignment);  // [Bq, S]
      probe("attn:alignment", alignment);

      // Context: alignment [Bq, S] * memory [Bq, S, E_mem] -> [Bq, E_mem].
      // Compute as [Bq, 1, S] × [Bq, S, E_mem] -> [Bq, 1, E_mem] -> squeeze.
      StorageView align_row;
      align_row.shallow_copy(alignment);
      align_row.reshape(Shape{Bq, 1, S});
      StorageView ctx3(dtype, device);
      ops::MatMul()(align_row, *mem_ptr, ctx3);  // [Bq, 1, E_mem]
      ctx3.reshape(Shape{Bq, E_mem});
      context = std::move(ctx3);
      probe("attn:context", context);
    }


    // --------------------------------------------------------- CGRUDecoderCell

    CGRUDecoderCell::CGRUDecoderCell(const models::Model& model,
                                     const std::string& scope)
      : _gru1(model, scope + "/gru1", /*has_input=*/true,  /*final=*/false)
      , _attn(model, scope + "/attention")
      , _gru2(model, scope + "/gru2", /*has_input=*/true,  /*final=*/true) {
    }

    DataType CGRUDecoderCell::output_type() const {
      return _gru2.output_type();
    }

    dim_t CGRUDecoderCell::output_size() const {
      return _gru2.output_size();
    }

    void CGRUDecoderCell::operator()(const StorageView& input_emb,
                                     const StorageView& prev_state,
                                     const StorageView& projected_memory,
                                     const StorageView& memory,
                                     const StorageView& memory_lengths,
                                     StorageView& new_state,
                                     StorageView& context,
                                     StorageView& alignment) const {
      const Device device = input_emb.device();
      const DataType dtype = input_emb.dtype();

      // s1 = gru1(input_emb, prev_state).
      StorageView s1(dtype, device);
      _gru1(&input_emb, prev_state, s1);

      // (context, alignment) = attention(query=s1, projected_memory, memory, lengths).
      _attn.compute(s1, projected_memory, memory, memory_lengths, context, alignment);

      // new_state = gru2(context, s1).
      _gru2(&context, s1, new_state);
    }


    // ---------------------------------------------------------- BiGRUEncoder

    BiGRUEncoder::BiGRUEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings")
      , _fwd(model, scope + "/forward_cell",  /*has_input=*/true, /*final=*/false)
      , _bwd(model, scope + "/backward_cell", /*has_input=*/true, /*final=*/false)
      , _dim_rnn(_fwd.output_size()) {
      if (s2s_debug_enabled()) {
        const StorageView& w = model.get_variable(scope + "/embeddings/weight");
        std::fprintf(stderr,
                     "[s2s_probe] model_var %s/embeddings/weight dtype=%d "
                     "shape0=%lld shape1=%lld\n",
                     scope.c_str(), static_cast<int>(w.dtype()),
                     (long long)(w.rank() > 0 ? w.dim(0) : 0),
                     (long long)(w.rank() > 1 ? w.dim(1) : 0));
        const StorageView* s = model.get_variable_if_exists(scope + "/embeddings/weight_scale");
        if (s) {
          std::fprintf(stderr,
                       "[s2s_probe] model_var %s/embeddings/weight_scale "
                       "dtype=%d size=%lld\n",
                       scope.c_str(), static_cast<int>(s->dtype()),
                       (long long)s->size());
          probe("enc:embedding_scale", *s);
        } else {
          std::fprintf(stderr,
                       "[s2s_probe] model_var %s/embeddings/weight_scale: "
                       "<not present>\n", scope.c_str());
        }
        probe("enc:embedding_table", w);
      }
    }

    DataType BiGRUEncoder::output_type() const {
      return _fwd.output_type();
    }

    dim_t BiGRUEncoder::output_size() const {
      return 2 * _dim_rnn;
    }

    void BiGRUEncoder::operator()(const std::vector<StorageView>& ids,
                                  const StorageView* lengths,
                                  StorageView& output) {
      if (ids.empty())
        throw std::invalid_argument("BiGRUEncoder requires one input id tensor");
      const StorageView& token_ids = ids.front();

      const Device device = token_ids.device();
      const DataType dtype = _fwd.output_type();
      const dim_t B = token_ids.dim(0);
      const dim_t S = token_ids.dim(1);
      const dim_t H = _dim_rnn;

      // Embed ids [B, S] -> [B, S, E].
      StorageView emb(dtype, device);
      _embeddings(token_ids, emb);
      probe("enc:src_emb", emb);

      // Forward pass: h_fwd[-1] = 0.
      StorageView h_fwd_prev(Shape{B, H}, dtype, device);
      h_fwd_prev.zero();
      std::vector<StorageView> h_fwd(S);
      for (dim_t t = 0; t < S; ++t) {
        StorageView x_t(dtype, device);
        slice_time(emb, t, x_t);
        StorageView h_t(dtype, device);
        _fwd(&x_t, h_fwd_prev, h_t);
        if (lengths) {
          // Keep h_fwd[t] = h_fwd[t-1] at pad positions so the state does not
          // get polluted by padding ids but we still have a well-defined
          // output to zero out at the end.
          StorageView mask = make_timestep_mask(lengths, B, t, dtype, device);
          StorageView gated(dtype, device);
          select_by_mask(mask, h_t, h_fwd_prev, gated);
          h_t = std::move(gated);
        }
        h_fwd_prev = h_t;  // copy
        h_fwd[t] = std::move(h_t);
      }

      // Backward pass: h_bwd[S] = 0; skip updates where t >= lengths[b].
      StorageView h_bwd_prev(Shape{B, H}, dtype, device);
      h_bwd_prev.zero();
      std::vector<StorageView> h_bwd(S);
      for (dim_t t = S - 1; t >= 0; --t) {
        StorageView x_t(dtype, device);
        slice_time(emb, t, x_t);
        StorageView h_t(dtype, device);
        _bwd(&x_t, h_bwd_prev, h_t);
        if (lengths) {
          StorageView mask = make_timestep_mask(lengths, B, t, dtype, device);
          StorageView gated(dtype, device);
          select_by_mask(mask, h_t, h_bwd_prev, gated);
          h_t = std::move(gated);
        }
        h_bwd_prev = h_t;
        h_bwd[t] = std::move(h_t);
      }

      // Concat forward/backward per timestep, then stack along time axis.
      std::vector<StorageView> stacked;
      stacked.reserve(S);
      for (dim_t t = 0; t < S; ++t) {
        StorageView concat_t(dtype, device);
        ops::Concat(-1)({&h_fwd[t], &h_bwd[t]}, concat_t);  // [B, 2H]
        concat_t.expand_dims(1);  // [B, 1, 2H]
        stacked.push_back(std::move(concat_t));
      }
      std::vector<const StorageView*> stacked_ptrs;
      stacked_ptrs.reserve(S);
      for (auto& s : stacked)
        stacked_ptrs.push_back(&s);
      ops::Concat(1)(stacked_ptrs, output);  // [B, S, 2H]
      probe("enc:output", output);

      // Zero out padded positions in the output (match TransformerEncoder's
      // post-padding semantics).
      if (lengths) {
        const StorageView lengths_host = lengths->to(Device::CPU);
        if (device == Device::CPU) {
          float* data = output.data<float>();
          const dim_t stride_s = output.stride(1);
          const dim_t stride_b = output.stride(0);
          for (dim_t b = 0; b < B; ++b) {
            const int32_t len = lengths_host.at<int32_t>(b);
            for (dim_t t = len; t < S; ++t) {
              float* row = data + b * stride_b + t * stride_s;
              std::fill(row, row + 2 * H, 0.0f);
            }
          }
        }
      }
    }


    // ------------------------------------------------------------- RNNDecoder

    RNNDecoder::RNNDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _embeddings(model, scope + "/embeddings")
      , _init_state_linear(model, scope + "/init_state/linear")
      , _init_state_ln(model, scope + "/init_state/layer_norm")
      , _cell(model, scope + "/cell")
      , _hidden_embed(model,   scope + "/hidden/proj_embed")
      , _hidden_state(model,   scope + "/hidden/proj_state")
      , _hidden_context(model, scope + "/hidden/proj_context")
      , _output(model, scope + "/output")
      , _dim_rnn(_cell.output_size()) {
    }

    DecoderState RNNDecoder::initial_state(bool /*iterative_decoding*/) const {
      // Insert placeholders for state entries we populate lazily on step 0:
      //   - "memory", "memory_lengths": set by the EncoderDecoderReplica
      //     after this call (batch-only; see replicate_state).
      //   - "projected_memory": the attention memory projection cache
      //     (batch-only).
      //   - "hidden": the running decoder hidden state (replicated per beam).
      // Empty StorageViews are skipped by Decoder::replicate_state so inserting
      // them here is safe.
      DecoderState state;
      const DataType dtype = output_type();
      state.emplace("projected_memory", StorageView(dtype, _device));
      state.emplace("hidden", StorageView(dtype, _device));
      return state;
    }

    // Mean-pools `[B, S, D]` over the valid time steps (per-batch length from
    // `lengths`) into `[B, D]`. Pure CPU implementation; memory and lengths
    // must live on the CPU device.
    static void mean_pool_memory(const StorageView& memory,
                                 const StorageView& lengths,
                                 StorageView& pooled) {
      const dim_t B = memory.dim(0);
      const dim_t S = memory.dim(1);
      const dim_t D = memory.dim(2);
      pooled.resize(Shape{B, D});
      const float* mem_data = memory.data<float>();
      float* out = pooled.data<float>();
      for (dim_t b = 0; b < B; ++b) {
        const int32_t len = lengths.at<int32_t>(b);
        if (len <= 0)
          throw std::invalid_argument("RNNDecoder: memory_lengths must be positive");
        for (dim_t d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (dim_t t = 0; t < len; ++t)
            sum += mem_data[(b * S + t) * D + d];
          out[b * D + d] = sum / static_cast<float>(len);
        }
      }
    }

    // Ensures the lazy state entries ("projected_memory", "hidden") are
    // populated. Must be called at the start of every step; it is a no-op after
    // the first step.
    void RNNDecoder::ensure_lazy_state(DecoderState& state, dim_t current_batch_size) {
      StorageView& memory = state.at("memory");
      const StorageView& memory_lengths = state.at("memory_lengths");

      StorageView& projected_memory = state.at("projected_memory");
      if (projected_memory.empty())
        _cell.attention().precompute(memory, projected_memory);

      StorageView& hidden = state.at("hidden");
      if (hidden.empty()) {
        // Mean-pool memory over valid timesteps and apply init-state MLP.
        // The pooling must run on CPU; move to CPU and back if needed.
        const Device device = memory.device();
        StorageView memory_cpu = device == Device::CPU ? memory : memory.to(Device::CPU);
        StorageView lengths_cpu = memory_lengths.device() == Device::CPU
                                    ? memory_lengths : memory_lengths.to(Device::CPU);
        StorageView pooled(DataType::FLOAT32, Device::CPU);
        mean_pool_memory(memory_cpu, lengths_cpu, pooled);
        if (device != Device::CPU)
          pooled = pooled.to(device);
        probe("init:pooled", pooled);

        // Marian's `ff_state` is `tanh(LN(dot(x, W), gamma, b))`. Without
        // the tanh, the initial hidden state is unbounded and the decoder's
        // GRU gates get driven to saturation on step 0.
        StorageView h0(output_type(), device);
        _init_state_linear(pooled, h0);
        probe("init:h0_linear", h0);
        StorageView h0_norm(output_type(), device);
        _init_state_ln(h0, h0_norm);
        probe("init:h0_ln", h0_norm);
        StorageView h0_act(output_type(), device);
        ops::Tanh()(h0_norm, h0_act);
        probe("init:h0_tanh", h0_act);
        hidden = std::move(h0_act);

        // Replicate to match the current (possibly beam-expanded) batch size.
        const dim_t source_batch = hidden.dim(0);
        if (current_batch_size > source_batch) {
          if (current_batch_size % source_batch != 0)
            throw std::invalid_argument(
                "RNNDecoder: current batch size is not a multiple of source batch");
          const dim_t repeats = current_batch_size / source_batch;
          repeat_batch(hidden, repeats);
        }
      }
    }

    void RNNDecoder::operator()(dim_t step,
                                const StorageView& ids,
                                DecoderState& state,
                                StorageView* logits,
                                StorageView* attention) {
      const Device device = ids.device();
      const DataType dtype = output_type();

      // Embed ids [B] -> [B, E]. The Embeddings op accepts [B] and returns [B, E].
      StorageView embed(dtype, device);
      _embeddings(ids, embed);
      if (s2s_debug_enabled()) {
        std::fprintf(stderr, "[s2s_probe] === step=%lld ===\n", (long long)step);
      }
      probe("dec:tgt_ids", ids);
      probe("dec:tgt_emb", embed);

      // On the first step, lazily build "projected_memory" and "hidden".
      const dim_t B = embed.dim(0);
      if (step == 0)
        ensure_lazy_state(state, B);
      else
        ensure_lazy_state(state, B);  // Idempotent after first call.

      StorageView& memory = state.at("memory");
      StorageView& memory_lengths = state.at("memory_lengths");
      StorageView& projected_memory = state.at("projected_memory");
      StorageView& prev_hidden = state.at("hidden");

      probe("dec:prev_hidden", prev_hidden);
      probe("dec:projected_memory", projected_memory);
      StorageView new_hidden(dtype, device);
      StorageView context(dtype, device);
      StorageView alignment(dtype, device);
      _cell(embed, prev_hidden, projected_memory, memory, memory_lengths,
            new_hidden, context, alignment);
      probe("dec:new_hidden", new_hidden);
      probe("dec:context", context);
      probe("dec:alignment", alignment);

      // Deep-output: Marian builds `tanh(sum_i LN(W_i @ x_i, gamma_i, b_i))`
      // where `i` iterates over (embeddings, decoder hidden, aligned context).
      // Each sub-affine has distinct weights/gamma/beta — do NOT fuse by
      // concatenating inputs through a single Dense+LN.
      StorageView y0(dtype, device);
      {
        StorageView proj(dtype, device);
        _hidden_embed.linear(embed, proj);
        probe("deepout:proj_embed_linear", proj);
        _hidden_embed.layer_norm(proj, y0);
        probe("deepout:proj_embed_ln", y0);
      }
      StorageView y1(dtype, device);
      {
        StorageView proj(dtype, device);
        _hidden_state.linear(new_hidden, proj);
        probe("deepout:proj_state_linear", proj);
        _hidden_state.layer_norm(proj, y1);
        probe("deepout:proj_state_ln", y1);
      }
      StorageView y2(dtype, device);
      {
        StorageView proj(dtype, device);
        _hidden_context.linear(context, proj);
        probe("deepout:proj_context_linear", proj);
        _hidden_context.layer_norm(proj, y2);
        probe("deepout:proj_context_ln", y2);
      }
      StorageView sum01(dtype, device);
      ops::Add()(y0, y1, sum01);
      StorageView sum012(dtype, device);
      ops::Add()(sum01, y2, sum012);
      probe("deepout:sum", sum012);
      StorageView h_tilde(dtype, device);
      ops::Tanh()(sum012, h_tilde);
      probe("deepout:h_tilde", h_tilde);

      if (logits) {
        // Dump bias BEFORE the matmul so we see the pristine _bias storage,
        // not something that may have been overwritten by a fused op or
        // aliased write into the output buffer.
        dump_bias_once(_output.bias());
        _output(h_tilde, *logits);
        probe("dec:logits", *logits);
        dump_logits_once(*logits);
      }

      if (attention) {
        // Single-step decoder contract: `[B, S]`. BeamSearch's greedy path
        // indexes `{i, 0}` on this tensor; a rank-3 `[B, 1, S]` would raise
        // "number of indexed dimensions (2) does not match the storage rank
        // (3)" at decoding.cc. The beam path later expand_dims it back up.
        *attention = std::move(alignment);
      }

      state["hidden"] = std::move(new_hidden);
    }

    void RNNDecoder::operator()(const StorageView& ids,
                                const StorageView& lengths,
                                DecoderState& state,
                                StorageView& logits,
                                StorageView* attention) {
      // Teacher-forced sequence pass. Loops over target steps carrying hidden
      // state and stacks logits along the time axis.
      const Device device = ids.device();
      const DataType dtype = output_type();
      const dim_t B = ids.dim(0);
      const dim_t T = ids.dim(1);
      (void)lengths;  // Length masking of the target is applied by the caller
                      // via the final log-prob reduction; this loop simply
                      // forwards each timestep.

      std::vector<StorageView> step_logits;
      step_logits.reserve(T);
      std::vector<StorageView> step_attn;
      if (attention)
        step_attn.reserve(T);

      for (dim_t t = 0; t < T; ++t) {
        // Build [B] slice from ids [B, T].
        StorageView ids_t(DataType::INT32, device);
        {
          std::vector<int32_t> idx(B, static_cast<int32_t>(t));
          StorageView index_host(Shape{B}, idx);
          StorageView index = device == Device::CPU
                                ? std::move(index_host)
                                : index_host.to(device);
          StorageView gathered(ids.dtype(), device);
          ops::Gather(1, 1)(ids, index, gathered);
          ids_t = std::move(gathered);
        }

        StorageView logits_t(dtype, device);
        StorageView attn_t(dtype, device);
        operator()(t, ids_t, state, &logits_t, attention ? &attn_t : nullptr);
        step_logits.push_back(std::move(logits_t));
        if (attention)
          step_attn.push_back(std::move(attn_t));
      }

      // Stack along axis 1: each entry is [B, V] -> [B, 1, V] -> concat.
      std::vector<StorageView> expanded_logits;
      expanded_logits.reserve(T);
      for (auto& lt : step_logits) {
        lt.expand_dims(1);
        expanded_logits.push_back(std::move(lt));
      }
      std::vector<const StorageView*> logit_ptrs;
      logit_ptrs.reserve(T);
      for (auto& e : expanded_logits)
        logit_ptrs.push_back(&e);
      ops::Concat(1)(logit_ptrs, logits);  // [B, T, V]

      if (attention) {
        std::vector<const StorageView*> attn_ptrs;
        attn_ptrs.reserve(T);
        for (auto& a : step_attn)
          attn_ptrs.push_back(&a);
        ops::Concat(1)(attn_ptrs, *attention);
      }
    }

    bool RNNDecoder::replicate_state(const std::string& name) const {
      // Batch-only state entries: not replicated across beams.
      if (name == "memory" || name == "memory_lengths" || name == "projected_memory")
        return false;
      return Decoder::replicate_state(name);
    }

  }
}
