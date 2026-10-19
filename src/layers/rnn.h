// This header is private — kept under src/layers/ because it is only
// consumed by src/models/rnn.cc. The public contract is the model spec
// name "RNNModelSpec".
#pragma once

#include <memory>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"

namespace ctranslate2 {
  namespace layers {

    // A single Marian GRU cell with scale-only layer norm.
    //
    // Gate axis layout in all weight/bias tensors is [reset | update | candidate].
    // The input and recurrent sides each produce a [B, 3H] projection; the two
    // are summed slice-wise and squashed by (sigmoid, sigmoid, tanh) to form
    // (r, u, c_pre). The new state is `u * prev + (1 - u) * c_pre`.
    //
    // When `final=true` the Marian source still uses `tanh(c_x + r * c_s)` for
    // the candidate; the flag is retained for future divergence (e.g. an alt
    // candidate path on the decoder's second cell).
    class GRUCell : public Layer {
    public:
      GRUCell(const models::Model& model, const std::string& scope,
              bool has_input, bool final);

      DataType output_type() const override;
      dim_t output_size() const override;

      // input may be nullptr for a stateless cell (no input side).
      void operator()(const StorageView* input,
                      const StorageView& prev_state,
                      StorageView& new_state) const;

    private:
      std::unique_ptr<Dense> _input_dense;
      std::unique_ptr<LayerNorm> _ln_input;
      Dense _recurrent_dense;
      std::unique_ptr<LayerNorm> _ln_recurrent;
      [[maybe_unused]] bool _final;
      dim_t _dim_state;
    };


    // Bahdanau additive attention. Memory-side projection is precomputed once
    // per encode(); query-side runs per decode step.
    class BahdanauAttention : public Layer {
    public:
      BahdanauAttention(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      dim_t output_size() const override;

      void precompute(const StorageView& memory,
                      StorageView& projected_memory) const;

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
      const StorageView& _v;
    };


    // Conditional GRU decoder cell: gru1 → attention → gru2.
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

      BahdanauAttention& attention() { return _attn; }

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

      using Encoder::operator();

    protected:
      void operator()(const std::vector<StorageView>& ids,
                      const StorageView* lengths,
                      StorageView& output) override;

    private:
      Embeddings _embeddings;
      GRUCell _fwd;
      GRUCell _bwd;
      dim_t _dim_rnn;
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
      using Decoder::replicate_state;

    protected:
      Dense& output_layer() override { return _output; }

    private:
      void ensure_lazy_state(DecoderState& state, dim_t current_batch_size);

      Embeddings _embeddings;
      Dense _init_state_linear;
      LayerNorm _init_state_ln;
      CGRUDecoderCell _cell;
      Dense _hidden_linear;
      LayerNorm _hidden_ln;
      Dense _output;
      dim_t _dim_rnn;
    };

  }
}
