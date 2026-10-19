#include "rnn.h"

#include <stdexcept>

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace layers {

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
      StorageView s_proj(dtype, device);
      _recurrent_dense(prev_state, s_proj);
      if (_ln_recurrent)
        (*_ln_recurrent)(s_proj, s_proj);
      StorageView rs(dtype, device), us(dtype, device), cs(dtype, device);
      ops::Split(-1, std::vector<dim_t>{H, H, H})(s_proj, rs, us, cs);

      // Input projection (if any): [B, Dx] -> [B, 3H], then scale-only LN.
      StorageView rx(dtype, device), ux(dtype, device), cx(dtype, device);
      const bool has_input = static_cast<bool>(_input_dense);
      if (has_input) {
        if (!input)
          throw std::invalid_argument(
              "GRUCell with an input side requires a non-null input tensor");
        StorageView x_proj(dtype, device);
        (*_input_dense)(*input, x_proj);
        if (_ln_input)
          (*_ln_input)(x_proj, x_proj);
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

    void BahdanauAttention::precompute(const StorageView& /*memory*/,
                                       StorageView& /*projected_memory*/) const {
      throw std::runtime_error("BahdanauAttention::precompute not implemented yet");
    }

    void BahdanauAttention::compute(const StorageView& /*query*/,
                                    const StorageView& /*projected_memory*/,
                                    const StorageView& /*memory*/,
                                    const StorageView& /*memory_lengths*/,
                                    StorageView& /*context*/,
                                    StorageView& /*alignment*/) const {
      throw std::runtime_error("BahdanauAttention::compute not implemented yet");
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

    void CGRUDecoderCell::operator()(const StorageView&, const StorageView&,
                                     const StorageView&, const StorageView&,
                                     const StorageView&,
                                     StorageView&, StorageView&, StorageView&) const {
      throw std::runtime_error("CGRUDecoderCell::operator() not implemented yet");
    }


    // ---------------------------------------------------------- BiGRUEncoder

    BiGRUEncoder::BiGRUEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings")
      , _fwd(model, scope + "/forward_cell",  /*has_input=*/true, /*final=*/false)
      , _bwd(model, scope + "/backward_cell", /*has_input=*/true, /*final=*/false)
      , _dim_rnn(_fwd.output_size()) {
    }

    DataType BiGRUEncoder::output_type() const {
      return _fwd.output_type();
    }

    dim_t BiGRUEncoder::output_size() const {
      return 2 * _dim_rnn;
    }

    void BiGRUEncoder::operator()(const std::vector<StorageView>& /*ids*/,
                                  const StorageView* /*lengths*/,
                                  StorageView& /*output*/) {
      throw std::runtime_error("BiGRUEncoder::operator() not implemented yet");
    }


    // ------------------------------------------------------------- RNNDecoder

    RNNDecoder::RNNDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _embeddings(model, scope + "/embeddings")
      , _init_state_linear(model, scope + "/init_state/linear")
      , _init_state_ln(model, scope + "/init_state/layer_norm")
      , _cell(model, scope + "/cell")
      , _hidden_linear(model, scope + "/hidden/linear")
      , _hidden_ln(model, scope + "/hidden/layer_norm")
      , _output(model, scope + "/output")
      , _dim_rnn(_cell.output_size()) {
    }

    DecoderState RNNDecoder::initial_state(bool /*iterative_decoding*/) const {
      throw std::runtime_error("RNNDecoder::initial_state not implemented yet");
    }

    void RNNDecoder::operator()(dim_t /*step*/,
                                const StorageView& /*ids*/,
                                DecoderState& /*state*/,
                                StorageView* /*logits*/,
                                StorageView* /*attention*/) {
      throw std::runtime_error("RNNDecoder::operator()(step) not implemented yet");
    }

    void RNNDecoder::operator()(const StorageView& /*ids*/,
                                const StorageView& /*lengths*/,
                                DecoderState& /*state*/,
                                StorageView& /*logits*/,
                                StorageView* /*attention*/) {
      throw std::runtime_error("RNNDecoder::operator()(seq) not implemented yet");
    }

    bool RNNDecoder::replicate_state(const std::string& name) const {
      // Batch-only state entries: not replicated across beams.
      if (name == "memory" || name == "memory_lengths" || name == "projected_memory")
        return false;
      return Decoder::replicate_state(name);
    }

  }
}
