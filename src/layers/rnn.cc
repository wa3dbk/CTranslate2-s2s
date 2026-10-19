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

    void GRUCell::operator()(const StorageView* /*input*/,
                             const StorageView& /*prev_state*/,
                             StorageView& /*new_state*/) const {
      throw std::runtime_error("GRUCell::operator() not implemented yet");
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
