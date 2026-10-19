#include "ctranslate2/models/rnn.h"

#include "ctranslate2/models/model_factory.h"

#include "../layers/rnn.h"

namespace ctranslate2 {
  namespace models {

    size_t RNNModel::current_spec_revision() const {
      return 2;
    }

    void RNNModel::initialize(ModelReader& model_reader) {
      SequenceToSequenceModel::initialize(model_reader);
    }

    bool RNNModel::is_quantizable(const std::string& variable_name) const {
      // Base `is_quantizable` returns true for anything ending in "weight".
      // That already covers all GRU, attention query/memory, dense, and output
      // weights; the attention 'v' is 1-D and not under /weight so it stays
      // float. Embeddings are excluded via `is_linear_weight`.
      return SequenceToSequenceModel::is_quantizable(variable_name);
    }

    bool RNNModel::is_linear_weight(const std::string& variable_name) const {
      // Weight tensors that participate in a Gemm — i.e. everything that is
      // quantizable except the embedding table (which is reused by the output
      // projection via aliasing and must stay float).
      return (is_quantizable(variable_name)
              && variable_name.find("embeddings") == std::string::npos);
    }

    std::unique_ptr<Model> RNNModel::clone() const {
      return std::make_unique<RNNModel>(*this);
    }

    std::unique_ptr<SequenceToSequenceReplica> RNNModel::as_sequence_to_sequence() const {
      const auto scoped_device_setter = get_scoped_device_setter();

      auto encoder = std::make_unique<layers::BiGRUEncoder>(*this, "encoder");
      auto decoder = std::make_unique<layers::RNNDecoder>(*this, "decoder");

      const auto model = std::static_pointer_cast<const RNNModel>(shared_from_this());
      return std::make_unique<EncoderDecoderReplica>(model, std::move(encoder), std::move(decoder));
    }

  }
}
