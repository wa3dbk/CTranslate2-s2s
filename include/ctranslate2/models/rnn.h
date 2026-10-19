#pragma once

#include "ctranslate2/models/sequence_to_sequence.h"

namespace ctranslate2 {
  namespace models {

    // Marian s2s (RNN encoder-decoder) model with bi-directional GRU encoder,
    // conditional-GRU decoder (two GRU transitions per step with Bahdanau
    // attention between them), scale-only layer normalisation, shared
    // input/output embeddings.
    //
    // See `README_s2s.md` at the repo root for the end-to-end design.
    class RNNModel : public SequenceToSequenceModel {
    public:
      size_t current_spec_revision() const override;
      std::unique_ptr<SequenceToSequenceReplica> as_sequence_to_sequence() const override;

    protected:
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      void initialize(ModelReader& model_reader) override;
      std::unique_ptr<Model> clone() const override;
    };

  }
}
