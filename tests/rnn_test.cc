#include "test_utils.h"

#include "ctranslate2/models/model.h"

#include "layers/rnn.h"

using namespace ctranslate2;

// Mock model exposing register_variable so a GRUCell can be constructed in
// isolation, without going through the full load path.
class RNNMockModel : public models::Model {
public:
  using Model::register_variable;
  std::unique_ptr<models::Model> clone() const override { return nullptr; }
  void finalize() {
    set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, /*update_weight=*/false);
  }
};

// Forward pass of a GRUCell with no layer norm matches the Marian reference
// values computed by numpy: gate order [r | u | c] along axis 0 of the
// concatenated weight; new_state = u * prev + (1 - u) * tanh(cx + r * cs).
TEST(RNNTest, GRUCellForwardNoLayerNorm) {
  constexpr dim_t H = 2;
  constexpr dim_t Dx = 2;

  RNNMockModel model;

  // Weights come from python/numpy reference (see commit message for script).
  // Shape [3H, in] with trans_b=true in the Dense gemm.
  model.register_variable("cell/recurrent/weight", StorageView({3 * H, H}, std::vector<float>{
      0.10f,  0.20f,
      0.30f, -0.10f,
     -0.20f,  0.10f,
      0.10f,  0.30f,
      0.20f, -0.20f,
      0.10f,  0.40f,
  }));
  model.register_variable("cell/recurrent/bias", StorageView({3 * H}, std::vector<float>{
      0.10f, -0.10f, 0.20f, 0.00f, -0.20f, 0.30f,
  }));
  model.register_variable("cell/input/weight", StorageView({3 * H, Dx}, std::vector<float>{
      0.50f, -0.10f,
     -0.20f,  0.30f,
      0.10f,  0.40f,
      0.30f, -0.30f,
     -0.10f,  0.20f,
      0.40f, -0.40f,
  }));
  model.register_variable("cell/input/bias", StorageView({3 * H}, std::vector<float>(3 * H, 0.0f)));

  model.finalize();

  layers::GRUCell cell(model, "cell", /*has_input=*/true, /*final=*/false);
  ASSERT_EQ(cell.output_size(), H);

  const StorageView prev_state({1, H}, std::vector<float>{0.3f, -0.5f});
  const StorageView input({1, Dx}, std::vector<float>{1.0f, 2.0f});

  StorageView new_state(DataType::FLOAT32);
  cell(&input, prev_state, new_state);

  const StorageView expected({1, H}, std::vector<float>{0.2918389310f, -0.3855459437f});
  expect_storage_eq(new_state, expected, 1e-5);
}
