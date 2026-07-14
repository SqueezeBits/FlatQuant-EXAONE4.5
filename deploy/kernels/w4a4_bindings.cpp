#include <torch/extension.h>
#include <w4a4_gemm.h>

namespace {

std::tuple<torch::Tensor, torch::Tensor> quantize_pack_i4_meta(
    const torch::Tensor& x, const torch::Tensor&) {
  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(x.size(1) % 64 == 0, "W4A4 requires K % 64 == 0");
  return {
      torch::empty({x.size(0), x.size(1) / 2}, x.options().dtype(torch::kUInt8)),
      torch::empty({x.size(0), 1}, x.options().dtype(torch::kFloat32))};
}

torch::Tensor w4a4_linear_meta(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor&, const torch::Tensor&, c10::ScalarType output_dtype) {
  TORCH_CHECK(packed_x.dim() == 2 && packed_w.dim() == 2,
              "packed_x and packed_w must be 2D tensors");
  TORCH_CHECK(packed_x.size(1) == packed_w.size(1), "packed K mismatch");
  return torch::empty(
      {packed_x.size(0), packed_w.size(0)}, packed_x.options().dtype(output_dtype));
}

}  // namespace

TORCH_LIBRARY(flatquant, m) {
  m.def("quantize_pack_i4(Tensor x, Tensor clip) -> (Tensor, Tensor)");
  m.def("w4a4_linear(Tensor packed_x, Tensor packed_w, Tensor x_scale, Tensor w_scale, ScalarType output_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(flatquant, CUDA, m) {
  m.impl("quantize_pack_i4", &quantize_pack_i4_cuda);
  m.impl("w4a4_linear", &w4a4_linear_cuda);
}

TORCH_LIBRARY_IMPL(flatquant, Meta, m) {
  m.impl("quantize_pack_i4", &quantize_pack_i4_meta);
  m.impl("w4a4_linear", &w4a4_linear_meta);
}
