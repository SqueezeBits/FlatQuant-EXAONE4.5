#include <torch/extension.h>
#include <w4a4_gemm.h>

namespace {

std::tuple<torch::Tensor, torch::Tensor> quantize_pack_i4_meta(
    const torch::Tensor& x, const torch::Tensor& clip) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "W4A4 activation must be BF16");
  TORCH_CHECK(clip.scalar_type() == torch::kFloat16 && clip.dim() == 1 && clip.size(0) == 1,
              "clip must have shape (1,) and dtype FP16");
  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(x.is_contiguous() && clip.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.size(1) % 64 == 0, "W4A4 requires K % 64 == 0");
  return {
      torch::empty({x.size(0), x.size(1) / 2}, x.options().dtype(torch::kUInt8)),
      torch::empty({x.size(0), 1}, x.options().dtype(torch::kFloat32))};
}

torch::Tensor w4a4_linear_meta(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype) {
  TORCH_CHECK(packed_x.dim() == 2 && packed_w.dim() == 2,
              "packed_x and packed_w must be 2D tensors");
  TORCH_CHECK(packed_x.is_contiguous() && packed_w.is_contiguous() &&
              x_scale.is_contiguous() && w_scale.is_contiguous(),
              "all W4A4 tensors must be contiguous");
  TORCH_CHECK(packed_x.scalar_type() == torch::kUInt8 && packed_w.scalar_type() == torch::kUInt8,
              "packed tensors must be uint8");
  TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32 && w_scale.scalar_type() == torch::kFloat16,
              "x_scale must be FP32 and w_scale must be FP16");
  TORCH_CHECK(packed_x.size(1) == packed_w.size(1), "packed K mismatch");
  TORCH_CHECK(packed_x.size(1) % 32 == 0, "W4A4 requires K % 64 == 0");
  TORCH_CHECK(packed_w.size(0) % 8 == 0, "W4A4 requires N % 8 == 0");
  TORCH_CHECK(x_scale.dim() == 2 && x_scale.size(0) == packed_x.size(0) && x_scale.size(1) == 1,
              "x_scale shape mismatch");
  TORCH_CHECK(w_scale.dim() == 2 && w_scale.size(0) == packed_w.size(0) && w_scale.size(1) == 1,
              "w_scale shape mismatch");
  TORCH_CHECK(output_dtype == torch::kBFloat16, "output_dtype must be BF16");
  return torch::empty(
      {packed_x.size(0), packed_w.size(0)}, packed_x.options().dtype(output_dtype));
}

}  // namespace

TORCH_LIBRARY(flatquant, m) {
  m.def("quantize_pack_i4(Tensor x, Tensor clip) -> (Tensor, Tensor)");
  m.def("w4a4_linear(Tensor packed_x, Tensor packed_w, Tensor x_scale, Tensor w_scale, ScalarType output_dtype) -> Tensor");
  m.def("w4a4_kernel_name(int M, int N, int K) -> str", &w4a4_kernel_name);
  m.def("_w4a4_linear_candidate(Tensor packed_x, Tensor packed_w, Tensor x_scale, Tensor w_scale, ScalarType output_dtype, int candidate) -> Tensor");
}

TORCH_LIBRARY_IMPL(flatquant, CUDA, m) {
  m.impl("quantize_pack_i4", &quantize_pack_i4_cuda);
  m.impl("w4a4_linear", &w4a4_linear_cuda);
  m.impl("_w4a4_linear_candidate", &w4a4_linear_cuda_candidate);
}

TORCH_LIBRARY_IMPL(flatquant, Meta, m) {
  m.impl("quantize_pack_i4", &quantize_pack_i4_meta);
  m.impl("w4a4_linear", &w4a4_linear_meta);
}
