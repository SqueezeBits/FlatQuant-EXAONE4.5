#include <torch/extension.h>
#include <w4a4_gemm.h>


namespace {

}  // namespace

TORCH_LIBRARY(flatquant, m) {
  m.def("quantize_pack_i4(Tensor x, Tensor clip) -> (Tensor, Tensor)");
  m.def("w4a4_linear(Tensor packed_x, Tensor packed_w, Tensor x_scale, Tensor w_scale, ScalarType output_dtype, Tensor? bias=None) -> Tensor");
  m.def("w4a4_kernel_name(int M, int N, int K) -> str", &w4a4_kernel_name);
  m.def("_w4a4_candidate_name(int candidate) -> str", &w4a4_candidate_name);
  m.def("_w4a4_linear_candidate(Tensor packed_x, Tensor packed_w, Tensor x_scale, Tensor w_scale, ScalarType output_dtype, int candidate, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(flatquant, CUDA, m) {
  m.impl("quantize_pack_i4", &quantize_pack_i4_cuda);
  m.impl("w4a4_linear", &w4a4_linear_cuda);
  m.impl("_w4a4_linear_candidate", &w4a4_linear_cuda_candidate);
}
