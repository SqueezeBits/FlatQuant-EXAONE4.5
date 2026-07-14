#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/gemm/device/gemm.h>
#include <w4a4_gemm.h>

namespace {

__global__ void quantize_pack_kernel(
    const __nv_bfloat16* x, uint8_t* packed, float* scales,
    int64_t rows, int64_t cols, const half* clip) {
  int row = blockIdx.x;
  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_max = fmaxf(local_max, fabsf(__bfloat162float(x[row * cols + col])));
  __shared__ float reduction[256];
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  // Match PyTorch's elementwise CUDA division (multiply by the rounded
  // reciprocal) so half-way rounding decisions agree with the reference.
  float scale = fmaxf(reduction[0] * __half2float(*clip), 1.0e-8f) * (1.0f / 7.0f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  for (int col = threadIdx.x * 2; col < cols; col += blockDim.x * 2) {
    int lo = max(-8, min(7, __float2int_rn(__bfloat162float(x[row * cols + col]) / scale)));
    int hi = max(-8, min(7, __float2int_rn(__bfloat162float(x[row * cols + col + 1]) / scale)));
    packed[row * (cols / 2) + col / 2] = (lo & 15) | ((hi & 15) << 4);
  }
}

template <typename Output>
__global__ void scale_output_kernel(
    const int32_t* accum, const float* x_scale, const half* w_scale,
    Output* output, int64_t rows, int64_t cols) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  int64_t row = index / cols;
  int64_t col = index % cols;
  float value = static_cast<float>(accum[index]);
  value *= x_scale[row] * __half2float(w_scale[col]);
  output[index] = static_cast<Output>(value);
}

void check_sm80() {
  cudaDeviceProp prop;
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(prop.major == 8 && prop.minor == 0,
              "packed W4A4 currently supports NVIDIA SM80 only");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_pack_i4_cuda(
    const torch::Tensor& x, const torch::Tensor& clip) {
  check_sm80();
  TORCH_CHECK(x.is_cuda() && clip.is_cuda(), "x and clip must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "W4A4 activation must be BF16");
  TORCH_CHECK(clip.scalar_type() == torch::kFloat16 && clip.numel() == 1,
              "clip must be a scalar FP16 tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 64 == 0, "W4A4 requires 2D x and K % 64 == 0");
  TORCH_CHECK(x.is_contiguous() && clip.is_contiguous(), "inputs must be contiguous");
  auto packed = torch::empty({x.size(0), x.size(1) / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({x.size(0), 1}, x.options().dtype(torch::kFloat32));
  quantize_pack_kernel<<<x.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), packed.data_ptr<uint8_t>(),
      scales.data_ptr<float>(), x.size(0), x.size(1),
      reinterpret_cast<const half*>(clip.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

torch::Tensor w4a4_linear_cuda(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype) {
  check_sm80();
  TORCH_CHECK(packed_x.scalar_type() == torch::kUInt8 && packed_w.scalar_type() == torch::kUInt8,
              "packed tensors must be uint8");
  TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32 && w_scale.scalar_type() == torch::kFloat16,
              "x_scale must be FP32 and w_scale must be FP16");
  TORCH_CHECK(packed_x.dim() == 2 && packed_w.dim() == 2 && packed_x.size(1) == packed_w.size(1),
              "packed tensors must be 2D with matching K");
  TORCH_CHECK(packed_x.size(1) % 32 == 0 && packed_w.size(0) % 8 == 0,
              "W4A4 requires K % 64 == 0 and N % 8 == 0");
  TORCH_CHECK(x_scale.sizes() == torch::IntArrayRef({packed_x.size(0), 1}), "x_scale shape mismatch");
  TORCH_CHECK(w_scale.sizes() == torch::IntArrayRef({packed_w.size(0), 1}), "w_scale shape mismatch");
  TORCH_CHECK(output_dtype == torch::kBFloat16 || output_dtype == torch::kFloat16,
              "output_dtype must be BF16 or FP16");
  c10::cuda::CUDAGuard guard(packed_x.device());
  int64_t m = packed_x.size(0), n = packed_w.size(0), k = packed_x.size(1) * 2;
  auto accum = torch::empty({m, n}, packed_x.options().dtype(torch::kInt32));
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t, cutlass::layout::RowMajor,
      cutlass::int4b_t, cutlass::layout::ColumnMajor,
      int32_t, cutlass::layout::RowMajor, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {int(m), int(n), int(k)},
      {reinterpret_cast<cutlass::int4b_t*>(packed_x.data_ptr()), int(k)},
      {reinterpret_cast<cutlass::int4b_t*>(packed_w.data_ptr()), int(k)},
      {accum.data_ptr<int32_t>(), int(n)}, {accum.data_ptr<int32_t>(), int(n)}, {1, 0});
  TORCH_CHECK(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess,
              "CUTLASS W4A4 GEMM failed");
  auto output = torch::empty({m, n}, packed_x.options().dtype(output_dtype));
  int blocks = (m * n + 255) / 256;
  if (output_dtype == torch::kBFloat16)
    scale_output_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        accum.data_ptr<int32_t>(), x_scale.data_ptr<float>(),
        reinterpret_cast<const half*>(w_scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), m, n);
  else
    scale_output_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        accum.data_ptr<int32_t>(), x_scale.data_ptr<float>(),
        reinterpret_cast<const half*>(w_scale.data_ptr()), reinterpret_cast<half*>(output.data_ptr()), m, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
