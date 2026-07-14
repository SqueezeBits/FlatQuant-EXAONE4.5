#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/memory.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
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
  float scale = fmaxf(reduction[0] * __half2float(*clip), 1.0e-8f) * (1.0f / 7.0f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  for (int col = threadIdx.x * 2; col < cols; col += blockDim.x * 2) {
    int lo = max(-8, min(7, __float2int_rn(__bfloat162float(x[row * cols + col]) / scale)));
    int hi = max(-8, min(7, __float2int_rn(__bfloat162float(x[row * cols + col + 1]) / scale)));
    packed[row * (cols / 2) + col / 2] = (lo & 15) | ((hi & 15) << 4);
  }
}

void check_sm80() {
  cudaDeviceProp prop;
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(prop.major == 8 && prop.minor == 0,
              "packed W4A4 currently supports NVIDIA SM80 only");
}

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::int4b_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

template <class ThreadblockShape, class WarpShape, int Stages>
struct FusedGemm {
  static constexpr int kEpilogueStages = 1;
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementC, 8, kEpilogueStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      ThreadMap, float, cute::Stride<cute::_1, cute::_0, int64_t>, false>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      ThreadMap, cutlass::half_t, cute::Stride<cute::_0, cute::_1, int64_t>, false>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleRows = cutlass::epilogue::threadblock::Sm80EVT<Multiply, Accum, XScale>;
  using ScaleCols = cutlass::epilogue::threadblock::Sm80EVT<Multiply, ScaleRows, WScale>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      ThreadMap, ElementC, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using AddBias = cutlass::epilogue::threadblock::Sm80EVT<Add, ScaleCols, Bias>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      ThreadMap, ElementC, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t>>;
  using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Store, AddBias>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 32,
      ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 32,
      ElementC, cutlass::layout::RowMajor, 8, ElementAccumulator, ElementCompute,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
      cutlass::arch::OpMultiplyAddSaturate, kEpilogueStages>::GemmKernel;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using KernelSmall = FusedGemm<cutlass::gemm::GemmShape<64, 128, 128>,
                              cutlass::gemm::GemmShape<32, 64, 128>, 3>;
using KernelMedium = FusedGemm<cutlass::gemm::GemmShape<128, 128, 128>,
                               cutlass::gemm::GemmShape<64, 64, 128>, 3>;
using KernelLarge = FusedGemm<cutlass::gemm::GemmShape<128, 256, 128>,
                              cutlass::gemm::GemmShape<64, 64, 128>, 3>;

int row_bucket(int64_t m) {
  return m < 32 ? 0 : m < 128 ? 1 : m < 512 ? 2 : m < 2048 ? 3 : 4;
}

int candidate_for(int64_t m, int64_t n, int64_t k) {
  int b = row_bucket(m);
  if (n == 5120 && k == 5120) { static constexpr int t[] = {0, 0, 0, 0, 1}; return t[b]; }
  if (n == 1024 && k == 5120) { static constexpr int t[] = {0, 0, 0, 0, 0}; return t[b]; }
  if (n == 27392 && k == 5120) { static constexpr int t[] = {0, 0, 1, 1, 1}; return t[b]; }
  if (n == 5120 && k == 27392) { static constexpr int t[] = {0, 0, 0, 1, 1}; return t[b]; }
  return b < 2 ? 0 : b == 2 ? 1 : 2;
}

const char* kernel_name(int64_t m, int64_t n, int64_t k) {
  static constexpr const char* names[] = {
      "sm80_64x128x128_w32x64x128_s3",
      "sm80_128x128x128_w64x64x128_s3",
      "sm80_128x256x128_w64x64x128_s3"};
  return names[candidate_for(m, n, k)];
}

template <class Config>
cutlass::Status launch_fused(
    int m, int n, int k, const torch::Tensor& packed_x,
    const torch::Tensor& packed_w, const torch::Tensor& x_scale,
    const torch::Tensor& w_scale, const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output) {
  using EVT = typename Config::Epilogue;
  const ElementC* bias_ptr = bias ? reinterpret_cast<const ElementC*>(bias->data_ptr()) : nullptr;
  typename EVT::Arguments epilogue{{{{{},
      {x_scale.data_ptr<float>(), 0.0f, {cute::_1{}, cute::_0{}, int64_t(m)}}, {}},
      {reinterpret_cast<const cutlass::half_t*>(w_scale.data_ptr()), cutlass::half_t(0),
       {cute::_0{}, cute::_1{}, int64_t(n)}}, {}},
      {bias_ptr, ElementC(0), {cute::_0{}, cute::_1{}, int64_t(n)}}, {}},
      {reinterpret_cast<ElementC*>(output.data_ptr()), {int64_t(n), cute::_1{}, int64_t(m) * n}}};
  typename Config::Device::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, epilogue,
      reinterpret_cast<ElementA*>(packed_x.data_ptr()),
      reinterpret_cast<ElementB*>(packed_w.data_ptr()), nullptr, nullptr,
      int64_t(m) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
  typename Config::Device gemm;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return status;
  return gemm(args, nullptr, at::cuda::getCurrentCUDAStream());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_pack_i4_cuda(
    const torch::Tensor& x, const torch::Tensor& clip) {
  TORCH_CHECK(x.is_cuda() && clip.is_cuda(), "x and clip must be CUDA tensors");
  TORCH_CHECK(x.get_device() == clip.get_device(), "x and clip must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(x.device());
  check_sm80();
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "W4A4 activation must be BF16");
  TORCH_CHECK(clip.scalar_type() == torch::kFloat16 && clip.dim() == 1 && clip.size(0) == 1,
              "clip must have shape (1,) and dtype FP16");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 64 == 0, "W4A4 requires 2D x and K % 64 == 0");
  TORCH_CHECK(x.is_contiguous() && clip.is_contiguous(), "inputs must be contiguous");
  auto packed = torch::empty({x.size(0), x.size(1) / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({x.size(0), 1}, x.options().dtype(torch::kFloat32));
  if (x.size(0) == 0) return {packed, scales};
  quantize_pack_kernel<<<x.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), packed.data_ptr<uint8_t>(),
      scales.data_ptr<float>(), x.size(0), x.size(1),
      reinterpret_cast<const half*>(clip.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

std::string w4a4_kernel_name(int64_t m, int64_t n, int64_t k) {
  TORCH_CHECK(m >= 0 && n > 0 && k > 0 && n % 8 == 0 && k % 64 == 0,
              "W4A4 kernel selection requires M >= 0, N % 8 == 0, K % 64 == 0");
  return kernel_name(m, n, k);
}

std::string w4a4_candidate_name(int64_t candidate) {
  TORCH_CHECK(candidate >= 0 && candidate < 3, "candidate must be 0, 1, or 2");
  static constexpr const char* names[] = {"small", "medium", "large"};
  return names[candidate];
}

torch::Tensor w4a4_linear_cuda_candidate(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype, int64_t candidate,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(packed_x.is_cuda() && packed_w.is_cuda() && x_scale.is_cuda() && w_scale.is_cuda(),
              "all W4A4 tensors must be CUDA tensors");
  TORCH_CHECK(packed_x.get_device() == packed_w.get_device() && packed_x.get_device() == x_scale.get_device() &&
              packed_x.get_device() == w_scale.get_device(), "all W4A4 tensors must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(packed_x.device());
  check_sm80();
  TORCH_CHECK(packed_x.is_contiguous() && packed_w.is_contiguous() && x_scale.is_contiguous() && w_scale.is_contiguous(),
              "all W4A4 tensors must be contiguous");
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
  TORCH_CHECK(output_dtype == torch::kBFloat16, "output_dtype must be BF16");
  if (bias) {
    TORCH_CHECK(bias->is_cuda() && bias->get_device() == packed_x.get_device(), "bias must be on the same CUDA device");
    TORCH_CHECK(bias->scalar_type() == torch::kBFloat16 && bias->is_contiguous() &&
                bias->dim() == 1 && bias->size(0) == packed_w.size(0),
                "bias must have shape (N,), dtype BF16, and be contiguous");
  }
  TORCH_CHECK(candidate >= -1 && candidate < 3, "candidate must be -1, 0, 1, or 2");
  int64_t m = packed_x.size(0), n = packed_w.size(0), k = packed_x.size(1) * 2;
  auto output = torch::empty({m, n}, packed_x.options().dtype(torch::kBFloat16));
  if (m == 0) return output;
  cutlass::Status status;
  int selected = candidate < 0 ? candidate_for(m, n, k) : candidate;
  switch (selected) {
    case 0: status = launch_fused<KernelSmall>(m, n, k, packed_x, packed_w, x_scale, w_scale, bias, output); break;
    case 1: status = launch_fused<KernelMedium>(m, n, k, packed_x, packed_w, x_scale, w_scale, bias, output); break;
    case 2: status = launch_fused<KernelLarge>(m, n, k, packed_x, packed_w, x_scale, w_scale, bias, output); break;
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS W4A4 fused GEMM failed: ", cutlassGetStatusString(status));
  return output;
}

torch::Tensor w4a4_linear_cuda(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype, const c10::optional<torch::Tensor>& bias) {
  return w4a4_linear_cuda_candidate(
      packed_x, packed_w, x_scale, w_scale, output_dtype, -1, bias);
}
