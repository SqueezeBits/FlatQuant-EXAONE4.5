#pragma once

#include <torch/extension.h>
#include <string>

std::tuple<torch::Tensor, torch::Tensor> quantize_pack_i4_cuda(
    const torch::Tensor& x, const torch::Tensor& clip);

torch::Tensor w4a4_linear_cuda(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype, const c10::optional<torch::Tensor>& bias);

std::string w4a4_kernel_name(int64_t m, int64_t n, int64_t k);
std::string w4a4_candidate_name(int64_t candidate);

torch::Tensor w4a4_linear_cuda_candidate(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype, int64_t candidate,
    const c10::optional<torch::Tensor>& bias);
