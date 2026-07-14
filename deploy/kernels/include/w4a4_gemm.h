#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> quantize_pack_i4_cuda(
    const torch::Tensor& x, const torch::Tensor& clip);

torch::Tensor w4a4_linear_cuda(
    const torch::Tensor& packed_x, const torch::Tensor& packed_w,
    const torch::Tensor& x_scale, const torch::Tensor& w_scale,
    c10::ScalarType output_dtype);
