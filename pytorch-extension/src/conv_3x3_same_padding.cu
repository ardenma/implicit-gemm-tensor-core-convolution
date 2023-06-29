#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void conv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output) {

    const int n = blockIdx.z * blockDim.z + threadIdx.z;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;

    // ASSUMES NHWC, 3x3 kernel with 1 padding
    // Input: NHWC
    // Weights: KCRS
    // Output: NPQK

    // Weights: out_channels, in_channels, kernel_size, kernel_size

    // Iterate over output channels
    for (uint32_t k = 0; k < weights.size(0); k++) {
        // Iterate over filter
        for (uint32_t r = 0; r < weights.size(2); r++) {
            for (uint32_t s = 0; s < weights.size(3); s++) {
                for (uint32_t c = 0; c < weights.size(1); c++) {
                  int y = p + r - 1;
                  int x = q + s - 1;

                  // Same size zero padding
                  if (x >= 0 && x < input.size(2) && y >= 0 && y < input.size(1))
                      output[n][p][q][k] += input[n][y][x][c] * weights[k][c][r][s];
                }  // r
            }  // s
        }  // c
        // Add bias
        output[n][p][q][k] += bias[k];
    }  // k
}

std::vector<torch::Tensor> conv_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {

  const auto batch_size = input.size(0);

  const uint32_t num_threads_x = 32;
  const uint32_t num_threads_y = 32;

  // Make sure inputs are a multiple of num_threads
  assert(input.size(2) % num_threads_x == 0);
  assert(input.size(1) % num_threads_y == 0);
  const uint32_t num_blocks_x = input.size(2) / num_threads_x;
  const uint32_t num_blocks_y = input.size(1) / num_threads_y;

  assert(bias.size(0) == weights.size(0));

  const dim3 threads(num_threads_x, num_threads_y, 1);
  const dim3 blocks(num_blocks_x, num_blocks_y, batch_size);

  auto options =
  torch::TensorOptions()
    .dtype(input.scalar_type())
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  
  // Constructing intermediate tensor
  torch::Tensor output = torch::zeros({input.size(0), input.size(1), input.size(2), weights.size(0)}, options);
  
  // https://github.com/pytorch/extension-cpp/issues/49
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv_forward_cuda", ([&] {
    conv_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return {output};
}