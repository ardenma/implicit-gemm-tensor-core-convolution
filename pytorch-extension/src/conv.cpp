#include <torch/extension.h>
#include <vector>
#include <cuda_fp16.h>
#include <chrono>

// CUDA forward declarations
std::vector<torch::Tensor> conv_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> conv_cuda_tc_forward_v1(
    torch::Tensor input,
    torch::Tensor weights);

// Added pragma unrolls
std::vector<torch::Tensor> conv_cuda_tc_forward_v2(
    torch::Tensor input,
    torch::Tensor weights);

std::vector<torch::Tensor> conv_cuda_tc_forward_v3(
    torch::Tensor input,
    torch::Tensor weights);


std::vector<torch::Tensor> conv_cuda_tc_forward_upper_bound(
    torch::Tensor input,
    torch::Tensor weights);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.options().device().is_cuda() , #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP16(x) AT_ASSERTM(x.options().dtype() == torch::kFloat16, #x " must be FP32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FP16(x) 


std::vector<torch::Tensor> conv_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  
  return conv_cuda_forward(input, weights, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_forward", &conv_cuda_forward, "Test Conv forward (CUDA)");
  m.def("conv_tc_forward_v1", &conv_cuda_tc_forward_v1, "Test Conv forward with tensor cores (CUDA)");
  m.def("conv_tc_forward_v2", &conv_cuda_tc_forward_v2, "Test Conv forward with tensor cores (CUDA)");
  m.def("conv_tc_forward_v3", &conv_cuda_tc_forward_v3, "Test Conv forward with tensor cores (CUDA)");
  m.def("conv_tc_forward_upper_bound", &conv_cuda_tc_forward_upper_bound, "Test Conv forward with tensor cores (CUDA)");
}