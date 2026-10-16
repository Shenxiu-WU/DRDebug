#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void conv3d_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *weight,
    uint8_t *bias,
    uint8_t *output,
    int N, int C, int D, int H, int W,
    int K, int D_out, int H_out, int W_out,
    int k_size, int stride, int padding, int dilation,
    int groups, int use_bias);

namespace conv3d_ext {

at::Tensor run_conv3d(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int N, int C, int D, int H, int W,
    int K, int D_out, int H_out, int W_out,
    int k_size, int stride, int padding, int dilation,
    int groups, int use_bias)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (use_bias) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    
    auto options = input.options();
    at::Tensor output = at::empty({N, K, D_out, H_out, W_out}, options);
    
    int total_spatial = N * D_out * H_out * W_out;
    int num_physical_cores = 20;
    int usedCoreNum = std::min(num_physical_cores, total_spatial);
    
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    conv3d_do(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t*>(const_cast<void*>(input.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(weight.storage().data())),
        use_bias ? static_cast<uint8_t*>(const_cast<void*>(bias.storage().data())) : nullptr,
        static_cast<uint8_t*>(const_cast<void*>(output.storage().data())),
        N, C, D, H, W,
        K, D_out, H_out, W_out,
        k_size, stride, padding, dilation,
        groups, use_bias);
    
    return output;
}

}  // namespace conv3d_ext

PYBIND11_MODULE(_conv3d_ext, m)
{
    m.doc() = "conv3d extension";
    m.def("run_conv3d", &conv3d_ext::run_conv3d, "");
}
