#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tanh_gated_residual_add_backward_tiling.h"

extern "C" void tanh_gated_residual_add_backward_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *gradOutput,
    uint8_t *gate,
    uint8_t *hiddenStates,
    uint8_t *mask,
    uint8_t *gradResidual,
    uint8_t *gradHiddenStates,
    uint8_t *gradGate,
    uint8_t *workspace,
    uint8_t *tiling);

namespace tanh_gated_residual_add_backward_ext {

pybind11::tuple run_tanh_gated_residual_add_backward(
    const at::Tensor &grad_output,
    const at::Tensor &gate,
    const at::Tensor &hidden_states,
    const at::Tensor &mask)
{
    TORCH_CHECK(grad_output.dim() >= 1, "grad_output must be at least 1D");
    TORCH_CHECK(gate.sizes() == grad_output.sizes(), "gate shape mismatch");
    TORCH_CHECK(hidden_states.sizes() == grad_output.sizes(), "hidden_states shape mismatch");
    TORCH_CHECK(mask.sizes() == grad_output.sizes(), "mask shape mismatch");
    TORCH_CHECK(grad_output.scalar_type() == at::kBFloat16, "grad_output must be bfloat16");
    TORCH_CHECK(gate.scalar_type() == at::kBFloat16, "gate must be bfloat16");
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(mask.scalar_type() == at::kBFloat16, "mask must be bfloat16");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");

    const int32_t numElements = static_cast<int32_t>(grad_output.numel());
    const int32_t numBlocks = (numElements + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, numBlocks);
    const int32_t tasksPerCore = (numBlocks + usedCoreNum - 1) / usedCoreNum;

    at::Tensor grad_residual = at::empty_like(grad_output);
    at::Tensor grad_hidden_states = at::empty_like(grad_output);
    at::Tensor grad_gate = at::empty_like(grad_output);
    at::Tensor workspace = at::empty({DEFAULT_NUM_PHYSICAL_CORES}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor tiling_cpu = at::empty(
        {static_cast<long>(sizeof(TanhGatedResidualAddBackwardTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<TanhGatedResidualAddBackwardTiling *>(tiling_cpu.data_ptr());
    tiling->numElements = numElements;
    tiling->blockSize = DEFAULT_BLOCK_SIZE;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->numBlocks = numBlocks;
    auto tiling_npu = tiling_cpu.to(at::kPrivateUse1);

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    tanh_gated_residual_add_backward_do_bf16(
        static_cast<uint32_t>(usedCoreNum),
        acl_stream,
        static_cast<uint8_t *>(const_cast<void *>(grad_output.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gate.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(hidden_states.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(mask.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(grad_residual.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(grad_hidden_states.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(grad_gate.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(workspace.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tiling_npu.storage().data())));

    return pybind11::make_tuple(grad_residual, grad_hidden_states, grad_gate);
}

}  // namespace tanh_gated_residual_add_backward_ext

PYBIND11_MODULE(_tanh_gated_residual_add_backward_ext, m)
{
    m.doc() = "tanh_gated_residual_add_backward extension";
    m.def("run_tanh_gated_residual_add_backward",
          &tanh_gated_residual_add_backward_ext::run_tanh_gated_residual_add_backward,
          "");
}
