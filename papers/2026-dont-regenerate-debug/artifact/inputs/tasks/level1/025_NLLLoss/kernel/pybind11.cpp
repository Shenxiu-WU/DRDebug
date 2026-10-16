#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "nllloss_tiling.h"

extern "C" void nllloss_none_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *target,
    uint8_t *weight,
    uint8_t *out,
    uint8_t *partialSum,
    uint8_t *partialCount,
    uint8_t *tiling);

extern "C" void nllloss_none_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *target,
    uint8_t *weight,
    uint8_t *out,
    uint8_t *partialSum,
    uint8_t *partialCount,
    uint8_t *tiling);

extern "C" void nllloss_reduce_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *target,
    uint8_t *weight,
    uint8_t *out,
    uint8_t *partialSum,
    uint8_t *partialCount,
    uint8_t *tiling);

extern "C" void nllloss_reduce_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *input,
    uint8_t *target,
    uint8_t *weight,
    uint8_t *out,
    uint8_t *partialSum,
    uint8_t *partialCount,
    uint8_t *tiling);

namespace nllloss_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline LaunchFn ResolveLaunchFn(int32_t reduction, at::ScalarType dtype)
{
    const bool isFp16 = (dtype == at::kHalf);
    const bool isFp32 = (dtype == at::kFloat);
    TORCH_CHECK(isFp16 || isFp32, "input must be float16 or float32");

    if (reduction == REDUCTION_NONE) {
        return isFp16 ? nllloss_none_fp16_do : nllloss_none_fp32_do;
    }
    return isFp16 ? nllloss_reduce_fp16_do : nllloss_reduce_fp32_do;
}

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

at::Tensor run_nllloss(
    const at::Tensor &input,
    const at::Tensor &target,
    const at::Tensor &weight,
    int64_t ignore_index,
    int64_t reduction)
{
    TORCH_CHECK(input.device().is_privateuseone(), "input must be on NPU");
    TORCH_CHECK(target.device().is_privateuseone(), "target must be on NPU");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(target.scalar_type() == at::kInt, "target must be int32");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat, "input must be float16 or float32");

    const int32_t numPos = static_cast<int32_t>(input.numel() / input.size(1));
    const int32_t C = static_cast<int32_t>(input.size(1));
    const int32_t reduction32 = static_cast<int32_t>(reduction);
    const int32_t hasWeight = (weight.defined() && weight.numel() > 0) ? 1 : 0;

    if (hasWeight != 0) {
        TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight.numel() == C, "weight size must match C");
        TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input");
    }

    const int32_t blockN = 64;
    const int32_t nNum = CeilDivI32(std::max(numPos, 1), blockN);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(nNum, 1));
    const int32_t tasksPerCore = CeilDivI32(nNum, usedCoreNum);
    const int32_t subBlockN = std::max(blockN / DEFAULT_VEC_NUM, 1);

    at::Tensor out;
    if (reduction32 == REDUCTION_NONE) {
        out = at::empty(target.sizes(), input.options());
    } else {
        out = at::empty({1}, input.options());
    }

    const int32_t totalSlots = usedCoreNum * DEFAULT_VEC_NUM;
    at::Tensor partialSum = at::empty({totalSlots}, input.options());
    at::Tensor partialCount = at::empty({totalSlots}, input.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(NLLLossTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<NLLLossTiling *>(tilingCpu.data_ptr());
    tiling->numPos = numPos;
    tiling->C = C;
    tiling->reduction = reduction32;
    tiling->hasWeight = hasWeight;
    tiling->ignoreIndex = static_cast<int32_t>(ignore_index);
    tiling->blockN = blockN;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockN = subBlockN;
    tiling->slotsPerCore = DEFAULT_VEC_NUM;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = ResolveLaunchFn(reduction32, input.scalar_type());
    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(input.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(target.storage().data())),
        hasWeight ? static_cast<uint8_t *>(const_cast<void *>(weight.storage().data())) : nullptr,
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(partialSum.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(partialCount.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    if (reduction32 == REDUCTION_NONE) {
        return out;
    } else {
        return out.reshape({});
    }
}

} // namespace nllloss_ext

PYBIND11_MODULE(_nllloss_ext, m)
{
    m.doc() = "nllloss AscendC extension";
    m.def("run_nllloss", &nllloss_ext::run_nllloss, "");
}
