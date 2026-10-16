#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "moe_finalize_routing_tiling.h"

extern "C" void moe_finalize_routing_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *expanded_permuted_rows, uint8_t *scales,
    uint8_t *expanded_src_to_dst_row, uint8_t *export_for_source_row,
    uint8_t *bias, uint8_t *skip1, uint8_t *skip2,
    uint8_t *output, uint8_t *tiling);

extern "C" void moe_finalize_routing_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *expanded_permuted_rows, uint8_t *scales,
    uint8_t *expanded_src_to_dst_row, uint8_t *export_for_source_row,
    uint8_t *bias, uint8_t *skip1, uint8_t *skip2,
    uint8_t *output, uint8_t *tiling);

namespace my_moe_finalize_routing {

using LaunchFn = void(*)(
    uint32_t, void*,
    uint8_t*, uint8_t*, uint8_t*, uint8_t*,
    uint8_t*, uint8_t*, uint8_t*,
    uint8_t*, uint8_t*);

at::Tensor run_moe_finalize_routing(
    const at::Tensor &expanded_permuted_rows,
    const at::Tensor &skip1,
    const at::Tensor &skip2,
    const at::Tensor &bias,
    const at::Tensor &scales,
    const at::Tensor &expanded_src_to_dst_row,
    const at::Tensor &export_for_source_row,
    int64_t drop_pad_mode)
{
    TORCH_CHECK(expanded_permuted_rows.dim() >= 2,
                "expanded_permuted_rows must be at least 2D");
    TORCH_CHECK(scales.dim() == 2, "scales must be 2D");
    TORCH_CHECK(expanded_src_to_dst_row.dtype() == at::kInt,
                "expanded_src_to_dst_row must be int32");
    TORCH_CHECK(export_for_source_row.dtype() == at::kInt,
                "export_for_source_row must be int32");

    int NUM_ROWS = static_cast<int>(scales.sizes()[0]);
    int K = static_cast<int>(scales.sizes()[1]);
    int H, E, C;
    int mode = static_cast<int>(drop_pad_mode);

    if (mode == 0 || mode == 2) {
        H = static_cast<int>(expanded_permuted_rows.sizes()[1]);
        E = bias.defined() && bias.numel() > 0
                ? static_cast<int>(bias.sizes()[0]) : 1;
        C = 0;
    } else {
        E = static_cast<int>(expanded_permuted_rows.sizes()[0]);
        C = static_cast<int>(expanded_permuted_rows.sizes()[1]);
        H = static_cast<int>(expanded_permuted_rows.sizes()[2]);
    }

    int H_TILE = 256;
    int h_num = (H + H_TILE - 1) / H_TILE;
    int total_tasks = NUM_ROWS * h_num;
    uint32_t usedCoreNum = 20;
    if (total_tasks < static_cast<int>(usedCoreNum)) {
        usedCoreNum = static_cast<uint32_t>(total_tasks);
    }

    at::Tensor output = at::empty({NUM_ROWS, H},
        at::device(at::kPrivateUse1).dtype(expanded_permuted_rows.dtype()));

    at::Tensor t = at::empty({sizeof(MoeFinalizeRoutingTiling)},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<MoeFinalizeRoutingTiling *>(t.data_ptr());
    tiling_ptr->NUM_ROWS = NUM_ROWS;
    tiling_ptr->K = K;
    tiling_ptr->H = H;
    tiling_ptr->H_TILE = H_TILE;
    tiling_ptr->mode = mode;
    tiling_ptr->has_bias = (bias.defined() && bias.numel() > 0) ? 1 : 0;
    tiling_ptr->has_skip1 = (skip1.defined() && skip1.numel() > 0) ? 1 : 0;
    tiling_ptr->has_skip2 = (skip2.defined() && skip2.numel() > 0) ? 1 : 0;
    tiling_ptr->E = E;
    tiling_ptr->C = C;
    tiling_ptr->totalTasks = total_tasks;
    tiling_ptr->tasksPerCore =
        static_cast<int32_t>((total_tasks + usedCoreNum - 1) / usedCoreNum);
    auto tiling_npu = t.to(at::kPrivateUse1);

    LaunchFn launchFn;
    if (expanded_permuted_rows.scalar_type() == at::kHalf) {
        launchFn = moe_finalize_routing_do_fp16;
    } else if (expanded_permuted_rows.scalar_type() == at::kFloat) {
        launchFn = moe_finalize_routing_do_fp32;
    } else {
        TORCH_CHECK(false, "Unsupported dtype for moe_finalize_routing");
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    launchFn(usedCoreNum, acl_stream,
             static_cast<uint8_t*>(const_cast<void*>(
                 expanded_permuted_rows.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(
                 scales.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(
                 expanded_src_to_dst_row.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(
                 export_for_source_row.storage().data())),
             (bias.defined() && bias.numel() > 0)
                 ? static_cast<uint8_t*>(const_cast<void*>(
                       bias.storage().data())) : nullptr,
             (skip1.defined() && skip1.numel() > 0)
                 ? static_cast<uint8_t*>(const_cast<void*>(
                       skip1.storage().data())) : nullptr,
             (skip2.defined() && skip2.numel() > 0)
                 ? static_cast<uint8_t*>(const_cast<void*>(
                       skip2.storage().data())) : nullptr,
             static_cast<uint8_t*>(const_cast<void*>(
                 output.storage().data())),
             static_cast<uint8_t*>(const_cast<void*>(
                 tiling_npu.storage().data())));

    return output;
}

} // namespace my_moe_finalize_routing

PYBIND11_MODULE(_moe_finalize_routing_ext, m)
{
    m.def("run_moe_finalize_routing",
          &my_moe_finalize_routing::run_moe_finalize_routing, "");
}
