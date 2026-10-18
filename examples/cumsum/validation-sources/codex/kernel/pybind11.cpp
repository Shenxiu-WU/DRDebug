#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "cumsum_tiling.h"

extern "C" void cumsum_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *tiling);
extern "C" void cumsum_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *tiling);
extern "C" void cumsum_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *tiling);

namespace cumsum_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);

at::Tensor run_cumsum(const at::Tensor &x, int64_t dim)
{
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kHalf ||
        x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int64_t ndim = x.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    const int32_t scanSize = static_cast<int32_t>(x.sizes()[dim]);

    int64_t outerSize = 1;
    int64_t innerSize = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outerSize *= x.sizes()[d];
    }
    for (int64_t d = dim + 1; d < ndim; ++d) {
        innerSize *= x.sizes()[d];
    }
    int32_t scanMode = 0;
    if (x.scalar_type() != at::kBFloat16 && innerSize == 1) {
        scanMode = 1;
    } else if (outerSize > 1) {
        scanMode = 2;
    }

    // Compute P = product of all dimensions except `dim`
    int32_t P = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            P *= static_cast<int32_t>(x.sizes()[d]);
        }
    }

    // Transpose so scan dim is last, then reshape to [P, S].
    // This ensures each "column" (scan sequence) is contiguous in memory.
    // Layout: row p contains the p-th independent scan sequence.
    std::vector<int64_t> perm;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            perm.push_back(d);
        }
    }
    perm.push_back(dim);

    at::Tensor x_t;
    if (ndim == 1) {
        x_t = x.reshape({1, scanSize});
    } else {
        x_t = x.permute(perm).contiguous().reshape({P, scanSize});
    }

    const int32_t S = scanSize;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, P);
    const int32_t tasksPerCore = (P + usedCoreNum - 1) / usedCoreNum;

    at::Tensor y_t = at::empty_like(x_t);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(CumsumTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<CumsumTiling *>(tilingCpu.data_ptr());
    tp->S = S;
    tp->P = P;
    tp->usedCoreNum = usedCoreNum;
    tp->tasksPerCore = tasksPerCore;
    tp->scanMode = scanMode;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        launch = cumsum_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = cumsum_do_fp16;
    } else {
        launch = cumsum_do_bf16;
    }
    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x_t.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y_t.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    // Reshape back and inverse-permute to original layout
    auto original_sizes = x.sizes().vec();
    if (ndim == 1) {
        return y_t.reshape(original_sizes);
    }

    // y_t is [P, S], reshape to original shape with scan dim last
    std::vector<int64_t> shape_no_scan;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
            shape_no_scan.push_back(x.sizes()[d]);
        }
    }
    shape_no_scan.push_back(scanSize);
    y_t = y_t.reshape(shape_no_scan);

    // Inverse permute to restore original dimension order
    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        inv_perm[perm[i]] = i;
    }
    return y_t.permute(inv_perm).contiguous();
}

}  // namespace cumsum_ext

PYBIND11_MODULE(_cumsum_ext, m)
{
    m.doc() = "cumsum extension";
    m.def("run_cumsum", &cumsum_ext::run_cumsum, "");
}
