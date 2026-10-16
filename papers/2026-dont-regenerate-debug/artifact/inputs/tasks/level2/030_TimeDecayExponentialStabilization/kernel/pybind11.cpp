#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "time_decay_exp_stabilization_tiling.h"

extern "C" void time_decay_exp_stabilization_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *timeDecay,
    uint8_t *key,
    uint8_t *timeFirst,
    uint8_t *value,
    uint8_t *maxState,
    uint8_t *numState,
    uint8_t *denState,
    uint8_t *output,
    uint8_t *maxStateOut,
    uint8_t *numStateOut,
    uint8_t *denStateOut,
    uint8_t *workspace,
    uint8_t *tiling);

namespace time_decay_exp_stabilization_ext {

pybind11::tuple run_time_decay_exp_stabilization(
    const at::Tensor &timeDecay,
    const at::Tensor &key,
    const at::Tensor &timeFirst,
    const at::Tensor &value,
    const at::Tensor &maxState,
    const at::Tensor &numState,
    const at::Tensor &denState)
{
    TORCH_CHECK(timeDecay.dim() == 1, "time_decay must be 1D");
    TORCH_CHECK(key.dim() == 3, "key must be 3D [batch, seq_len, hidden]");
    TORCH_CHECK(timeFirst.dim() == 1, "time_first must be 1D");
    TORCH_CHECK(value.dim() == 3, "value must be 3D [batch, seq_len, hidden]");
    TORCH_CHECK(maxState.dim() == 2, "max_state must be 2D [batch, hidden]");
    TORCH_CHECK(numState.dim() == 2, "num_state must be 2D [batch, hidden]");
    TORCH_CHECK(denState.dim() == 2, "den_state must be 2D [batch, hidden]");
    
    int32_t batch = static_cast<int32_t>(key.size(0));
    int32_t seqLen = static_cast<int32_t>(key.size(1));
    int32_t hiddenSize = static_cast<int32_t>(key.size(2));
    
    TORCH_CHECK(hiddenSize == HIDDEN_SIZE, "hidden_size must be 2048");
    TORCH_CHECK(timeDecay.size(0) == hiddenSize, "time_decay shape mismatch");
    TORCH_CHECK(timeFirst.size(0) == hiddenSize, "time_first shape mismatch");
    TORCH_CHECK(value.size(0) == batch && value.size(1) == seqLen && value.size(2) == hiddenSize,
                "value shape mismatch");
    TORCH_CHECK(maxState.size(0) == batch && maxState.size(1) == hiddenSize,
                "max_state shape mismatch");
    TORCH_CHECK(numState.size(0) == batch && numState.size(1) == hiddenSize,
                "num_state shape mismatch");
    TORCH_CHECK(denState.size(0) == batch && denState.size(1) == hiddenSize,
                "den_state shape mismatch");
    
    at::Tensor output = at::empty_like(key);
    at::Tensor maxStateOut = at::empty_like(maxState);
    at::Tensor numStateOut = at::empty_like(numState);
    at::Tensor denStateOut = at::empty_like(denState);
    at::Tensor workspace = at::empty({32}, at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(TimeDecayExpStabilizationTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<TimeDecayExpStabilizationTiling *>(tilingCpu.data_ptr());
    tiling->batch = batch;
    tiling->seqLen = seqLen;
    tiling->hiddenSize = hiddenSize;
    tiling->tileSize = TILE_SIZE;
    tiling->numTiles = NUM_TILES;
    tiling->blockNum = batch * NUM_TILES;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    
    time_decay_exp_stabilization_do(
        tiling->blockNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void*>(timeDecay.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(key.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(timeFirst.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(value.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(maxState.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(numState.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(denState.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(output.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(maxStateOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(numStateOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(denStateOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(workspace.storage().data())),
        static_cast<uint8_t *>(const_cast<void*>(tilingNpu.storage().data())));
    
    return pybind11::make_tuple(output, maxStateOut, numStateOut, denStateOut);
}

}  // namespace time_decay_exp_stabilization_ext

PYBIND11_MODULE(_time_decay_exp_stabilization_ext, m)
{
    m.doc() = "time_decay_exp_stabilization extension";
    m.def("run_time_decay_exp_stabilization", &time_decay_exp_stabilization_ext::run_time_decay_exp_stabilization, "");
}
