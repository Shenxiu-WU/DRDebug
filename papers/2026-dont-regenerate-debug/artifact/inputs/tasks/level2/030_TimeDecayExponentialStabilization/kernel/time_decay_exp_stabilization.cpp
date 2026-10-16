#include "time_decay_exp_stabilization_kernel.h"

extern "C" __global__ __aicore__ void time_decay_exp_stabilization_custom(
    GM_ADDR timeDecay, GM_ADDR key, GM_ADDR timeFirst,
    GM_ADDR value, GM_ADDR maxState, GM_ADDR numState,
    GM_ADDR denState, GM_ADDR output,
    GM_ADDR maxStateOut, GM_ADDR numStateOut, GM_ADDR denStateOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TimeDecayExpStabilizationKernel kernel;
    kernel.Init(timeDecay, key, timeFirst, value, maxState, numState, denState,
                output, maxStateOut, numStateOut, denStateOut, workspace, tiling, &pipe);
    kernel.Process();
}

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
    uint8_t *tiling)
{
    time_decay_exp_stabilization_custom<<<blockDim, nullptr, stream>>>(
        timeDecay, key, timeFirst, value, maxState, numState, denState,
        output, maxStateOut, numStateOut, denStateOut, workspace, tiling);
}
