#ifndef WORKSPACE_QUEUE_H
#define WORKSPACE_QUEUE_H

#include "kernel_operator.h"

template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    __aicore__ inline WorkspaceQueue() {}

    __aicore__ inline void Init(GM_ADDR workspace, uint32_t slotSize,
                                uint16_t cubeNotifyVecId, uint16_t vecNotifyCubeId);
    __aicore__ inline void InitFreeSlots();

    __aicore__ inline AscendC::GlobalTensor<T> ProducerAcquire();
    __aicore__ inline void ProducerRelease();

    __aicore__ inline AscendC::GlobalTensor<T> ConsumerAcquire();
    __aicore__ inline void ConsumerRelease();

private:
    AscendC::GlobalTensor<T> workspace_;
    uint32_t slotSize_;
    uint32_t head_;
    uint32_t tail_;
    uint16_t cubeNotifyVecId_;
    uint16_t vecNotifyCubeId_;
};

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::Init(
    GM_ADDR workspace, uint32_t slotSize,
    uint16_t cubeNotifyVecId, uint16_t vecNotifyCubeId)
{
    slotSize_ = slotSize;
    head_ = 0;
    tail_ = 0;
    cubeNotifyVecId_ = cubeNotifyVecId;
    vecNotifyCubeId_ = vecNotifyCubeId;
    workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace), DEPTH * slotSize);
}

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::InitFreeSlots()
{
    for (uint32_t i = 0; i < DEPTH; ++i) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
    }
}

template <typename T, uint32_t DEPTH>
__aicore__ inline AscendC::GlobalTensor<T> WorkspaceQueue<T, DEPTH>::ProducerAcquire()
{
    AscendC::CrossCoreWaitFlag<0x2>(vecNotifyCubeId_);
    return workspace_[head_ % DEPTH * slotSize_];
}

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::ProducerRelease()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeNotifyVecId_);
    head_++;
}

template <typename T, uint32_t DEPTH>
__aicore__ inline AscendC::GlobalTensor<T> WorkspaceQueue<T, DEPTH>::ConsumerAcquire()
{
    AscendC::CrossCoreWaitFlag<0x2>(cubeNotifyVecId_);
    return workspace_[tail_ % DEPTH * slotSize_];
}

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::ConsumerRelease()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
    tail_++;
}

#endif // WORKSPACE_QUEUE_H
