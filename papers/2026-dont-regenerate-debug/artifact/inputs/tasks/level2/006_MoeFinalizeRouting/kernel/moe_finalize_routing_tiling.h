#ifndef MOE_FINALIZE_ROUTING_TILING_H
#define MOE_FINALIZE_ROUTING_TILING_H

struct MoeFinalizeRoutingTiling {
    int32_t NUM_ROWS;
    int32_t K;
    int32_t H;
    int32_t H_TILE;
    int32_t mode;
    int32_t has_bias;
    int32_t has_skip1;
    int32_t has_skip2;
    int32_t E;
    int32_t C;
    int32_t totalTasks;
    int32_t tasksPerCore;
};

#endif
