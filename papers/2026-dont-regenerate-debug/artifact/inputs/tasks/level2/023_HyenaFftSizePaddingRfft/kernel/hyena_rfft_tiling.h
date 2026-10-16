#ifndef HYENA_RFFT_TILING_H
#define HYENA_RFFT_TILING_H

#include <cstdint>

struct HyenaRfftTiling {
    int32_t B;
    int32_t C;
    int32_t L;
    int32_t N;
    int32_t totalSeq;
};

#endif
