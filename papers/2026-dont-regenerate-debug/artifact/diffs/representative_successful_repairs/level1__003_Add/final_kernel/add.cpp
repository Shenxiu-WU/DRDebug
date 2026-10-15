// Kernel entry points have been split into per-dtype files:
//   add_fp32.cpp  – add_alpha_fp32 / add_alpha_do_fp32
//   add_fp16.cpp  – add_alpha_fp16 / add_alpha_do_fp16
//   add_bf16.cpp  – add_alpha_bf16 / add_alpha_do_bf16
// This avoids multiple __global__ __aicore__ functions in a single
// translation unit, which can cause RegisterAscendBinary failure.
