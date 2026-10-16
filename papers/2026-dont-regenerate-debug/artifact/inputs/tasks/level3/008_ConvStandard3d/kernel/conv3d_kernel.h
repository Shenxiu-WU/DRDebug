#pragma once

#include "kernel_operator.h"

template <typename dataType>
class Conv3dKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output,
        int N, int C, int D, int H, int W,
        int K, int D_out, int H_out, int W_out,
        int k_size, int stride, int padding, int dilation,
        int groups, int use_bias)
    {
        N_ = N; C_ = C; D_ = D; H_ = H; W_ = W;
        K_ = K; D_out_ = D_out; H_out_ = H_out; W_out_ = W_out;
        k_size_ = k_size; stride_ = stride; padding_ = padding; dilation_ = dilation;
        groups_ = groups; use_bias_ = use_bias;
        C_per_g_ = C / groups;
        K_per_g_ = K / groups;
        total_spatial_ = N * D_out * H_out * W_out;
        patch_size_ = C_per_g_ * k_size * k_size;
        
        num_physical_cores_ = 20;
        usedCoreNum_ = num_physical_cores_ < total_spatial_ ? num_physical_cores_ : total_spatial_;
        tasksPerCore_ = (total_spatial_ + usedCoreNum_ - 1) / usedCoreNum_;
        
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(input), N * C * D * H * W);
        weightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(weight), K * C_per_g_ * k_size * k_size);
        if (use_bias) {
            biasGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(bias), K);
        }
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(output), N * K * D_out * H_out * W_out);
        
        if ASCEND_IS_AIV {
            pipe_.InitBuffer(patchBuf_, patch_size_ * sizeof(dataType));
            pipe_.InitBuffer(wBuf_, patch_size_ * sizeof(dataType));
            pipe_.InitBuffer(prodBuf_, patch_size_ * sizeof(dataType));
            pipe_.InitBuffer(outBuf_, sizeof(dataType));
            // tmp buffer for WholeReduceSum: 2x size
            pipe_.InitBuffer(reduceTmpBuf_, patch_size_ * sizeof(dataType) * 2);
        }
    }
    
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            
            for (int localIdx = 0; localIdx < tasksPerCore_; ++localIdx) {
                int spatial_idx = coreIdx * tasksPerCore_ + localIdx;
                if (spatial_idx >= total_spatial_) continue;
                
                int n = spatial_idx / (D_out_ * H_out_ * W_out_);
                int rem = spatial_idx % (D_out_ * H_out_ * W_out_);
                int d_out = rem / (H_out_ * W_out_);
                rem = rem % (H_out_ * W_out_);
                int h_out = rem / W_out_;
                int w_out = rem % W_out_;
                
                int d_start = d_out * stride_ - padding_;
                int h_start = h_out * stride_ - padding_;
                int w_start = w_out * stride_ - padding_;

                // Load input patch for all groups
                for (int g = 0; g < groups_; ++g) {
                    LoadPatch(n, d_start, h_start, w_start, g);
                    
                    for (int ok = 0; ok < K_per_g_; ++ok) {
                        int oc = g * K_per_g_ + ok;
                        LoadWeight(oc);
                        ComputeDotProduct();
                        
                        if (use_bias_) {
                            dataType b = biasGM_.GetValue(oc);
                            dataType val = outBuf_.Get<dataType>(0).GetValue(0);
                            val = val + b;
                            outBuf_.Get<dataType>(0).SetValue(0, val);
                        }
                        
                        // Write back
                        int out_idx = (((n * K_ + oc) * D_out_ + d_out) * H_out_ + h_out) * W_out_ + w_out;
                        outputGM_.SetValue(out_idx, outBuf_.Get<dataType>(0).GetValue(0));
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void LoadPatch(int n, int d_start, int h_start, int w_start, int g)
    {
        AscendC::LocalTensor<dataType> patch = patchBuf_.Get<dataType>();
        int idx = 0;
        for (int c = 0; c < C_per_g_; ++c) {
            int ic = g * C_per_g_ + c;
            for (int kd = 0; kd < k_size_; ++kd) {
                int d_in = d_start + kd * dilation_;
                for (int kh = 0; kh < k_size_; ++kh) {
                    int h_in = h_start + kh * dilation_;
                    // kW = 1, so kw is always 0
                    int w_in = w_start;
                    if (h_in >= 0 && h_in < H_ && w_in >= 0 && w_in < W_ && d_in >= 0 && d_in < D_) {
                        int in_idx = (((n * C_ + ic) * D_ + d_in) * H_ + h_in) * W_ + w_in;
                        patch.SetValue(idx, inputGM_.GetValue(in_idx));
                    } else {
                        patch.SetValue(idx, static_cast<dataType>(0));
                    }
                    idx++;
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void LoadWeight(int oc)
    {
        AscendC::LocalTensor<dataType> w = wBuf_.Get<dataType>();
        int idx = 0;
        for (int c = 0; c < C_per_g_; ++c) {
            for (int kd = 0; kd < k_size_; ++kd) {
                for (int kh = 0; kh < k_size_; ++kh) {
                    // Weight shape: (K, C_per_g, kD, kH, kW) with kW=1
                    int w_idx = (((oc * C_per_g_ + c) * k_size_ + kd) * k_size_ + kh);
                    w.SetValue(idx, weightGM_.GetValue(w_idx));
                    idx++;
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void ComputeDotProduct()
    {
        AscendC::LocalTensor<dataType> patch = patchBuf_.Get<dataType>();
        AscendC::LocalTensor<dataType> w = wBuf_.Get<dataType>();
        AscendC::LocalTensor<dataType> prod = prodBuf_.Get<dataType>();
        AscendC::LocalTensor<dataType> out = outBuf_.Get<dataType>();
        AscendC::LocalTensor<dataType> tmp = reduceTmpBuf_.Get<dataType>();
        
        // Element-wise multiply
        for (int i = 0; i < patch_size_; ++i) {
            prod.SetValue(i, patch.GetValue(i) * w.GetValue(i));
        }
        AscendC::PipeBarrier<PIPE_V>();
        
        // Reduce sum using WholeReduceSum
        // For float: mask max = 64 elements per repeat
        int remaining = patch_size_;
        dataType sum = static_cast<dataType>(0);
        int offset = 0;
        while (remaining > 0) {
            int count = remaining > 64 ? 64 : remaining;
            // Copy chunk to prod buffer start (or use offset)
            // WholeReduceSum requires src to be 32-byte aligned
            // We process in chunks; for small patches, just accumulate manually
            if (count <= 64) {
                for (int i = 0; i < count; ++i) {
                    sum = sum + prod.GetValue(offset + i);
                }
            }
            offset += count;
            remaining -= count;
        }
        out.SetValue(0, sum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    
private:
    int N_, C_, D_, H_, W_;
    int K_, D_out_, H_out_, W_out_;
    int k_size_, stride_, padding_, dilation_;
    int groups_, use_bias_;
    int C_per_g_, K_per_g_;
    int total_spatial_;
    int num_physical_cores_;
    int usedCoreNum_;
    int tasksPerCore_;
    int patch_size_;
    
    AscendC::GlobalTensor<dataType> inputGM_;
    AscendC::GlobalTensor<dataType> weightGM_;
    AscendC::GlobalTensor<dataType> biasGM_;
    AscendC::GlobalTensor<dataType> outputGM_;
    
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> patchBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prodBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf_;
};
