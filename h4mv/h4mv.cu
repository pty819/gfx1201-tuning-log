// h4mv — production W4 (e2m1 + e8m0) matvec for the vllm-radiance op boundary.
// Contract: x [M,K] bf16 (contiguous), W [N,K/2] u8, St [N,K/32] u8 e8m0
// (pre-transposed from checkpoint's [K/32,N]), out [M,N] bf16.  1 <= M <= 8.
//
// v2 design (per RDNA4 ISA research, 2026-09-14):
//   * one wave per output row; warpSize-agnostic (gfx1201 hipcc = wave32!)
//   * a uint4 load (16B) = 32 weights = exactly one e8m0 group
//     -> scales are 1/16 of weight traffic, coalesced via the [N,K/32] transpose
//   * LDS LUT keyed by raw weight BYTE -> bf16x2 pair (zero & subnormal e2m1
//     nibbles baked into the table). e2m1 is exact in bf16.
//   * V_DOT2_F32_BF16 via inline asm (no clang builtin): x stays bf16 with
//     ZERO conversion, products are exact (1-bit x 7-bit mantissa), f32 acc.
//   * group scale applied once per (group, m): 2^(b-127) is exactly the f32
//     whose bits are (b << 23) — one shift, one mul.
//   * template<int MT> per batch row: exact registers, fully unrolled
//     (runtime-M predication measured 265/110 GB/s for M=1/4 — dead end).
//   * plain kernel launch on the caller's stream: CUDA-graph capturable.

#include <hip/hip_runtime.h>
#include <cstdint>

#define H4MV_MAX_M 8

typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2;

__device__ __forceinline__ float dot2_bf16(uint32_t a, uint32_t b, float c) {
    float d;
    __asm__("v_dot2_f32_bf16 %[d], %[a], %[b], %[c]"
            : [d] "=v"(d) : [a] "v"(a), [b] "v"(b), [c] "v"(c));
    return d;
}

__device__ __forceinline__ uint32_t e2m1_to_bf16_raw(uint32_t v) {
    const uint32_t mag = v & 7u;
    if (mag == 0u) return 0u;                          // +0 and -0 (0b1000)
    const uint32_t s = v >> 3;
    const uint32_t e = (v >> 1) & 3u;
    const uint32_t m = v & 1u;
    const uint32_t M = (e == 0u) ? 0u : (m << 6);      // subnormal 0.5 = 1.0*2^-1
    return (s << 15) | ((e + 126u) << 7) | M;          // bf16 bias 127
}

template <int MT>
__global__ void __launch_bounds__(256) h4mv_t(
    const uint32_t* __restrict__ W,   // [N, K/8] u32
    const uint8_t*  __restrict__ St,  // [N, K/32]
    const uint32_t* __restrict__ X32, // [M, K/2] u32 (bf16 pairs)
    uint16_t*       __restrict__ Out, // [M, N] bf16 bits
    int N, int K)
{
    __shared__ uint32_t lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e2m1_to_bf16_raw((uint32_t)i & 0xFu) | (e2m1_to_bf16_raw((uint32_t)i >> 4) << 16);
    __syncthreads();

    const int wpb  = blockDim.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int row  = blockIdx.x * wpb + wid;
    if (row >= N) return;

    const int ngrp = K >> 5;
    const uint32_t* wrow = W + (size_t)row * (size_t)(K >> 3);
    const uint8_t*  srow = St + (size_t)row * (size_t)ngrp;

    float acc[MT];
    #pragma unroll
    for (int m = 0; m < MT; m++) acc[m] = 0.0f;

    for (int g = lane; g < ngrp; g += warpSize) {
        const uint4 w = *reinterpret_cast<const uint4*>(wrow + (g << 2));
        const float sc = __uint_as_float((uint32_t)srow[g] << 23);   // 2^(b-127), exact
        float part[MT];
        #pragma unroll
        for (int m = 0; m < MT; m++) part[m] = 0.0f;
        const uint32_t xbase = (g << 4);                              // words into a row
        #pragma unroll
        for (int j = 0; j < 4; j++) {                                 // j-th weight u32
            const uint32_t wv = (j == 0) ? w.x : (j == 1) ? w.y : (j == 2) ? w.z : w.w;
            #pragma unroll
            for (int b = 0; b < 4; b++) {                             // byte <-> x word
                const uint32_t wp = lut[(wv >> (8 * b)) & 0xFFu];
                const uint32_t xw0 = xbase + (j << 2) + b;
                #pragma unroll
                for (int m = 0; m < MT; m++)
                    part[m] = dot2_bf16(wp, X32[(size_t)m * (K >> 1) + xw0], part[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < MT; m++) acc[m] += part[m] * sc;
    }

    #pragma unroll
    for (int m = 0; m < MT; m++) {
        float v = acc[m];
        #pragma unroll
        for (int off = warpSize / 2; off > 0; off >>= 1)
            v += __shfl_xor(v, off);
        if (lane == 0) {
            const uint32_t f = __float_as_uint(v);
            Out[(size_t)m * N + row] = (uint16_t)((f >> 16) + ((f >> 15) & 1u)); // f32->bf16 RN
        }
    }
}

extern "C" void h4mv_launch(const void* W, const void* St, const void* Xb, void* Out,
                            int N, int K, int M, void* stream) {
    const dim3 block(256);
    const dim3 grid((N + 3) / 4);      // surplus waves exit; safe at any wave size
    const uint32_t* X32 = (const uint32_t*)Xb;
    hipStream_t s = (hipStream_t)stream;
    switch (M) {
        case 1: h4mv_t<1><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        case 2: h4mv_t<2><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        case 3: h4mv_t<3><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        case 4: h4mv_t<4><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        case 5: h4mv_t<5><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        case 6: h4mv_t<6><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        case 7: h4mv_t<7><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
        default: h4mv_t<8><<<grid, block, 0, s>>>((const uint32_t*)W, (const uint8_t*)St, X32, (uint16_t*)Out, N, K); break;
    }
}
