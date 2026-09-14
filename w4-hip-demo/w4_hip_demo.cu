// W4 (e2m1 + e8m0 scale) matvec — demo kernels for gfx1201 (RX 9070 GRE)
//
// Data layout (matches Quark AWQ-MXFP4 checkpoint, WPERM=0):
//   W  : [N, K/2] uint8, row-major. Each byte = two e2m1 nibbles; LOW nibble = even k.
//   St : [N, K/32] uint8 e8m0 (pre-transposed on host from checkpoint's [K/32, N]);
//        scale value = 2^(byte - 127).
//   X  : [K] activations (v1: bf16, v2: f16).  Y: [N] f32 output.
//
// Strategy (both versions): ONE WAVE per output row. One uint4 load (16B)
// = 32 weights = EXACTLY one scale group, so scale traffic is 1B per 16B of
// weights and consecutive lanes read consecutive scale bytes (coalesced).
// No LDS staging of activations, no __syncthreads in the hot path.
//
// Wave-size agnostic: gfx1201 hipcc defaults to WAVEFRONT SIZE 32 (verified
// via .amdhsa_wavefront_size32 in the ISA). All lane math uses warpSize; the
// launchers over-provision blocks (ceil(N/4)) and surplus waves exit early.
//
// e2m1 facts used everywhere (both versions):
//   zero: (v & 7) == 0  (covers -0 == 0b1000)
//   subnormal (e==0, m==1) value is 0.5 = 1.0 * 2^-1 -> mantissa must be 0
//
// v1: per-nibble bit-concat to f32, exponent folds the e8m0 scale
//     (E = e + d + 126), fmaf accumulate. ~230 VALU ops per 16B.
// v2: 256-entry LDS LUT keyed by the raw weight BYTE -> f16x2 pair (zero and
//     subnormals live inside the table), scale applied with one packed f16
//     multiply (2^d broadcast; zero-safe, exact), then dot2 (f16 products are
//     exact for e2m1 x 2^d, f32 accumulate). ~3x fewer VALU ops per 16B.

#include <hip/hip_runtime.h>
#include <cstdint>

typedef _Float16 __attribute__((ext_vector_type(2))) f16x2;

__device__ __forceinline__ f16x2 u2h(uint32_t u) {
    union { uint32_t u; f16x2 h; } c; c.u = u; return c.h;
}

// ------------------------------- v1 ------------------------------------------
__device__ __forceinline__ float unpack_e2m1_f32(uint32_t v, int d) {
    const uint32_t mag = v & 7u;
    if (mag == 0u) return 0.0f;                    // +0 and -0 (v == 8)
    const uint32_t s = v >> 3;
    const uint32_t e = (v >> 1) & 3u;
    const uint32_t m = v & 1u;
    const uint32_t M = (e == 0u) ? 0u : (m << 22); // subnormal: mantissa folds into exponent
    const int E = (int)e + d + 126;
    const uint32_t bits = (s << 31) | ((uint32_t)E << 23) | M;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float bf16_lo(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }

extern "C" __global__ void __launch_bounds__(256) w4matvec_v1(
    const uint32_t* __restrict__ W, const uint8_t* __restrict__ St,
    const uint16_t* __restrict__ X, float* __restrict__ Y, int N, int K)
{
    const int wpb  = blockDim.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int row  = blockIdx.x * wpb + wid;
    if (row >= N) return;

    const int ngrp = K >> 5;
    const uint32_t* wrow = W + (size_t)row * (size_t)(K >> 3);
    const uint8_t*  srow = St + (size_t)row * (size_t)ngrp;

    float acc = 0.0f;
    for (int g = lane; g < ngrp; g += warpSize) {
        const uint4 w = *reinterpret_cast<const uint4*>(wrow + (g << 2));
        const int   d = (int)srow[g] - 127;
        const uint16_t* xg = X + (g << 5);
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t wvv = (j == 0) ? w.x : (j == 1) ? w.y : (j == 2) ? w.z : w.w;
            const uint4 xv = *reinterpret_cast<const uint4*>(xg + (j << 3));
            #pragma unroll
            for (int h = 0; h < 4; h++) {
                uint32_t xp = (h == 0) ? xv.x : (h == 1) ? xv.y : (h == 2) ? xv.z : xv.w;
                uint32_t v0 = wvv & 0xFu; wvv >>= 4;
                uint32_t v1 = wvv & 0xFu; wvv >>= 4;
                part = fmaf(unpack_e2m1_f32(v0, d), bf16_lo(xp), part);
                part = fmaf(unpack_e2m1_f32(v1, d), bf16_hi(xp), part);
            }
        }
        acc += part;
    }
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_xor(acc, off);
    if (lane == 0) Y[row] = acc;
}

// ------------------------------- v2 ------------------------------------------
__device__ __forceinline__ uint32_t e2m1_to_f16_raw(uint32_t v) {
    const uint32_t mag = v & 7u;
    if (mag == 0u) return 0u;
    const uint32_t s = v >> 3;
    const uint32_t e = (v >> 1) & 3u;
    const uint32_t m = v & 1u;
    const uint32_t M = (e == 0u) ? 0u : (m << 9);   // subnormal 0.5 = 1.0*2^-1
    return (s << 15) | ((e + 14u) << 10) | M;       // f16 bias 15
}

extern "C" __global__ void __launch_bounds__(256) w4matvec_v2(
    const uint32_t* __restrict__ W, const uint8_t* __restrict__ St,
    const uint16_t* __restrict__ X, float* __restrict__ Y, int N, int K)
{
    __shared__ uint32_t lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e2m1_to_f16_raw((uint32_t)i & 0xFu) | (e2m1_to_f16_raw((uint32_t)i >> 4) << 16);
    __syncthreads();

    const int wpb  = blockDim.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int row  = blockIdx.x * wpb + wid;
    if (row >= N) return;

    const int ngrp = K >> 5;
    const uint32_t* wrow = W + (size_t)row * (size_t)(K >> 3);
    const uint8_t*  srow = St + (size_t)row * (size_t)ngrp;

    float acc = 0.0f;
    for (int g = lane; g < ngrp; g += warpSize) {
        const uint4 w = *reinterpret_cast<const uint4*>(wrow + (g << 2));
        const int d = (int)srow[g] - 127;
        const uint32_t se = (uint32_t)(d + 15) << 10;          // f16 bits of 2^d (d in [-14,15])
        const f16x2 scale2 = u2h(se | (se << 16));
        const uint4* xv = reinterpret_cast<const uint4*>(X + (g << 5));
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {                 // j-th weight u32 (8 weights) ...
            const uint32_t wv = (j == 0) ? w.x : (j == 1) ? w.y : (j == 2) ? w.z : w.w;
            const uint4 xj = xv[j];                   // ... pairs with ITS 8 f16 (uint4)
            const uint32_t xb[4] = {xj.x, xj.y, xj.z, xj.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {             // b-th byte <-> xb[b] f16 pair
                const f16x2 wq = u2h(lut[(wv >> (8 * b)) & 0xFFu]) * scale2;
                part = __builtin_amdgcn_fdot2(wq, u2h(xb[b]), part, false);
            }
        }
        acc += part;
    }
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_xor(acc, off);
    if (lane == 0) Y[row] = acc;
}

// ----------------------------- launchers --------------------------------------
extern "C" __global__ void get_warpsize(int* out) { out[0] = warpSize; }

static inline dim3 w4_grid(int N) { return dim3((N + 3) / 4); }   // surplus waves exit

extern "C" void w4matvec_v1_launch(const void* W, const void* St, const void* X, void* Y,
                                   int N, int K, void* stream) {
    w4matvec_v1<<<w4_grid(N), 256, 0, (hipStream_t)stream>>>(
        (const uint32_t*)W, (const uint8_t*)St, (const uint16_t*)X, (float*)Y, N, K);
}

extern "C" void w4matvec_v2_launch(const void* W, const void* St, const void* X, void* Y,
                                   int N, int K, void* stream) {
    w4matvec_v2<<<w4_grid(N), 256, 0, (hipStream_t)stream>>>(
        (const uint32_t*)W, (const uint8_t*)St, (const uint16_t*)X, (float*)Y, N, K);
}

extern "C" void get_warpsize_launch(void* out, void* stream) {
    get_warpsize<<<1, 64, 0, (hipStream_t)stream>>>((int*)out);
}
