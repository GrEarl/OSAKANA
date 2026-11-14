#include "fp4a8.cuh"

#include <cuda_fp8.h>

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Tile layout of a weight matrix [k, n_out] with nb = k / 32 blocks per row:
//   words:  [n_out / 16 tiles][nb blocks][32 x uint64], lane t of a warp reads word t and gets the codes of its 4 MMA registers
//   MXFP6 only: [n_out / 16 tiles][nb blocks][32 x uint32], bits 64..95 of the 16 codes of 6 bits of lane t
//   scales: [n_out rows][nb] E8M0 bytes
// This is the same size as the block_mxfp4 / block_mxfp6 rows it replaces.

static std::mutex                      g_fp4a8_mutex;
static std::unordered_set<const void *> g_fp4a8_ready;

// MXV payload (see mxv_quant.py): header, column permutation, scales [n_out][nb], then per segment its code arrays [tiles][blocks][32 lanes].
// A segment is a range of blocks with one bit width: 2, 3, 4 (uniform odd levels as e2m3), 6 (e2m3) or 8 (e4m3).
#define FP4A8_MXV_MAX_SEG 5
struct fp4a8_mxv {
    int              n_seg = 0;
    int              bits[FP4A8_MXV_MAX_SEG];
    int              blk0[FP4A8_MXV_MAX_SEG + 1]; // first block of each segment, then nb
    const void *     a0[FP4A8_MXV_MAX_SEG];
    const void *     a1[FP4A8_MXV_MAX_SEG];
    const uint8_t *  scales = nullptr;
    const uint16_t * perm   = nullptr;
    uint64_t         perm_id = 0; // hash of the permutation: matrices with the same input share the quantized activations
};
static std::unordered_map<const void *, fp4a8_mxv> g_fp4a8_mxv;

bool ggml_cuda_fp4a8_enabled(int device) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_FP4A8"); return e != nullptr && atoi(e) != 0; }();
    return on && blackwell_mma_available(ggml_cuda_info().devices[device].cc);
}

bool ggml_cuda_fp4a8_ready(const ggml_tensor * src0) {
    if (src0->type != GGML_TYPE_MXFP4 && src0->type != GGML_TYPE_MXFP6 && !ggml_is_mxv(src0->type)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
    return g_fp4a8_ready.count(src0->data) > 0;
}

// BITS = 4: weights are e2m1, BITS = 6: e2m3
template <int BITS> static __device__ __forceinline__ void fp4a8_mma(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) {
#ifdef BLACKWELL_MMA_AVAILABLE
    if constexpr (BITS == 4) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(sa), "r"(sb));
    } else if constexpr (BITS == 6) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(sa), "r"(sb));
    } else {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(sa), "r"(sb));
    }
#else
    GGML_UNUSED_VARS(D, A, B, sa, sb);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// 4 packed fp4 codes -> 4 bytes with the code in bits 2..5, as the instruction reads fp4 in an 8-bit container
static __device__ __forceinline__ uint32_t fp4a8_spread(uint32_t v) {
    uint32_t x = v & 0xFFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    return x << 2;
}

// 4 packed e2m3 codes (24 bits) -> 4 bytes with the code in bits 0..5
static __device__ __forceinline__ uint32_t fp6a8_spread(uint32_t v) {
    return (v & 0x3F) | ((v & 0xFC0) << 2) | ((v & 0x3F000) << 4) | ((v & 0xFC0000) << 6);
}

// MXV widths 2 to 4: 4 packed fields (sign above the level index m) -> 4 e2m3 codes of the levels (2m+1)/8.
// Uniform levels measured better on code than the e2m1 levels, and the unpack is not what limits the kernel.
static __device__ __forceinline__ uint32_t mxv_spread2(uint32_t v) {
    const uint32_t y = (v & 0x3) | ((v & 0xC) << 6) | ((v & 0x30) << 12) | ((v & 0xC0) << 18);
    return ((y & 0x02020202) << 4) | ((y & 0x01010101) << 1) | 0x01010101;
}
static __device__ __forceinline__ uint32_t mxv_spread3(uint32_t v) {
    const uint32_t y = (v & 0x7) | ((v & 0x38) << 5) | ((v & 0x1C0) << 10) | ((v & 0xE00) << 15);
    return ((y & 0x04040404) << 3) | ((y & 0x03030303) << 1) | 0x01010101;
}
static __device__ __forceinline__ uint32_t mxv_spread4(uint32_t v) {
    uint32_t y = v & 0xFFFF;
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    return ((y & 0x08080808) << 2) | ((y & 0x07070707) << 1) | 0x01010101;
}

// Quantized activations are stored in the order the warps read them ("lane order"), for NT tiles of 8 rows:
//   xq: [block][lane][tile] 8 bytes = the two B registers of that lane for that tile, so a lane loads all its tiles of a block
//       with NT / 2 loads of 16 bytes. With one load per register, the loads were what made rows expensive.
//   xs: [g = lane / 4][block][tile] one E8M0 byte
// B and sb get the registers and the scale of every tile for block b.
template <int NT> static __device__ __forceinline__ void fp4a8_load_b(
        uint32_t (*B)[2], uint32_t * sb, const uint8_t * xq, const uint8_t * xs, const int b, const int lane, const int g, const int nb) {
    const uint8_t * bp = xq + ((int64_t) b * 32 + lane) * (NT * 8);
    const uint8_t * sp = xs + ((int64_t) g * nb + b) * NT;
    if constexpr (NT == 1) {
        const uint64_t v = *(const uint64_t *) bp;
        B[0][0] = (uint32_t) v;
        B[0][1] = (uint32_t) (v >> 32);
        sb[0]   = *sp;
    } else {
#pragma unroll
        for (int q = 0; q < NT / 2; ++q) {
            const uint4 v = *(const uint4 *) (bp + 16 * q);
            B[2 * q][0]     = v.x;
            B[2 * q][1]     = v.y;
            B[2 * q + 1][0] = v.z;
            B[2 * q + 1][1] = v.w;
        }
        const uint64_t w = NT == 2 ? *(const uint16_t *) sp : NT == 4 ? *(const uint32_t *) sp : *(const uint64_t *) sp;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            sb[n] = (uint32_t) (w >> (8 * n)) & 0xFF;
        }
    }
}

// block_mxfp4 / block_mxfp6 rows -> tile layout
template <int BITS> static __global__ void fp4a8_repack(const uint8_t * __restrict__ src, uint64_t * __restrict__ words, uint32_t * __restrict__ words_hi,
                                                        uint8_t * __restrict__ scales, const int n_out, const int nb) {
    constexpr int blk_size = 1 + 4 * BITS;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; // (tile, block, lane)
    const int lane = i & 31;
    const int64_t tb = i >> 5;
    const int b = tb % nb;
    const int tile = tb / nb;
    if (tile * 16 >= n_out) {
        return;
    }
    const int g = lane >> 2, tt = lane & 3;
    uint64_t lo = 0;
    uint32_t hi = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = tile * 16 + g + (r & 1) * 8;
        const uint8_t * blk = src + ((int64_t) row * nb + b) * blk_size;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = 4 * tt + (r >> 1) * 16 + j; // element of the block
            if constexpr (BITS == 4) {
                // qs[j] holds element j in the low nibble and element j + 16 in the high nibble
                const uint8_t code = e < 16 ? (blk[1 + e] & 0x0F) : (blk[1 + e - 16] >> 4);
                lo |= (uint64_t) code << (16 * r + 4 * j);
            } else {
                const int      bit  = 6 * e;
                const uint32_t v    = blk[1 + (bit >> 3)] | ((bit >> 3) + 1 < 24 ? (uint32_t) blk[2 + (bit >> 3)] << 8 : 0);
                const uint64_t code = (v >> (bit & 7)) & 63;
                const int      pos  = 24 * r + 6 * j;
                if (pos < 64) {
                    lo |= code << pos;
                }
                if (pos + 6 > 64) {
                    hi |= pos >= 64 ? (uint32_t) code << (pos - 64) : (uint32_t) (code >> (64 - pos));
                }
            }
        }
    }
    words[i] = lo;
    if constexpr (BITS == 6) {
        words_hi[i] = hi;
    }
    if (lane < 16) {
        const int row = tile * 16 + lane;
        scales[(int64_t) row * nb + b] = src[((int64_t) row * nb + b) * blk_size];
    }
}

// activations [k] per token -> e4m3 bytes and one E8M0 scale per 32 values, in lane order for nt tiles of 8 rows (see fp4a8_load_b);
// rows from n_tok up to nt * 8 are zero. One warp per (row, block of 32 values).
static __global__ void fp4a8_quantize(const float * __restrict__ x, const int64_t stride, uint8_t * __restrict__ xq, uint8_t * __restrict__ xs,
                                      const int k, const int n_tok, const int nt, const uint16_t * __restrict__ perm) {
    const int nb = k / 32;
    const int lane = threadIdx.x & 31;
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5; // (row, block)
    const int b = i % nb;
    const int row = i / nb;
    if (row >= nt * 8) {
        return;
    }
    const int n = row >> 3, g = row & 7;
    // value 'lane' of the block is byte lane & 3 of register (lane >> 4) of the warp lane 4 * g + ((lane & 15) >> 2)
    uint8_t * q  = xq + (((int64_t) b * 32 + g * 4 + ((lane & 15) >> 2)) * nt + n) * 8 + (lane >> 4) * 4 + (lane & 3);
    uint8_t * sc = xs + ((int64_t) g * nb + b) * nt + n;
    if (row >= n_tok) {
        *q = 0;
        if (lane == 0) {
            *sc = 127;
        }
        return;
    }
    const float v = x[row * stride + (perm ? (int) perm[b * 32 + lane] : b * 32 + lane)];
    const float amax = warp_reduce_max<32>(fabsf(v));
    // smallest power of two that brings the block into the e4m3 range (448 = 0.875 * 2^9)
    int e = 0;
    if (amax > 0.0f) {
        int ex;
        const float m = frexpf(amax, &ex);
        e = m <= 0.875f ? ex - 9 : ex - 8;
        e = max(-126, min(127, e));
    }
    const __nv_fp8_e4m3 f(ldexpf(v, -e));
    *q = f.__x;
    if (lane == 0) {
        *sc = (uint8_t) (e + 127);
    }
}

// TC tiles of 16 output rows per block, NT tiles of 8 activation rows. S warps split the blocks of a tile along k, so that a matrix with
// few output rows still has enough reads in flight; their sums are added in warp order. P blocks are loaded while the P blocks before
// them are used. A result does not depend on the number of activation rows.
template <int BITS, int NT, int S, int TC> static __global__ void __launch_bounds__(32 * S * TC)
fp4a8_gemm(const uint64_t * __restrict__ w, const uint32_t * __restrict__ w_hi, const uint8_t * __restrict__ ws, const uint8_t * __restrict__ xq, const uint8_t * __restrict__ xs,
           float * __restrict__ y, const int64_t y_stride, const int n_out, const int k, const int n_tok) {
    constexpr int P = 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tt = lane & 3;
    const int split = warp % S, tl = warp / S;
    const int tile = blockIdx.x * TC + tl;
    const int nb = k / 32;
    const int nb_s = nb / S;
    const int b_beg = split * nb_s, b_end = b_beg + nb_s;
    const bool live = tile * 16 < n_out;

    float D[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        D[n][0] = D[n][1] = D[n][2] = D[n][3] = 0.0f;
    }

    if (live) {
        const uint64_t * wp = w + (int64_t) tile * nb * 32 + lane;
        const uint64_t * sp = (const uint64_t *) (ws + ((int64_t) tile * 16 + g + (tt & 1) * 8) * nb);

        const uint32_t * hp = w_hi + (int64_t) tile * nb * 32 + lane;
        uint64_t wv[P];
        uint32_t hv[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            wv[p] = wp[(int64_t) (b_beg + p) * 32];
            if constexpr (BITS == 6) {
                hv[p] = hp[(int64_t) (b_beg + p) * 32];
            }
        }
        uint64_t sv = sp[b_beg / P];

        for (int b0 = b_beg; b0 < b_end; b0 += P) {
            uint64_t wc[P];
            uint32_t hc[P];
#pragma unroll
            for (int p = 0; p < P; ++p) {
                wc[p] = wv[p];
                if constexpr (BITS == 6) {
                    hc[p] = hv[p];
                }
            }
            const uint64_t sc = sv;
            if (b0 + P < b_end) {
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    wv[p] = wp[(int64_t) (b0 + P + p) * 32];
                    if constexpr (BITS == 6) {
                        hv[p] = hp[(int64_t) (b0 + P + p) * 32];
                    }
                }
                sv = sp[(b0 + P) / P];
            }
#pragma unroll
            for (int p = 0; p < P; ++p) {
                uint32_t A[4];
                if constexpr (BITS == 4) {
                    A[0] = fp4a8_spread((uint32_t) wc[p]);
                    A[1] = fp4a8_spread((uint32_t) (wc[p] >> 16));
                    A[2] = fp4a8_spread((uint32_t) (wc[p] >> 32));
                    A[3] = fp4a8_spread((uint32_t) (wc[p] >> 48));
                } else {
                    A[0] = fp6a8_spread((uint32_t) wc[p]);
                    A[1] = fp6a8_spread((uint32_t) (wc[p] >> 24));
                    A[2] = fp6a8_spread((uint32_t) (wc[p] >> 48) | (hc[p] << 16));
                    A[3] = fp6a8_spread(hc[p] >> 8);
                }
                const uint32_t sa = (uint32_t) (sc >> (8 * p)) & 0xFF;
                uint32_t B[NT][2];
                uint32_t sb[NT];
                fp4a8_load_b<NT>(B, sb, xq, xs, b0 + p, lane, g, nb);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    fp4a8_mma<BITS>(D[n], A, B[n], sa, sb[n]);
                }
            }
        }
    }

    if constexpr (S > 1) {
        __shared__ float part[TC * S][NT * 4][32];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                part[warp][n * 4 + i][lane] = D[n][i];
            }
        }
        __syncthreads();
        if (split != 0) {
            return;
        }
#pragma unroll
        for (int s = 1; s < S; ++s) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    D[n][i] += part[tl * S + s][n * 4 + i][lane];
                }
            }
        }
    }
    if (!live) {
        return;
    }

#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = tile * 16 + g + (i >> 1) * 8;
            const int col = n * 8 + 2 * tt + (i & 1);
            if (col < n_tok) {
                y[col * y_stride + row] = D[n][i];
            }
        }
    }
}

// split and tiles per block as measured on the RTX 5090: about 1024 warps or more, 4 tiles per block while the split is small
template <int BITS, int NT> static void fp4a8_launch(int n_split, cudaStream_t stream, const uint64_t * w, const uint32_t * w_hi, const uint8_t * ws,
                                                     const uint8_t * xq, const uint8_t * xs, float * y, const int n_out, const int k, const int n_tok) {
    const int n_tiles = n_out / 16;
    switch (n_split) {
        case 1:  fp4a8_gemm<BITS, NT, 1, 4><<<(n_tiles + 3) / 4, 128, 0, stream>>>(w, w_hi, ws, xq, xs, y, n_out, n_out, k, n_tok); break;
        case 2:  fp4a8_gemm<BITS, NT, 2, 4><<<(n_tiles + 3) / 4, 256, 0, stream>>>(w, w_hi, ws, xq, xs, y, n_out, n_out, k, n_tok); break;
        case 4:  fp4a8_gemm<BITS, NT, 4, 1><<<n_tiles, 128, 0, stream>>>(w, w_hi, ws, xq, xs, y, n_out, n_out, k, n_tok); break;
        default: GGML_ABORT("fp4a8: bad split %d", n_split);
    }
}

template <int BITS> static void fp4a8_launch_nt(int nt, int n_split, cudaStream_t stream, const uint64_t * w, const uint32_t * w_hi, const uint8_t * ws,
                                                const uint8_t * xq, const uint8_t * xs, float * y, const int n_out, const int k, const int n_tok) {
    switch (nt) {
        case 1:  fp4a8_launch<BITS, 1>(n_split, stream, w, w_hi, ws, xq, xs, y, n_out, k, n_tok); break;
        case 2:  fp4a8_launch<BITS, 2>(n_split, stream, w, w_hi, ws, xq, xs, y, n_out, k, n_tok); break;
        case 4:  fp4a8_launch<BITS, 4>(n_split, stream, w, w_hi, ws, xq, xs, y, n_out, k, n_tok); break;
        default: fp4a8_launch<BITS, 8>(n_split, stream, w, w_hi, ws, xq, xs, y, n_out, k, n_tok); break;
    }
}

struct fp4a8_mxv_args {
    int          n_seg;
    int          bits[FP4A8_MXV_MAX_SEG];
    int          blk0[FP4A8_MXV_MAX_SEG + 1];
    const void * a0[FP4A8_MXV_MAX_SEG];
    const void * a1[FP4A8_MXV_MAX_SEG];
};

// Independent accumulators per token tile: with one, the products of a warp form a single dependency chain and the
// kernel is bound by the latency of the instruction, not by memory (a smaller file was no faster).
#ifndef FP4A8_NA1
#define FP4A8_NA1 4
#endif
#ifndef FP4A8_NA2
#define FP4A8_NA2 2
#endif
template <int NT> constexpr int mxv_na = NT == 1 ? FP4A8_NA1 : NT == 2 ? FP4A8_NA2 : 1;

// blocks [lo, hi) of one segment of width BITS for one warp; nbs = blocks of the segment, seg0 = its first block
template <int BITS, int NT, int P> static __device__ __forceinline__ void mxv_run_p(
        float (*D)[4], const void * a0, const void * a1, const uint64_t * sp, const uint8_t * xq, const uint8_t * xs,
        const int tile, const int lane, const int g, const int tt, const int nbs, const int seg0, const int lo, const int hi, const int k, const int nb) {
    // P blocks in flight
    const uint8_t * sp8 = (const uint8_t *) sp;
    const int64_t base = ((int64_t) tile * nbs - seg0) * 32 + lane; // word of block b is base + b * 32
    uint64_t w0[P];
    uint64_t w1[P];
    auto load = [&](int p, int b) {
        const int64_t i = base + (int64_t) b * 32;
        if constexpr (BITS == 2) {
            w0[p] = ((const uint32_t *) a0)[i];
        } else if constexpr (BITS == 3) {
            w0[p] = ((const uint32_t *) a0)[i] | ((uint64_t) ((const uint16_t *) a1)[i] << 32);
        } else if constexpr (BITS == 4) {
            w0[p] = ((const uint64_t *) a0)[i];
        } else if constexpr (BITS == 6) {
            w0[p] = ((const uint64_t *) a0)[i];
            w1[p] = ((const uint32_t *) a1)[i];
        } else {
            w0[p] = ((const uint64_t *) a0)[i];
            w1[p] = ((const uint64_t *) a1)[i];
        }
    };
#pragma unroll
    for (int p = 0; p < P; ++p) {
        load(p, lo + p);
    }
    constexpr int PS = (P + 7) / 8; // scale words of 8 blocks
    uint64_t sv[PS];
#pragma unroll
    for (int i = 0; i < PS; ++i) {
        sv[i] = P >= 8 ? *(const uint64_t *) (sp8 + lo + 8 * i) : *(const uint32_t *) (sp8 + lo);
    }
    for (int b0 = lo; b0 < hi; b0 += P) {
        uint64_t c0[P];
        uint64_t c1[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            c0[p] = w0[p];
            if constexpr (BITS >= 6) {
                c1[p] = w1[p];
            }
        }
        uint64_t sc[PS];
#pragma unroll
        for (int i = 0; i < PS; ++i) {
            sc[i] = sv[i];
        }
        if (b0 + P < hi) {
#pragma unroll
            for (int p = 0; p < P; ++p) {
                load(p, b0 + P + p);
            }
#pragma unroll
            for (int i = 0; i < PS; ++i) {
                sv[i] = P >= 8 ? *(const uint64_t *) (sp8 + b0 + P + 8 * i) : *(const uint32_t *) (sp8 + b0 + P);
            }
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
            uint32_t A[4];
            if constexpr (BITS == 2) {
                const uint32_t w = (uint32_t) c0[p];
                A[0] = mxv_spread2(w); A[1] = mxv_spread2(w >> 8); A[2] = mxv_spread2(w >> 16); A[3] = mxv_spread2(w >> 24);
            } else if constexpr (BITS == 3) {
                A[0] = mxv_spread3((uint32_t) c0[p]);         A[1] = mxv_spread3((uint32_t) (c0[p] >> 12));
                A[2] = mxv_spread3((uint32_t) (c0[p] >> 24)); A[3] = mxv_spread3((uint32_t) (c0[p] >> 36));
            } else if constexpr (BITS == 4) {
                A[0] = mxv_spread4((uint32_t) c0[p]);         A[1] = mxv_spread4((uint32_t) (c0[p] >> 16));
                A[2] = mxv_spread4((uint32_t) (c0[p] >> 32)); A[3] = mxv_spread4((uint32_t) (c0[p] >> 48));
            } else if constexpr (BITS == 6) {
                A[0] = fp6a8_spread((uint32_t) c0[p]);
                A[1] = fp6a8_spread((uint32_t) (c0[p] >> 24));
                A[2] = fp6a8_spread((uint32_t) (c0[p] >> 48) | ((uint32_t) c1[p] << 16));
                A[3] = fp6a8_spread((uint32_t) c1[p] >> 8);
            } else {
                A[0] = (uint32_t) c0[p]; A[1] = (uint32_t) (c0[p] >> 32); A[2] = (uint32_t) c1[p]; A[3] = (uint32_t) (c1[p] >> 32);
            }
            const uint32_t sa = (uint32_t) (sc[p / 8] >> (8 * (p & 7))) & 0xFF;
            uint32_t B[NT][2];
            uint32_t sb[NT];
            fp4a8_load_b<NT>(B, sb, xq, xs, b0 + p, lane, g, nb);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                fp4a8_mma<BITS == 8 ? 8 : 6>(D[(p % mxv_na<NT>) * NT + n], A, B[n], sa, sb[n]);
            }
        }
    }
}

// 8 blocks in flight; wide batches take 4, their registers are needed for the sums
template <int BITS, int NT> static __device__ __forceinline__ void mxv_run(
        float (*D)[4], const void * a0, const void * a1, const uint64_t * sp, const uint8_t * xq, const uint8_t * xs,
        const int tile, const int lane, const int g, const int tt, const int nbs, const int seg0, const int lo, const int hi, const int k, const int nb) {
    if constexpr (NT >= 4) {
        mxv_run_p<BITS, NT, 4>(D, a0, a1, sp, xq, xs, tile, lane, g, tt, nbs, seg0, lo, hi, k, nb);
    } else {
        mxv_run_p<BITS, NT, 8>(D, a0, a1, sp, xq, xs, tile, lane, g, tt, nbs, seg0, lo, hi, k, nb);
    }
}

// as fp4a8_gemm, for an MXV matrix: the blocks of a warp may cross segments of different width
template <int NT, int S, int TC> static __device__ __forceinline__ void
fp4a8_mxv_body(const fp4a8_mxv_args & m, const uint8_t * __restrict__ ws, const uint8_t * __restrict__ xq, const uint8_t * __restrict__ xs,
               float * __restrict__ y, const int64_t y_stride, const int n_out, const int k, const int n_tok, const int bx) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tt = lane & 3;
    const int split = warp % S, tl = warp / S;
    const int tile = bx * TC + tl;
    const int nb = k / 32;
    const int nb_s = nb / S;
    const int b_beg = split * nb_s, b_end = b_beg + nb_s;
    const bool live = tile * 16 < n_out;

    float D[NT * mxv_na<NT>][4];
#pragma unroll
    for (int n = 0; n < NT * mxv_na<NT>; ++n) {
        D[n][0] = D[n][1] = D[n][2] = D[n][3] = 0.0f;
    }

    if (live) {
        const uint64_t * sp = (const uint64_t *) (ws + ((int64_t) tile * 16 + g + (tt & 1) * 8) * nb);
        for (int s = 0; s < m.n_seg; ++s) {
            const int lo = max(b_beg, m.blk0[s]), hi = min(b_end, m.blk0[s + 1]);
            if (lo >= hi) {
                continue;
            }
            const int nbs = m.blk0[s + 1] - m.blk0[s];
            switch (m.bits[s]) {
                case 2:  mxv_run<2, NT>(D, m.a0[s], m.a1[s], sp, xq, xs, tile, lane, g, tt, nbs, m.blk0[s], lo, hi, k, nb); break;
                case 3:  mxv_run<3, NT>(D, m.a0[s], m.a1[s], sp, xq, xs, tile, lane, g, tt, nbs, m.blk0[s], lo, hi, k, nb); break;
                case 4:  mxv_run<4, NT>(D, m.a0[s], m.a1[s], sp, xq, xs, tile, lane, g, tt, nbs, m.blk0[s], lo, hi, k, nb); break;
                case 6:  mxv_run<6, NT>(D, m.a0[s], m.a1[s], sp, xq, xs, tile, lane, g, tt, nbs, m.blk0[s], lo, hi, k, nb); break;
                default: mxv_run<8, NT>(D, m.a0[s], m.a1[s], sp, xq, xs, tile, lane, g, tt, nbs, m.blk0[s], lo, hi, k, nb); break;
            }
        }
    }

#pragma unroll
    for (int a = 1; a < mxv_na<NT>; ++a) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                D[n][i] += D[a * NT + n][i];
            }
        }
    }

    if constexpr (S > 1) {
        __shared__ float part[TC * S][NT * 4][32];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                part[warp][n * 4 + i][lane] = D[n][i];
            }
        }
        __syncthreads();
        if (split != 0) {
            return;
        }
#pragma unroll
        for (int s = 1; s < S; ++s) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    D[n][i] += part[tl * S + s][n * 4 + i][lane];
                }
            }
        }
    }
    if (!live) {
        return;
    }

#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = tile * 16 + g + (i >> 1) * 8;
            const int col = n * 8 + 2 * tt + (i & 1);
            if (col < n_tok) {
                y[col * y_stride + row] = D[n][i];
            }
        }
    }
}

template <int NT, int S, int TC> static __global__ void __launch_bounds__(32 * S * TC)
fp4a8_gemm_mxv(const fp4a8_mxv_args m, const uint8_t * __restrict__ ws, const uint8_t * __restrict__ xq, const uint8_t * __restrict__ xs,
               float * __restrict__ y, const int64_t y_stride, const int n_out, const int k, const int n_tok) {
    fp4a8_mxv_body<NT, S, TC>(m, ws, xq, xs, y, y_stride, n_out, k, n_tok, blockIdx.x);
}

// Two matrices of one shape that read the same activations, in one launch: blocks from nblk0 on belong to the second.
// More tiles are in flight than with one matrix after the other. A result is the one of the single kernel.
template <int NT, int S, int TC> static __global__ void __launch_bounds__(32 * S * TC)
fp4a8_gemm_mxv_pair(const fp4a8_mxv_args m0, const fp4a8_mxv_args m1, const uint8_t * __restrict__ ws0, const uint8_t * __restrict__ ws1,
                    const uint8_t * __restrict__ xq, const uint8_t * __restrict__ xs, float * __restrict__ y0, float * __restrict__ y1,
                    const int64_t y_stride, const int n_out, const int k, const int n_tok, const int nblk0) {
    const bool second = (int) blockIdx.x >= nblk0;
    fp4a8_mxv_body<NT, S, TC>(second ? m1 : m0, second ? ws1 : ws0, xq, xs, second ? y1 : y0, y_stride, n_out, k, n_tok,
                              second ? (int) blockIdx.x - nblk0 : (int) blockIdx.x);
}

template <int NT> static void fp4a8_launch_mxv_pair(int n_split, cudaStream_t stream, const fp4a8_mxv_args & m0, const fp4a8_mxv_args & m1,
                                                    const uint8_t * ws0, const uint8_t * ws1, const uint8_t * xq, const uint8_t * xs,
                                                    float * y0, float * y1, const int n_out, const int k, const int n_tok) {
    const int n_tiles = n_out / 16;
    const int nb4 = (n_tiles + 3) / 4;
    switch (n_split) {
        case 1:  fp4a8_gemm_mxv_pair<NT, 1, 4><<<2 * nb4, 128, 0, stream>>>(m0, m1, ws0, ws1, xq, xs, y0, y1, n_out, n_out, k, n_tok, nb4); break;
        case 2:  fp4a8_gemm_mxv_pair<NT, 2, 4><<<2 * nb4, 256, 0, stream>>>(m0, m1, ws0, ws1, xq, xs, y0, y1, n_out, n_out, k, n_tok, nb4); break;
        case 4:  fp4a8_gemm_mxv_pair<NT, 4, 1><<<2 * n_tiles, 128, 0, stream>>>(m0, m1, ws0, ws1, xq, xs, y0, y1, n_out, n_out, k, n_tok, n_tiles); break;
        default: GGML_ABORT("fp4a8: bad split %d", n_split);
    }
}

template <int NT> static void fp4a8_launch_mxv(int n_split, cudaStream_t stream, const fp4a8_mxv_args & m, const uint8_t * ws,
                                               const uint8_t * xq, const uint8_t * xs, float * y, const int n_out, const int k, const int n_tok) {
    const int n_tiles = n_out / 16;
    switch (n_split) {
        case 1:  fp4a8_gemm_mxv<NT, 1, 4><<<(n_tiles + 3) / 4, 128, 0, stream>>>(m, ws, xq, xs, y, n_out, n_out, k, n_tok); break;
        case 2:  fp4a8_gemm_mxv<NT, 2, 4><<<(n_tiles + 3) / 4, 256, 0, stream>>>(m, ws, xq, xs, y, n_out, n_out, k, n_tok); break;
        case 4:  fp4a8_gemm_mxv<NT, 4, 1><<<n_tiles, 128, 0, stream>>>(m, ws, xq, xs, y, n_out, n_out, k, n_tok); break;
        default: GGML_ABORT("fp4a8: bad split %d", n_split);
    }
}

// quantized activations of the last input, per backend context: the matrices that share an input quantize it once
struct fp4a8_acts {
    const ggml_tensor * src1  = nullptr;
    uint64_t            perm_id = 0;
    int64_t             t0    = -1;
    uint64_t            epoch = 0;
    uint64_t            epoch_now = 0;
    uint8_t *           xq    = nullptr;
    uint8_t *           xs    = nullptr;
    size_t              cap_q = 0, cap_s = 0;
};
static std::unordered_map<const void *, fp4a8_acts> g_fp4a8_acts;

static bool fp4a8_eligible(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_MXFP4 || t->type == GGML_TYPE_MXFP6 || ggml_is_mxv(t->type)) && t->view_src == nullptr && t->op == GGML_OP_NONE && t->data != nullptr &&
           t->ne[2] == 1 && t->ne[3] == 1 && t->ne[1] % 16 == 0 && t->ne[0] % 256 == 0 && ggml_is_contiguous(t);
}

void ggml_cuda_fp4a8_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    {
        std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
        g_fp4a8_acts[&ctx].epoch_now++;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const ggml_tensor * src0 = node->src[0];
        if (!fp4a8_eligible(src0) || node->src[1]->type != GGML_TYPE_F32) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
            if (g_fp4a8_ready.count(src0->data) > 0) {
                continue;
            }
        }
        const int    n_out = src0->ne[1];
        const int    nb    = src0->ne[0] / 32;
        if (ggml_is_mxv(src0->type)) {
            // the payload is in the tile layout already: read its header and the permutation
            const int k = src0->ne[0];
            std::vector<uint8_t> head(64 + 2 * (size_t) k);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            CUDA_CHECK(cudaMemcpy(head.data(), src0->data, head.size(), cudaMemcpyDeviceToHost));
            const uint32_t * h = (const uint32_t *) head.data();
            GGML_ASSERT(h[0] == 0x3156584D && (int) h[1] == k && (int) h[2] == n_out && h[3] >= 1 && h[3] <= FP4A8_MXV_MAX_SEG);
            fp4a8_mxv info;
            info.n_seg = h[3];
            const uint8_t * base = (const uint8_t *) src0->data;
            size_t off = (64 + 2 * (size_t) k + 255) / 256 * 256;
            info.perm   = (const uint16_t *) (base + 64);
            info.scales = base + off;
            off += (size_t) n_out * nb;
            int blk = 0;
            for (int i = 0; i < info.n_seg; ++i) {
                info.bits[i] = h[4 + 2 * i];
                info.blk0[i] = blk;
                const size_t n_lane = (size_t) (n_out / 16) * h[5 + 2 * i] * 32;
                const size_t sz0 = info.bits[i] <= 3 ? 4 : 8;
                const size_t sz1 = info.bits[i] == 3 ? 2 : info.bits[i] == 6 ? 4 : info.bits[i] == 8 ? 8 : 0;
                info.a0[i] = base + off;
                off += (n_lane * sz0 + 255) / 256 * 256;
                info.a1[i] = base + off;
                off += (n_lane * sz1 + 255) / 256 * 256;
                blk += h[5 + 2 * i];
                GGML_ASSERT(h[5 + 2 * i] % 8 == 0);
            }
            info.blk0[info.n_seg] = blk;
            GGML_ASSERT(blk == nb && off <= ggml_nbytes(src0));
            uint64_t hash = 1469598103934665603ull;
            for (size_t i = 64; i < head.size(); ++i) {
                hash = (hash ^ head[i]) * 1099511628211ull;
            }
            info.perm_id = hash | 1;

            std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
            g_fp4a8_mxv[src0->data] = info;
            g_fp4a8_ready.insert(src0->data);
            continue;
        }
        const bool   six   = src0->type == GGML_TYPE_MXFP6;
        const size_t n_w   = (size_t) n_out * nb * 16;
        const size_t n_h   = six ? (size_t) n_out * nb * 8 : 0;
        const size_t n_s   = (size_t) n_out * nb;
        GGML_ASSERT(n_w + n_h + n_s == ggml_nbytes(src0));

        uint8_t * tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, n_w + n_h + n_s));
        const int64_t n_threads = (int64_t) (n_out / 16) * nb * 32;
        if (six) {
            fp4a8_repack<6><<<(n_threads + 255) / 256, 256, 0, ctx.stream()>>>((const uint8_t *) src0->data, (uint64_t *) tmp, (uint32_t *) (tmp + n_w), tmp + n_w + n_h, n_out, nb);
        } else {
            fp4a8_repack<4><<<(n_threads + 255) / 256, 256, 0, ctx.stream()>>>((const uint8_t *) src0->data, (uint64_t *) tmp, nullptr, tmp + n_w, n_out, nb);
        }
        CUDA_CHECK(cudaMemcpyAsync(src0->data, tmp, n_w + n_h + n_s, cudaMemcpyDeviceToDevice, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaFree(tmp));

        std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
        g_fp4a8_ready.insert(src0->data);
    }
}

void ggml_cuda_mul_mat_fp4a8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(dst));

    const int     k     = src0->ne[0];
    const int     n_out = src0->ne[1];
    const int     nb    = k / 32;
    const int64_t n_tok = src1->ne[1] * src1->ne[2] * src1->ne[3];

    const float *    x  = (const float *) src1->data;
    float *          y  = (float *) dst->data;
    const bool mxv = ggml_is_mxv(src0->type);
    fp4a8_mxv_args margs = {};
    const uint16_t * perm    = nullptr;
    uint64_t         perm_id = 0;
    const uint8_t *  mxv_scales = nullptr;
    if (mxv) {
        std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
        const fp4a8_mxv & info = g_fp4a8_mxv.at(src0->data);
        margs.n_seg = info.n_seg;
        for (int i = 0; i < info.n_seg; ++i) {
            margs.bits[i] = info.bits[i];
            margs.blk0[i] = info.blk0[i];
            margs.a0[i]   = info.a0[i];
            margs.a1[i]   = info.a1[i];
        }
        margs.blk0[info.n_seg] = info.blk0[info.n_seg];
        perm       = info.perm;
        perm_id    = info.perm_id;
        mxv_scales = info.scales;
    }

    const bool       six  = src0->type == GGML_TYPE_MXFP6;
    const uint64_t * w    = (const uint64_t *) src0->data;
    const uint32_t * w_hi = (const uint32_t *) ((const uint8_t *) src0->data + (size_t) n_out * nb * 16);
    const uint8_t *  ws   = (const uint8_t *) src0->data + (size_t) n_out * nb * (six ? 24 : 16);
    cudaStream_t     stream = ctx.stream();

    const int n_max = 64;
    fp4a8_acts * acts;
    {
        std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
        acts = &g_fp4a8_acts[&ctx];
    }
    if (acts->cap_q < (size_t) n_max * k) {
        // the stream may still read the old buffers
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (acts->xq) { CUDA_CHECK(cudaFree(acts->xq)); CUDA_CHECK(cudaFree(acts->xs)); }
        acts->cap_q = (size_t) n_max * k;
        acts->cap_s = (size_t) n_max * nb;
        CUDA_CHECK(cudaMalloc(&acts->xq, acts->cap_q));
        CUDA_CHECK(cudaMalloc(&acts->xs, acts->cap_s));
        acts->src1 = nullptr;
    }

    const int n_tiles = n_out / 16;
    int n_split = 1;
    while (n_split < 4 && n_tiles * n_split < 1024 && (nb / (n_split * 2)) % 8 == 0) {
        n_split *= 2;
    }
    for (int64_t t0 = 0; t0 < n_tok; t0 += n_max) {
        const int n  = (int) std::min<int64_t>(n_max, n_tok - t0);
        const int nt = n <= 8 ? 1 : n <= 16 ? 2 : n <= 32 ? 4 : 8;
        const int n_rows = nt * 8;

        if (acts->src1 != src1 || acts->perm_id != perm_id || acts->t0 != t0 || acts->epoch != acts->epoch_now) {
            fp4a8_quantize<<<(n_rows * nb + 7) / 8, 256, 0, stream>>>(x + t0 * k, k, acts->xq, acts->xs, k, n, nt, perm);
            acts->src1    = src1;
            acts->perm_id = perm_id;
            acts->t0    = t0;
            acts->epoch = acts->epoch_now;
        }
        float * yt = y + t0 * n_out;
        if (mxv) {
            switch (nt) {
                case 1:  fp4a8_launch_mxv<1>(n_split, stream, margs, mxv_scales, acts->xq, acts->xs, yt, n_out, k, n); break;
                case 2:  fp4a8_launch_mxv<2>(n_split, stream, margs, mxv_scales, acts->xq, acts->xs, yt, n_out, k, n); break;
                case 4:  fp4a8_launch_mxv<4>(n_split, stream, margs, mxv_scales, acts->xq, acts->xs, yt, n_out, k, n); break;
                default: fp4a8_launch_mxv<8>(n_split, stream, margs, mxv_scales, acts->xq, acts->xs, yt, n_out, k, n); break;
            }
        } else if (six) {
            fp4a8_launch_nt<6>(nt, n_split, stream, w, w_hi, ws, acts->xq, acts->xs, yt, n_out, k, n);
        } else {
            fp4a8_launch_nt<4>(nt, n_split, stream, w, w_hi, ws, acts->xq, acts->xs, yt, n_out, k, n);
        }
    }
}

static bool fp4a8_mxv_args_of(const ggml_tensor * src0, fp4a8_mxv_args & margs, const uint16_t *& perm, uint64_t & perm_id, const uint8_t *& scales) {
    if (!ggml_is_mxv(src0->type)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
    const auto it = g_fp4a8_mxv.find(src0->data);
    if (it == g_fp4a8_mxv.end()) {
        return false;
    }
    const fp4a8_mxv & info = it->second;
    margs.n_seg = info.n_seg;
    for (int i = 0; i < info.n_seg; ++i) {
        margs.bits[i] = info.bits[i];
        margs.blk0[i] = info.blk0[i];
        margs.a0[i]   = info.a0[i];
        margs.a1[i]   = info.a1[i];
    }
    margs.blk0[info.n_seg] = info.blk0[info.n_seg];
    perm    = info.perm;
    perm_id = info.perm_id;
    scales  = info.scales;
    return true;
}

// two MXV matrices of one shape times the same input (the same column order): one launch. false = not that case, nothing done
bool ggml_cuda_mul_mat_fp4a8_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0a, const ggml_tensor * src0b, const ggml_tensor * src1,
                                  ggml_tensor * dsta, ggml_tensor * dstb) {
    static const bool off = getenv("GGML_CUDA_FP4A8_NO_PAIR") != nullptr;
    if (off || src0a->ne[0] != src0b->ne[0] || src0a->ne[1] != src0b->ne[1] || src1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dsta) || !ggml_is_contiguous(dstb) || dsta->data == dstb->data) {
        return false;
    }
    const int     k     = src0a->ne[0];
    const int     n_out = src0a->ne[1];
    const int     nb    = k / 32;
    const int64_t n_tok = src1->ne[1] * src1->ne[2] * src1->ne[3];
    fp4a8_mxv_args m0 = {}, m1 = {};
    const uint16_t * perm0 = nullptr, * perm1 = nullptr;
    uint64_t pid0 = 0, pid1 = 0;
    const uint8_t * sc0 = nullptr, * sc1 = nullptr;
    if (n_tok < 1 || n_tok > 16 || !fp4a8_mxv_args_of(src0a, m0, perm0, pid0, sc0) || !fp4a8_mxv_args_of(src0b, m1, perm1, pid1, sc1) || pid0 != pid1) {
        return false;
    }
    fp4a8_acts * acts;
    {
        std::lock_guard<std::mutex> lock(g_fp4a8_mutex);
        acts = &g_fp4a8_acts[&ctx];
    }
    if (acts->cap_q < (size_t) 64 * k) {
        return false; // the single path allocates
    }
    cudaStream_t stream = ctx.stream();
    const int n_tiles = n_out / 16;
    int n_split = 1;
    while (n_split < 4 && n_tiles * n_split < 1024 && (nb / (n_split * 2)) % 8 == 0) {
        n_split *= 2;
    }
    const int n  = (int) n_tok;
    const int nt = n <= 8 ? 1 : 2;
    if (acts->src1 != src1 || acts->perm_id != pid0 || acts->t0 != 0 || acts->epoch != acts->epoch_now) {
        fp4a8_quantize<<<(nt * 8 * nb + 7) / 8, 256, 0, stream>>>((const float *) src1->data, k, acts->xq, acts->xs, k, n, nt, perm0);
        acts->src1    = src1;
        acts->perm_id = pid0;
        acts->t0      = 0;
        acts->epoch   = acts->epoch_now;
    }
    if (nt == 1) {
        fp4a8_launch_mxv_pair<1>(n_split, stream, m0, m1, sc0, sc1, acts->xq, acts->xs, (float *) dsta->data, (float *) dstb->data, n_out, k, n);
    } else {
        fp4a8_launch_mxv_pair<2>(n_split, stream, m0, m1, sc0, sc1, acts->xq, acts->xs, (float *) dsta->data, (float *) dstb->data, n_out, k, n);
    }
    return true;
}
