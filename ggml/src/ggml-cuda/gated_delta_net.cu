#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "gdn-chunk.cuh" // [gdn-chunk]
#include "gated-delta-net-mma.cuh" // [gdn-mma]

#include <mutex>
#include <unordered_map>

// Update log of one GDN layer for a tree batch: header [epoch, -, input state offset (int64)], then per row and head
// [k (S_v), delta (S_v), g]. Rows are in batch order.
#define GDN_LOG_HEADER      4
#define GDN_LOG_EPOCH_SHIFT 10
#define GDN_LOG_MAX_ROWS    16

struct gdn_log {
    const float * s_base     = nullptr; // tensor the input state was read from
    float *       state_out  = nullptr; // rollback slot 0
    int64_t       slot_stride = 0;
    int64_t       H = 0;
    int           device = 0;
    float *       log = nullptr;
};
static std::mutex                               g_gdn_log_mutex;
static std::unordered_map<const float *, gdn_log> g_gdn_logs; // by state_out

static float * gdn_log_get(int device, const float * s_base, float * state_out, int64_t slot_stride, int64_t H, int64_t S_v) {
    std::lock_guard<std::mutex> lock(g_gdn_log_mutex);
    gdn_log & e = g_gdn_logs[state_out];
    if (e.log == nullptr) {
        CUDA_CHECK(cudaMalloc(&e.log, (GDN_LOG_HEADER + (size_t) GDN_LOG_MAX_ROWS * H * (2 * S_v + 1)) * sizeof(float)));
        CUDA_CHECK(cudaMemset(e.log, 0, GDN_LOG_HEADER * sizeof(float)));
    }
    e.s_base      = s_base;
    e.state_out   = state_out;
    e.slot_stride = slot_stride;
    e.H           = H;
    e.device      = device;
    return e.log;
}

struct gdn_path {
    int32_t epoch;
    int32_t n;
    int8_t  rows[GDN_LOG_MAX_ROWS];
};

// all logged layers in one launch (blockIdx.y = layer)
#define GDN_COMMIT_MAX_LAYERS 64
struct gdn_commit_layers {
    const float * s_base[GDN_COMMIT_MAX_LAYERS];
    float *       state_out[GDN_COMMIT_MAX_LAYERS];   // the rollback slot to write
    const float * log[GDN_COMMIT_MAX_LAYERS];
};

// the state after the rows of path, from the state the batch started with: the same updates in the same order as
// gated_delta_net_cuda makes them. Same block layout as that kernel with 4 columns per warp.
static __global__ void __launch_bounds__(128) gdn_commit_cuda(const gdn_commit_layers L, const int64_t H, const gdn_path path) {
    constexpr int S_v = 128, cols_per_warp = 4, lanes_per_col = 32 / cols_per_warp, rows_per_lane = S_v / lanes_per_col;
    const float * s_base    = L.s_base[blockIdx.y];
    float *       state_out = L.state_out[blockIdx.y];
    const float * log       = L.log[blockIdx.y];
    if (((const int32_t *) log)[0] != path.epoch) {
        return; // this layer did not log the batch: its snapshots are in place
    }
    const int h_idx       = blockIdx.x;
    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col         = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const float * src = s_base + *(const int64_t *) (log + 2) + (int64_t) h_idx * S_v * S_v + col * S_v;
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = src[r * lanes_per_col + lane_in_col];
    }
    for (int j = 0; j < path.n; j++) {
        const float * lr = log + GDN_LOG_HEADER + ((int64_t) path.rows[j] * H + h_idx) * (2 * S_v + 1);
        const float g_val     = lr[2 * S_v];
        const float delta_col = lr[S_v + col];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_val * s_shard[r] + lr[r * lanes_per_col + lane_in_col] * delta_col;
        }
    }
    float * dst = state_out + (int64_t) h_idx * S_v * S_v + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        dst[r * lanes_per_col + lane_in_col] = s_shard[r];
    }
}

int ggml_cuda_gdn_commit(ggml_backend_cuda_context & ctx, const int32_t * rows, int n_rows, int n_batch, int epoch) {
    GGML_ASSERT(n_rows >= 1 && n_rows <= GDN_LOG_MAX_ROWS && n_batch <= GDN_LOG_MAX_ROWS);
    gdn_path path = {};
    path.epoch = epoch;
    path.n     = n_rows;
    for (int i = 0; i < n_rows; ++i) {
        path.rows[i] = (int8_t) rows[i];
    }
    // the snapshot that is (rows after the last kept row) back
    const int slot = n_batch - 1 - rows[n_rows - 1];
    int n = 0, nl = 0;
    int64_t H = 0;
    gdn_commit_layers L = {};
    const auto flush = [&]() {
        if (nl > 0) {
            gdn_commit_cuda<<<dim3(H, nl, 128 / 16), dim3(32, 4, 1), 0, ctx.stream()>>>(L, H, path);
            CUDA_CHECK(cudaGetLastError());
            nl = 0;
        }
    };
    std::lock_guard<std::mutex> lock(g_gdn_log_mutex);
    for (const auto & it : g_gdn_logs) {
        const gdn_log & e = it.second;
        if (e.device != ctx.device) {
            continue;
        }
        if (nl == GDN_COMMIT_MAX_LAYERS || (nl > 0 && e.H != H)) {
            flush();
        }
        H = e.H;
        L.s_base[nl]    = e.s_base;
        L.state_out[nl] = e.state_out + slot * e.slot_stride;
        L.log[nl]       = e.log;
        nl++;
        n++;
    }
    flush();
    return n;
}

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int           rs_stride,
                                     const int32_t * state_rows,
                                     int64_t       state_row_stride,
                                     const int32_t * depth,
                                     float *       log) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // [gdn-cols] the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;                // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // state_rows: read the input state straight from the cache row of each sequence (deferred gather)
    const int64_t state_in_offset      = (state_rows ? state_rows[sequence] * state_row_stride : sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    // tree rows: stk[d] is the state after the last token of depth d, which is an ancestor of every later deeper token
    constexpr int max_depth = 16;
    float stk[max_depth][rows_per_lane];

    // update log (see gdn_log): with a tree set for the batch (epoch in depth[0]) no snapshot is written; the update of every
    // row goes to the log and ggml_cuda_gdn_commit builds the state of the kept path from it
    const bool logging = log != nullptr && depth != nullptr && (depth[0] >> GDN_LOG_EPOCH_SHIFT) != 0;
    if (logging && h_idx == 0 && col == 0 && lane_in_col == 0) {
        ((int32_t *) log)[0] = depth[0] >> GDN_LOG_EPOCH_SHIFT;
        *(int64_t *) (log + 2) = state_in_offset;
    }

    for (int t = 0; t < n_tokens; t++) {
        // depth[t]: bits 0..7 depth, bit 8 the state after this row is read again, bit 9 this row does not follow row t - 1
        const int dt = depth != nullptr ? depth[t] : 0;
        if (dt & 512) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = stk[(dt & 255) - 1][r];
            }
        }
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            if (logging) {
                float * lr = log + GDN_LOG_HEADER + ((int64_t) t * H + h_idx) * (2 * S_v + 1);
                if (lane_in_col == 0) {
                    lr[S_v + col] = delta_col;
                }
                if (col == 0) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        lr[r * lanes_per_col + lane_in_col] = k_reg[r];
                    }
                    if (lane_in_col == 0) {
                        lr[2 * S_v] = g_val;
                    }
                }
            }

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if (dt & 256) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                stk[dt & 255][r] = s_shard[r];
            }
        }

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s * rs_stride tokens back.
            // Slots older than the batch are not written; they are caller-owned.
            const int back        = (int) n_tokens - 1 - t;
            const int target_slot = back / rs_stride;
            if (!logging && back % rs_stride == 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t, int cols_per_warp>
static void launch_gated_delta_net_cols(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, int rs_stride, const int32_t * state_rows, int64_t state_row_stride,
        const int32_t * depth, float * log, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // [gdn-cols] shrink the block when the wider block would leave SMs without one, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one block covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth, log);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth, log);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth, log);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth, log);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// [gdn-cols] GGML_CUDA_NO_GDN_COLS4=1: one state column per warp, as before
template <bool KDA, bool keep_rs_t, typename... Args>
static void launch_gated_delta_net(Args... args) {
    static const bool one = getenv("GGML_CUDA_NO_GDN_COLS4") != nullptr;
    if (one) {
        launch_gated_delta_net_cols<KDA, keep_rs_t, 1>(args...);
    } else {
        launch_gated_delta_net_cols<KDA, keep_rs_t, 4>(args...);
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // the state gather was deferred (see ggml_cuda_gdn_state_gather_deferred): read the cache rows directly
    const int32_t * state_rows       = nullptr;
    int64_t         state_row_stride = 0;
    {
        const ggml_tensor * gr = src_state->op == GGML_OP_GET_ROWS ? src_state : src_state->view_src;
        const int32_t * rows = gr && gr->op == GGML_OP_GET_ROWS ? ggml_cuda_gdn_state_gather_rows(gr) : nullptr;
        if (rows) {
            s_d              = (const float *) gr->src[0]->data;
            state_rows       = rows;
            state_row_stride = gr->src[0]->nb[1] / sizeof(float);
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // tree rows (see ggml_gated_delta_net_set_depth): short batches only, they never reach the chunked kernels below
    const int32_t * depth_d = dst->src[6] ? (const int32_t *) dst->src[6]->data : nullptr;
    GGML_ASSERT(depth_d == nullptr || n_tokens <= 16);
    float * log_d = nullptr; // set below, once the snapshot destination is known

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;
    const int rs_stride = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(rs_stride >= 1);
    // tokens that the recurrent kernel must run so that it writes all snapshot slots
    const int n_tail = (K - 1) * rs_stride + 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }
    // GGML_CUDA_NO_GDN_LOG=1: tree batches write their snapshots as chains do
    static const bool log_off = getenv("GGML_CUDA_NO_GDN_LOG") != nullptr;
    if (!log_off && depth_d != nullptr && cache != nullptr && keep_rs && rs_stride == 1 && !kda && state_rows != nullptr && n_seqs == 1 &&
            S_v == 128 && neq0 == 128 && n_tokens <= GDN_LOG_MAX_ROWS) {
        log_d = gdn_log_get(ctx.device, s_d, state_d, state_slot_stride, H, S_v);
    }

    // [gdn-chunk] env GGML_CUDA_GDN_CHUNK=1 on Volta (gdn-chunk.cu): tokens [0, n_c) through the chunked tensor-core kernel into a
    // temporary state, the remaining n_tokens - n_c >= K tokens through the recurrent kernel from that state with the original
    // state output / slot stride / K, so attention rows and snapshot slots land where the unsplit run writes them
    {
        const int64_t n_c = ggml_cuda_gdn_chunk_tokens(ctx.device, kda, S_v, neq0, H, neqk1, n_tokens, n_seqs, n_tail);
        if (n_c > 0) {
            ggml_cuda_pool_alloc<float> s_mid(ctx.pool(), S_v * S_v * H);

            ggml_cuda_gdn_chunk_args ca;
            ca.q                = q_d;
            ca.k                = k_d;
            ca.v                = v_d;
            ca.g                = g_d;
            ca.beta             = b_d;
            ca.state            = s_d;
            ca.state_rows       = state_rows;
            ca.state_row_stride = state_row_stride;
            ca.dst              = dst_d;
            ca.state_out        = s_mid.get();
            ca.H                = H;
            ca.H_k              = neqk1;
            ca.n_tokens         = n_c;
            ca.sq1              = sq1;
            ca.sq2              = sq2;
            ca.sv1              = sv1;
            ca.sv2              = sv2;
            ca.sb1              = sb1;
            ca.sb2              = sb2;
            ca.scale            = scale;
            ggml_cuda_gdn_chunk_launch(ctx.device, ca, n_tokens, K, stream);

            const int64_t n_r = n_tokens - n_c; // >= K
            if (keep_rs) {
                launch_gated_delta_net<false, true>(q_d + n_c * sq2, k_d + n_c * sq2, v_d + n_c * sv2, g_d + n_c * sb2,
                    b_d + n_c * sb2, (const float *) s_mid.get(), dst_d + n_c * S_v * H, state_d,
                    S_v, H, n_r, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, (const int32_t *) nullptr, (int64_t) 0, (const int32_t *) nullptr, (float *) nullptr, stream);
            } else {
                launch_gated_delta_net<false, false>(q_d + n_c * sq2, k_d + n_c * sq2, v_d + n_c * sv2, g_d + n_c * sb2,
                    b_d + n_c * sb2, (const float *) s_mid.get(), dst_d + n_c * S_v * H, state_d,
                    S_v, H, n_r, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, (const int32_t *) nullptr, (int64_t) 0, (const int32_t *) nullptr, (float *) nullptr, stream);
            }
            return;
        }
    }

    // [gdn-mma] tokens [0, n_c) through upstream's tensor-core kernel into a temporary state, the remaining n_tokens - n_c >= K tokens
    // through the recurrent kernel from that state with the original state output / slot stride / K, so attention rows and rollback
    // snapshots land where the unsplit run writes them. Upstream's kernel writes slot 0 only, which would leave stale rollback snapshots.
    {
        static const bool off = getenv("GGML_CUDA_NO_GDN_MMA") != nullptr;
        const int64_t n_c = n_tokens > n_tail ? ((n_tokens - n_tail) / 16) * 16 : 0;
        if (!off && !kda && S_v == 128 && neq0 == 128 && n_seqs == 1 && rq3 == 1 && K >= 1 && n_c >= 64) {
            ggml_cuda_pool_alloc<float> s_mid(ctx.pool(), S_v * S_v * H);
            ggml_cuda_gdn_mma_args a = {};
            a.q                = q_d;
            a.k                = k_d;
            a.v                = v_d;
            a.g                = g_d;
            a.beta             = b_d;
            a.state            = s_d;
            a.state_rows       = state_rows;
            a.state_row_stride = state_row_stride;
            a.dst              = dst_d;
            a.state_out        = s_mid.get();
            a.H                = H;
            a.H_k              = neqk1;
            a.n_tokens         = n_c;
            a.n_seqs           = 1;
            a.rq3              = 1;
            a.sq1              = sq1;
            a.sq2              = sq2;
            a.sq3              = sq3;
            a.sv1              = sv1;
            a.sv2              = sv2;
            a.sv3              = sv3;
            a.sb1              = sb1;
            a.sb2              = sb2;
            a.sb3              = sb3;
            a.scale            = scale;
            a.eligible         = true;
            if (ggml_cuda_gdn_mma_launch(ctx.device, a, stream)) {
                const int64_t n_r = n_tokens - n_c; // >= K
                if (keep_rs) {
                    launch_gated_delta_net<false, true>(q_d + n_c * sq2, k_d + n_c * sq2, v_d + n_c * sv2, g_d + n_c * sb2,
                        b_d + n_c * sb2, (const float *) s_mid.get(), dst_d + n_c * S_v * H, state_d,
                        S_v, H, n_r, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, (const int32_t *) nullptr, (int64_t) 0, (const int32_t *) nullptr, (float *) nullptr, stream);
                } else {
                    launch_gated_delta_net<false, false>(q_d + n_c * sq2, k_d + n_c * sq2, v_d + n_c * sv2, g_d + n_c * sb2,
                        b_d + n_c * sb2, (const float *) s_mid.get(), dst_d + n_c * S_v * H, state_d,
                        S_v, H, n_r, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, (const int32_t *) nullptr, (int64_t) 0, (const int32_t *) nullptr, (float *) nullptr, stream);
                }
                return;
            }
        }
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth_d, log_d, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth_d, log_d, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth_d, log_d, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rs_stride, state_rows, state_row_stride, depth_d, log_d, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
