#include "common.cuh"

// YTR: trellis-coded weights from YASAKANI .yskn files (see ytr.cu for the tensor header).

// read the headers of the YTR weights of the graph (before any CUDA graph capture)
void ggml_cuda_ytr_prepare(const ggml_cgraph * cgraph);

// y = W x for a YTR weight: input rotation, decode, product, output rotation
void ggml_cuda_mul_mat_ytr(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// node i is a YTR product: compute it with the later products of the same input in one launch, folding the following
// ADD + RMS_NORM + MUL or SWIGLU (and the next product's input rotation) into its finish. Returns the nodes to skip after i,
// or -1 if not taken.
// sides: small dense products of the same input (F32 / F16 / BF16 weights of few rows, e.g. the GDN alpha and beta projections) that
// ride along as extra blocks of node i's launch, with their activation: 0 none, 1 sigmoid, 2 softplus(x + pre[row]) * mul[row].
// They are computed into a private buffer (up to 16 input rows) and copied out by ggml_cuda_ytr_side_finish when the graph
// reaches their node, so the allocator's reuse of the result memory in between does not matter.
struct ggml_cuda_ytr_side {
    const ggml_tensor * node;      // the MUL_MAT
    const ggml_tensor * w;
    int                 post_op;
    const float *       pre;
    const float *       mul;
};
#define GGML_CUDA_YTR_MAX_SIDE 2
int ggml_cuda_ytr_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const ggml_cuda_ytr_side * sides = nullptr, int n_sides = 0,
                           bool * sides_taken = nullptr);

// copy the result of a side product computed earlier to out ([rows] per input row, out_col floats apart); false if node has none
bool ggml_cuda_ytr_side_finish(ggml_backend_cuda_context & ctx, const ggml_tensor * node, float * out, int64_t out_col);

// GET_ROWS of a YTR weight (the token embedding): decoded rows as F32
bool ggml_cuda_get_rows_ytr(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// called for every node before it is computed: true if the node's result came from an earlier YTR launch and was copied out here
// (the TOP_K and GET_ROWS of a head product served by the shortlist head, GGML_YTR_SHEAD)
bool ggml_cuda_ytr_node_hook(ggml_backend_cuda_context & ctx, ggml_tensor * node);
