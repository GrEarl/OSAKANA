#include "common.cuh"

// MXFP4 weights x FP8 (e4m3) activations on the block-scaled tensor-core instruction of Blackwell.
// One pass reads the weights once for up to 64 activation rows. Enable with GGML_CUDA_FP4A8=1.

bool ggml_cuda_fp4a8_enabled(int device);

// repack the MXFP4 weights of the graph's MUL_MAT nodes into the tile layout of the kernel (in place, once per tensor)
void ggml_cuda_fp4a8_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);

// true when the data of this weight tensor is in the tile layout: only ggml_cuda_mul_mat_fp4a8 can read it
bool ggml_cuda_fp4a8_ready(const ggml_tensor * src0);

void ggml_cuda_mul_mat_fp4a8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// two MXV matrices of one shape times the same input in one launch; false = not that case, nothing was done
bool ggml_cuda_mul_mat_fp4a8_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0a, const ggml_tensor * src0b, const ggml_tensor * src1,
                                  ggml_tensor * dsta, ggml_tensor * dstb);
