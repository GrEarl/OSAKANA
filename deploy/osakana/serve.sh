#!/usr/bin/env bash
# Serve Qwen3.8-27B with the OSAKANA model files (3-bit trellis weights + QSA indexers) at 262K context. Stop with Ctrl-C.
#   OSAKANA_MODEL_DIR   directory holding the model files (default: ./models)
#   OSAKANA_CTX         context size in tokens (default: 262144; about 26.8 GiB of VRAM with the defaults below)
#   OSAKANA_KV          KV cache type: q8_0 (default) or f16. f16 fits up to about 229K tokens on a 32 GB card
#   OSAKANA_DRAFTS      rows of the draft tree (default: 15)
#   OSAKANA_TREE        1 (default) tree drafts, 0 chain drafts of 7
#   OSAKANA_HOST, OSAKANA_PORT   default 127.0.0.1 and 8091
#   Choose the GPU with CUDA_VISIBLE_DEVICES. Extra arguments go to llama-server.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=${OSAKANA_BIN:-$ROOT/build/bin}
DIR=${OSAKANA_MODEL_DIR:-./models}
KV=${OSAKANA_KV:-q8_0}

# QSA: pooled block cache, raw indexer keys kept for the newest 8192 tokens only
export LLAMA_QSA_RESERVE_BLK=1 LLAMA_QSA_BLK_REBUILD=1 LLAMA_QSA27B_MASKED_PREFILL_MAX=8192 LLAMA_QSA_RAW_RING=8192 GGML_CUDA_TOPK_DET=1
export LLAMA_REUSE_OUT_ORDER=1 LLAMA_KV_PAD_GEO=2 LLAMA_CKPT_TAIL_UBATCH=1
# CUDA graphs kept per batch shape (speculative windows come in many row counts)
export GGML_FT_GRAPH_CACHE=1 GGML_FT_GRAPH_CACHE_MIN_NODES=32 GGML_FT_GRAPH_CACHE_MAX=16 CUDA_SCALE_LAUNCH_QUEUES=4x
# drafting: device argmax and top-K, wide copy windows, host rows reserved for them, shortlist head for the drafter
export LLAMA_SPEC_PREFILL_TAIL=2048 LLAMA_GPU_ARGMAX=1 LLAMA_SAMPLE_TOPK=1 LLAMA_SPEC_WIDE=1 LLAMA_OUTPUT_RESERVE=64
export GGML_YTR_SHEAD=${OSAKANA_SHEAD:-$DIR/osakana-shead.ysh}
unset GGML_CUDA_GRAPH_OPT

if [ "${OSAKANA_TREE:-1}" = 1 ]; then
  export LLAMA_SPEC_TREE=1 LLAMA_DFLASH_TREE=${OSAKANA_DRAFTS:-15}
  drafts=${OSAKANA_DRAFTS:-15}
else
  unset LLAMA_SPEC_TREE LLAMA_DFLASH_TREE
  drafts=${OSAKANA_DRAFTS:-7}
fi

exec "$BIN/llama-server" -m "$DIR/Qwen3.8-27B-OSAKANA-speed-QSA.gguf" --alias qwen3.8-27b-osakana \
  -ngl 99 -fa on -np 1 --fit off -t 8 -c "${OSAKANA_CTX:-262144}" -b 2048 -ub 256 \
  -ctk "$KV" -ctv "$KV" -ctkd f16 -ctvd f16 --no-context-shift --jinja --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 1024 \
  -md "$DIR/Qwen3.8-27B-DFlash2-OSAKANA-IQ4_XS.gguf" -ngld 99 --spec-type copy,draft-dflash --spec-copy-n-max 63 \
  --spec-draft-p-min 0 --spec-draft-n-max "$drafts" \
  --host "${OSAKANA_HOST:-127.0.0.1}" --port "${OSAKANA_PORT:-8091}" "$@"
