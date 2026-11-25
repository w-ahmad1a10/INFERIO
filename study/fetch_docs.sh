#!/usr/bin/env bash
# Fetches the INFERIO study corpus (papers, docs, reference repos).
# Re-runnable: overwrites existing files. Results logged to study/raw/fetch_log.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p study/raw/pdf study/raw/html study/raw/json study/vendor
LOG=study/raw/fetch_log.txt
: > "$LOG"

fetch() { # name url outfile
  local name="$1" url="$2" out="$3"
  local code
  code=$(curl -sSL --retry 2 --connect-timeout 20 -m 240 \
        -A "Mozilla/5.0 (X11; Linux x86_64; INFERIO doc-fetch)" \
        -w "%{http_code}" -o "$out" "$url" 2>>"$LOG") || code="ERR"
  local size=0; [ -f "$out" ] && size=$(wc -c < "$out")
  printf "%-34s %-4s %10s bytes  %s\n" "$name" "$code" "$size" "$url" | tee -a "$LOG"
}

echo "=== papers / datasheets (pdf) ===" | tee -a "$LOG"
fetch cuda_programming_guide.pdf   "https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf"      study/raw/pdf/cuda_programming_guide.pdf
fetch cuda_best_practices.pdf      "https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf"     study/raw/pdf/cuda_best_practices.pdf
fetch t4_datasheet.pdf             "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf" study/raw/pdf/t4_datasheet.pdf
fetch turing_whitepaper.pdf        "https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf" study/raw/pdf/turing_whitepaper.pdf
fetch flash_attention_v1.pdf       "https://arxiv.org/pdf/2205.14135"                                      study/raw/pdf/flash_attention_v1.pdf
fetch flash_attention_v2.pdf       "https://arxiv.org/pdf/2307.08691"                                      study/raw/pdf/flash_attention_v2.pdf
fetch self_attention_linear_mem.pdf "https://arxiv.org/pdf/2112.05682"                                     study/raw/pdf/self_attention_linear_mem.pdf
fetch online_softmax.pdf           "https://arxiv.org/pdf/1805.02867"                                      study/raw/pdf/online_softmax.pdf
fetch position_interpolation.pdf   "https://arxiv.org/pdf/2306.15595"                                      study/raw/pdf/position_interpolation.pdf
fetch gpt2_paper.pdf               "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" study/raw/pdf/gpt2_paper.pdf

sleep 2
echo "=== guides / blogs (html) ===" | tee -a "$LOG"
fetch turing_tuning_guide.html     "https://docs.nvidia.com/cuda/turing-tuning-guide/index.html"           study/raw/html/turing_tuning_guide.html
fetch cublas_docs.html             "https://docs.nvidia.com/cuda/cublas/index.html"                        study/raw/html/cublas_docs.html
fetch ptx_isa.html                 "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"     study/raw/html/ptx_isa.html
fetch triton_index.html            "https://triton-lang.org/main/index.html"                               study/raw/html/triton_index.html
fetch triton_matmul_tutorial.html  "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html" study/raw/html/triton_matmul_tutorial.html
fetch triton_fused_attention.html  "https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html" study/raw/html/triton_fused_attention.html
fetch siboehm_cuda_matmul.html     "https://siboehm.com/articles/22/CUDA-MMM"                              study/raw/html/siboehm_cuda_matmul.html
fetch horace_brrr.html             "https://horace.io/brrr_intro.html"                                     study/raw/html/horace_brrr.html

echo "=== model metadata (json/md) ===" | tee -a "$LOG"
fetch gpt2_config.json             "https://huggingface.co/openai-community/gpt2/raw/main/config.json"     study/raw/json/gpt2_config.json
fetch gpt2_model_card.md           "https://huggingface.co/openai-community/gpt2/raw/main/README.md"       study/raw/json/gpt2_model_card.md

echo "=== reference repos (shallow, sparse) ===" | tee -a "$LOG"
if [ ! -d study/vendor/llm.c ]; then
  git clone -q --depth 1 https://github.com/karpathy/llm.c study/vendor/llm.c 2>>"$LOG" && echo "cloned llm.c" | tee -a "$LOG"
else echo "llm.c already present" | tee -a "$LOG"; fi
if [ ! -d study/vendor/cutlass ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/cutlass study/vendor/cutlass 2>>"$LOG" \
    && git -C study/vendor/cutlass sparse-checkout set docs media 2>>"$LOG" \
    && echo "cloned cutlass (sparse: docs media)" | tee -a "$LOG"
else echo "cutlass already present" | tee -a "$LOG"; fi
if [ ! -d study/vendor/triton ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/triton-lang/triton study/vendor/triton 2>>"$LOG" \
    && git -C study/vendor/triton sparse-checkout set docs python/tutorials python/triton/language 2>>"$LOG" \
    && echo "cloned triton (sparse: docs tutorials language)" | tee -a "$LOG"
else echo "triton already present" | tee -a "$LOG"; fi
for d in study/vendor/llm.c study/vendor/cutlass study/vendor/triton; do
  [ -d "$d/.git" ] && printf "%-28s %s\n" "$d" "$(git -C "$d" rev-parse --short HEAD)" | tee -a "$LOG"
done

echo "=== done ===" | tee -a "$LOG"
