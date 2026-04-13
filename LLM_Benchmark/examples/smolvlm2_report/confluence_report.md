# SmolVLM2-MNN Benchmark Report

| | |
|---|---|
| **Model** | SmolVLM2-MNN |
| **Device** | R5CY71BJJ9D |
| **Image** | image.png |
| **Prompt** | `What are the labels GT and PRED? Describe the photo: <img>img0</img>` |

## Performance Comparison

| Metric | Unit | cpu_best | opencl_best | opencl_kernel_ops | Winner |
|--------|------|------|------|------|--------|
| LLM Backend |  | cpu | opencl | opencl |  |
| MLLM (Vision) |  | cpu (t=4, normal) | cpu (t=4, normal) | cpu (t=4, normal) |  |
| Precision |  | normal | low | low |  |
| Model Load | ms | 2,954.3 | 2,645.6 | **1,992.8** | opencl_kernel_ops |
| Vision Encode | ms | 2,150.9 | **2,136.5** | 2,277.7 | opencl_best |
| LLM Forward | ms | 847.0 | **543.6** | 586.8 | opencl_best |
| Embed+Overhead | ms | **162.8** | 640.0 | 406.3 | cpu_best |
| TTFT | ms | **3,160.7** | 3,320.1 | 3,270.8 | cpu_best |
| Decode Total | ms | 9,030.3 | 10,206.8 | **8,818.7** | opencl_kernel_ops |
| Decode Throughput | tok/s | **21.7** | 19.6 | 14.2 | cpu_best |
| TPOT | ms | **46.2** | 51.0 | 70.5 | cpu_best |
| Prefill Throughput | tok/s | 116.3 | **180.4** | 167.0 | opencl_best |
| End-to-End | ms | 12,191.0 | 13,527.0 | **12,089.5** | opencl_kernel_ops |
| Output Status |  | NORMAL_FINISHED | NORMAL_FINISHED | NORMAL_FINISHED |  |
| Output Tokens |  | 215.0 | 219.0 | 187.0 |  |

> **Best decode throughput:** cpu_best at 21.7 tok/s

## Model Outputs

### cpu_best
*Status: NORMAL_FINISHED, Tokens: 215*

```
Assistant: The labels GT and PRED are not visible in the provided image. However, the image appears to be a photograph of a person standing on a snow-covered mountain, holding a snowboard. The person is wearing a helmet and a jacket, and the snowboard has a blue and white design. The background shows a snow-covered mountain with other people in the distance. The sky is blue, indicating it might be a sunny day. The image also has some text at the bottom, which is partially visible and includes th
... (truncated)
```

### opencl_best
*Status: NORMAL_FINISHED, Tokens: 219*

```
Assistant: The image is a digital composite featuring a person standing on a snow-covered slope, holding a snowboard. The individual is dressed in winter sports attire, including a helmet and goggles, suggesting that the photo was taken in a cold environment, possibly during a snowboarding activity. The background showcases a mountainous landscape with ski lifts and other people engaged in winter sports, indicating a popular ski resort. The sky is clear and blue, suggesting favorable weather con
... (truncated)
```

### opencl_kernel_ops
*Status: NORMAL_FINISHED, Tokens: 187*

```
Assistant: The photo shows a person holding a snowboard in a snowy landscape, likely at a ski resort. The individual appears to be preparing for a snowboarding activity. The snowboard has a design with a blue and white color scheme. There are other people in the background, suggesting that this might be a popular spot for snow sports. The sky is partly cloudy, indicating a cold but clear day. The person's attire, including a helmet and goggles, indicates that the weather is likely cold and snowy
... (truncated)
```

## Key Findings

1. **Best decode:** cpu_best (21.7 tok/s) -- 1.5x faster than opencl_kernel_ops (14.2 tok/s)
2. **Lowest TPOT:** cpu_best (46.2 ms/token)

## GPU Kernel Analysis

### Wall-Clock vs GPU Kernel Time

| Metric | Wall-Clock | GPU Kernel | Overhead |
|--------|-----------|-----------|----------|
| **TPOT (ms)** | 51.0 | 56.3 | -10% (-5.3 ms) |
| **Decode (tok/s)** | 19.6 | 17.8 | |
| **TTFT (ms)** | 3320.1 | 618.5 | 81% (2701.6 ms) |

*GPU kernel time is measured via Session_Debug profiling. The overhead represents CPU-side scheduling, memory transfers, and queue management.*

### Phase Summary

| Phase | GPU Time (ms) | Steps | Avg/Step (ms) |
|-------|-------------|-------|--------------|
| **Prefill** | 1,236.9 | 2 | 618.5 |
| **Decode** | 25,268.3 | 449 | 56.3 |
| **Total** | 26,505.2 | 451 | |

### Decode Kernels (Top 10)

| Kernel | Total (ms) | Count | Avg (us) | Share |
|--------|-----------|-------|---------|-------|
| Convolution0 | 22,261.3 | 75,872 | 293 | 88.1% |
| Raster0 | 923.4 | 175,083 | 5 | 3.7% |
| BinaryOp0 | 526.3 | 54,321 | 10 | 2.1% |
| matmul_qkv | 390.6 | 10,775 | 36 | 1.5% |
| matmul_qk_div_mask | 386.5 | 10,775 | 36 | 1.5% |
| While0 | 198.8 | 43,095 | 5 | 0.8% |
| softmax | 181.1 | 10,775 | 17 | 0.7% |
| LayerNorm0 | 175.4 | 21,998 | 8 | 0.7% |
| UnaryOp0 | 108.3 | 33,219 | 3 | 0.4% |
| Raster1 | 80.0 | 21,548 | 4 | 0.3% |

### Prefill Kernels (Top 10)

| Kernel | Total (ms) | Count | Avg (us) | Share |
|--------|-----------|-------|---------|-------|
| Convolution0 | 1,118.6 | 338 | 3,310 | 90.4% |
| Raster0 | 53.7 | 784 | 69 | 4.3% |
| BinaryOp0 | 35.7 | 242 | 148 | 2.9% |
| While0 | 6.3 | 192 | 33 | 0.5% |
| UnaryOp0 | 5.0 | 148 | 34 | 0.4% |
| softmax | 4.6 | 48 | 97 | 0.4% |
| matmul_qk_div_mask | 4.4 | 48 | 93 | 0.4% |
| matmul_qkv | 3.1 | 48 | 65 | 0.3% |
| rearrange_k | 1.6 | 48 | 32 | 0.1% |
| Raster1 | 1.5 | 96 | 16 | 0.1% |

> **Note on profiling artifacts:** The following kernels had individual call times exceeding 10x the module's total reported time, indicating Session_Debug GPU sync overhead rather than real GPU execution time. These entries have been excluded from the totals above:
>
> - **While0**: 1 occurrences flagged, total 1929.11M us (example: single call 1,929,109,034 us vs module total 1,929,146,426 us)

## Stage Configurations

| Parameter | cpu_best | opencl_best |
|-----------|---|---|
| name | cpu_best | opencl_best |
| enabled | True | True |
| backend | cpu | opencl |
| precision | normal | low |
| memory | low | low |
| power | normal | high |
| threads | 4 | - |
| gpu_mode | - | ["MNN_GPU_MEMORY_BUFFER", "MNN_GPU_TUNING_WIDE"] |
| mllm_backend | cpu | cpu |
| mllm_threads | 4 | 4 |
| warmup_rounds | 3 | 3 |
| measure_rounds | 5 | 5 |
| max_gen_tokens | 200 | 200 |
| enable_op_profile | False | False |
| no_tuning | True | False |
| mllm_precision | normal | normal |
| prompt_tokens | 0 | 0 |
| use_vlm_input | True | True |
| vlm_image_index | 0 | 0 |

---

*Profiling methodology: GPU kernel times are collected via MNN Session_Debug mode which inserts clFinish() barriers after each kernel for accurate per-kernel timing. This adds overhead that inflates wall-clock time during profiling runs (opencl_kernel_ops). Wall-clock metrics (TPOT, TTFT, tok/s) are taken from non-profiling runs (opencl_best). Kernels whose individual call times exceed 10x their module total are flagged as profiling artifacts and excluded from analysis.*
