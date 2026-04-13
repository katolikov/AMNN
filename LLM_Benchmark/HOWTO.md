# MNN LLM/VLM Benchmark Suite — How-To Guide

## Overview

The LLM Benchmark suite measures inference performance of MNN LLM and VLM models on Android devices. It supports CPU, OpenCL, and Vulkan backends with configurable precision, memory modes, and vision encoder settings.

**Key capabilities:**
- Multi-stage benchmark comparison (CPU vs GPU, different thread counts, etc.)
- VLM (Vision-Language Model) benchmarking with real images
- Automated cross-compilation, deployment, and execution via ADB
- Auto-generated Excel and Confluence-ready Markdown reports
- OpenCL kernel-level GPU profiling (when enabled)

## Quick Start

```bash
cd LLM_Benchmark/

# 1. Create a config JSON (see examples below)
# 2. Run the benchmark
python3 run_benchmark.py my_config.json

# Results saved to benchmark_results/<timestamp>/
# - confluence_report.md   (if reports.markdown=true)
# - full_report.xlsx       (if reports.excel=true)
```

## Config JSON Structure

```json
{
    "build": {
        "ndk_path": "/path/to/android-ndk",
        "cmake_path": "cmake",
        "build_type": "Release",
        "jobs": 8,
        "cmake_flags": [
            "-DANDROID_ABI=arm64-v8a",
            "-DANDROID_STL=c++_static",
            "-DANDROID_NATIVE_API_LEVEL=android-28",
            "-DMNN_ARM82=ON",
            "-DMNN_OPENCL=ON"
        ]
    },
    "model": {
        "converted_mnn_dir": "/path/to/Model-MNN",
        "model_type": "smolvlm"
    },
    "inputs": {
        "images": ["/path/to/image.png"],
        "vlm_prompt_template": "Describe the photo: <img>img0</img>"
    },
    "device": {
        "adb_device_id": "DEVICE_SERIAL",
        "target_dir": "/data/local/tmp/mnn_bench"
    },
    "reports": {
        "markdown": true,
        "excel": true
    },
    "stages": [ ... ]
}
```

## Stage Configuration Reference

Each stage defines one benchmark run. Multiple stages run sequentially for comparison.

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Stage identifier (used in output dirs and reports) |
| `enabled` | bool | true | Skip stage if false |
| `backend` | string | "cpu" | Backend: `cpu`, `opencl`, `vulkan`, `metal`, `cuda` |
| `precision` | string | "low" | Precision: `low` (fp16), `normal` (fp32), `high` |
| `memory` | string | "low" | Memory mode: `low`, `high` |
| `power` | string | "normal" | Power mode: `low`, `normal`, `high` |
| `attention_mode` | int | 0 | Attention/KV optimization (replaces deprecated `quant_qkv`) |

### CPU-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threads` | int | 4 | CPU thread count |

### GPU-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_mode` | string[] | [] | GPU flags: `MNN_GPU_MEMORY_BUFFER`, `MNN_GPU_TUNING_WIDE`, `MNN_GPU_RECORD_BATCH` |
| `no_tuning` | bool | false | Skip kernel tuning (use cached kernels) |

### Benchmark Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_rounds` | int | 2 | Warmup iterations (not measured) |
| `measure_rounds` | int | 5 | Measurement iterations |
| `prompt_tokens` | int | 128 | Synthetic prompt length (text-only mode, 0 = use VLM) |
| `max_gen_tokens` | int | 64 | Maximum tokens to generate per round |

### VLM (Vision) Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_vlm_input` | bool | false | Enable VLM mode with image input |
| `vlm_image_index` | int | null | Which image from `inputs.images` (null = all) |
| `mllm_backend` | string | "" | Vision encoder backend: `cpu` or `opencl` |
| `mllm_precision` | string | "low" | Vision encoder precision |
| `mllm_threads` | int | 4 | Vision encoder CPU threads (for CPU backend) |

### Advanced Engine Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mmap` | bool | false | Memory-mapped weight loading |
| `max_all_tokens` | int | 0 | Total sequence length limit (0 = engine default 2048) |
| `dynamic_quant` | int | 0 | Dynamic quantization option |
| `kvcache_limit` | int | -1 | KV cache size limit (-1 = unlimited) |
| `enable_op_profile` | bool | false | Enable per-operator profiling |

### Sampler Options

```json
"sampler": {
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_window": 0
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Controls randomness. Applied as `logits /= temperature` before sampling. **0.0** = greedy (deterministic), **< 1.0** = sharper (more confident), **> 1.0** = flatter (more random). Use 0.0 for reproducible benchmarks |
| `repetition_penalty` | 1.0 | Penalizes repeated tokens. **> 1.0** reduces repetition (e.g., 1.1). Applied multiplicatively to logits of previously seen tokens |
| `presence_penalty` | 0.0 | Constant penalty subtracted from logits of any previously seen token. Unlike repetition_penalty, does not scale with frequency |
| `frequency_penalty` | 0.0 | Penalty proportional to token frequency. `logit -= frequency_penalty * count(token)` |
| `penalty_window` | 0 | How many recent tokens to consider for penalties. 0 = full history |

### Convenience Hint Shortcuts

These map to MNN runtime hints internally:

| Parameter | Type | Hint | Description |
|-----------|------|------|-------------|
| `prefer_decode` | bool | DYNAMIC_QUANT_OPTIONS=1 | Optimize for decode throughput |
| `encoder_commit_batch` | int | OP_ENCODER_NUMBER_FOR_COMMIT | OpenCL command batch size |
| `cpu_littlecore_rate` | int | CPU_LITTLECORE_DECREASE_RATE | Little core scheduling rate |
| `sme2_instructions` | int | CPU_SME2_INSTRUCTIONS | Enable SME2 instructions |

Raw hints can still be passed via the `hints` dict for less common options.

## Example Configs

### CPU vs OpenCL VLM Comparison

```json
{
    "stages": [
        {
            "name": "cpu",
            "backend": "cpu",
            "threads": 4,
            "precision": "normal",
            "warmup_rounds": 1,
            "measure_rounds": 3,
            "max_gen_tokens": 200,
            "use_vlm_input": true,
            "mllm_backend": "cpu",
            "mllm_threads": 4
        },
        {
            "name": "opencl",
            "backend": "opencl",
            "gpu_mode": ["MNN_GPU_MEMORY_BUFFER", "MNN_GPU_TUNING_WIDE"],
            "precision": "low",
            "power": "high",
            "warmup_rounds": 1,
            "measure_rounds": 3,
            "max_gen_tokens": 200,
            "use_vlm_input": true,
            "mllm_backend": "cpu",
            "mllm_threads": 4
        }
    ]
}
```

### OpenCL Kernel Profiling

Requires `-DMNN_GPU_TIME_PROFILE=ON` in `cmake_flags`:

```json
{
    "build": {
        "cmake_flags": ["-DMNN_OPENCL=ON"]
    },
    "stages": [
        {
            "name": "opencl_profile",
            "backend": "opencl",
            "gpu_mode": ["MNN_GPU_MEMORY_BUFFER", "MNN_GPU_TUNING_WIDE"],
            "precision": "low",
            "warmup_rounds": 0,
            "measure_rounds": 1,
            "max_gen_tokens": 32,
            "enable_op_profile": true,
            "use_vlm_input": true
        }
    ]
}
```

`MNN_GPU_TIME_PROFILE` is automatically enabled when any stage has `enable_op_profile: true`.
Kernel timing data is written to logcat and parsed into the Excel report automatically.

## Output Structure

Results are organized by stage name for easy comparison across runs:

```
benchmark_results/
├── <stage_name>/                      # Per-stage results
│   └── <timestamp>/                   # Run timestamp
│       ├── benchmark_config.json      # Config snapshot
│       └── <model_name>/
│           └── output_stdout.txt      # Benchmark output
├── _combined/                         # Shared artifacts per run
│   └── <timestamp>/
│       ├── benchmark_config.json
│       ├── confluence_report.md       # Markdown report (if reports.markdown=true)
│       ├── full_report.xlsx           # Excel report (if reports.excel=true)
│       ├── logcat_capture.txt         # Full logcat
│       ├── logcat_mnnjni.txt          # MNN-only logcat
└── logcat_mnnjni.txt              # MNN-only logcat (kernel profiling data)
```

## Timing Metrics Explained

| Metric | Description |
|--------|-------------|
| **TTFT** | Time To First Token = total wall-clock minus decode time. Includes vision encoding + embedding + LLM prefill |
| **Vision encode** | Vision encoder (ViT/CNN) forward pass time only |
| **LLM forward** | LLM transformer prefill forward pass (excludes embedding lookup) |
| **Embed+overhead** | Vision embedding lookup, concatenation, GC, and other overhead |
| **Decode** | Total time for all generated tokens (sample + forward per token) |
| **TPOT** | Time Per Output Token = decode_total / num_tokens |
| **Prefill tok/s** | prompt_tokens / (LLM_forward_time_seconds) |
| **Decode tok/s** | generated_tokens / (decode_time_seconds) |

## Tips

- **Best OpenCL config for Samsung (Xclipse/Mali)**: `gpu_mode: ["MNN_GPU_MEMORY_BUFFER", "MNN_GPU_TUNING_WIDE"]`, `precision: "low"`, `power: "high"`
- **Vision encoder**: Keep on CPU (`mllm_backend: "cpu"`, `mllm_threads: 4`). OpenCL MLLM is slower due to GPU contention
- **CPU threads for MLLM**: 4 is optimal. 8 threads causes memory bandwidth contention with GPU decode
- **Warmup matters**: Use at least 1 warmup round. First OpenCL run compiles kernels and is 2-10x slower
- **Thermal throttling**: Run stages from coldest to hottest (CPU first, then GPU). Wait between heavy stages
- **`attention_mode`**: Replaces deprecated `quant_qkv`. Set to 0 for no KV quantization
- **`MNN_GPU_RECORD_BATCH`**: Not beneficial on Samsung Xclipse 950 — actually slower. Test on your device
- **Kernel profiling overhead**: `MNN_GPU_TIME_PROFILE=ON` adds ~20% overhead. Don't use for absolute timing

## Standalone Report Generation

Reports can be generated independently from existing results:

```bash
# Generate from a previous run
python3 generate_report.py benchmark_results/20260413_224936/

# Standalone kernel profiling Excel
python3 parse_opencl_profile.py benchmark_results/20260413_212811/logcat_mnnjni.txt output.xlsx
```
