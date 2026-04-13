"""Stage execution — builds commands and runs benchmark on device."""

from __future__ import annotations
import pathlib
import shlex
import time
from typing import Dict, List

from .constants import C, STAGE_FLAGS
from .config import Config
from .device import DeviceManager
from .logger import Logger


class StageRunner:
    def __init__(self, config: Config, device: DeviceManager, logger: Logger):
        self.config = config
        self.device = device
        self.logger = logger

    def build_command(self, stage: dict) -> List[str]:
        target = self.config.target_dir
        cmd = [f"{target}/llm_benchmark", f"{target}/model/config.json",
               "--tmp_path", f"{target}/tmp"]

        if "quant_qkv" in stage and "attention_mode" not in stage:
            stage["attention_mode"] = stage["quant_qkv"]
            self.logger.warn("'quant_qkv' is deprecated, use 'attention_mode' instead")

        for key, flag, typ, default, cond in STAGE_FLAGS:
            if cond and not cond(stage):
                continue
            val = stage.get(key, default)
            if typ == bool:
                if val:
                    cmd.append(flag)
            elif val != default:
                cmd += [flag, str(val)]
            elif key in ("backend", "precision", "memory", "power"):
                cmd += [flag, str(val)]

        backend = stage.get("backend", "cpu")
        if backend in ("opencl", "vulkan", "metal", "cuda"):
            gpu_mode = stage.get("gpu_mode", [])
            if gpu_mode:
                cmd += ["--gpu_mode", ",".join(str(f) for f in gpu_mode)]

        hints = dict(stage.get("hints", {}))
        if stage.get("prefer_decode", False):
            hints.setdefault("DYNAMIC_QUANT_OPTIONS", 1)
        if "encoder_commit_batch" in stage:
            hints.setdefault("OP_ENCODER_NUMBER_FOR_COMMIT", stage["encoder_commit_batch"])
        if "cpu_littlecore_rate" in stage:
            hints.setdefault("CPU_LITTLECORE_DECREASE_RATE", stage["cpu_littlecore_rate"])
        if "sme2_instructions" in stage:
            hints.setdefault("CPU_SME2_INSTRUCTIONS", stage["sme2_instructions"])
        if hints:
            cmd += ["--hints", ",".join(f"{k}:{v}" for k, v in hints.items())]

        mllm_backend = stage.get("mllm_backend", "")
        if mllm_backend:
            cmd += ["--mllm_backend", mllm_backend,
                    "--mllm_precision", stage.get("mllm_precision", "low"),
                    "--mllm_threads", str(stage.get("mllm_threads", 4))]

        sampler = stage.get("sampler", {})
        for skey, sflag, sdefault in [
            ("temperature", "--temperature", 1.0),
            ("repetition_penalty", "--repetition_penalty", 1.0),
            ("presence_penalty", "--presence_penalty", 0.0),
            ("frequency_penalty", "--frequency_penalty", 0.0),
            ("penalty_window", "--penalty_window", 0),
        ]:
            val = sampler.get(skey, sdefault)
            if val != sdefault:
                cmd += [sflag, str(val)]

        self.logger.debug(f"Command: {' '.join(cmd)}")
        return cmd

    def _format_header(self, stage: dict, idx: int, total: int) -> str:
        colors = {"cpu": C.GREEN, "opencl": C.BLUE, "vulkan": C.MAGENTA, "metal": C.CYAN, "cuda": C.YELLOW}
        bc = colors.get(stage.get("backend", "cpu"), C.WHITE)
        parts = [stage.get("backend", "cpu").upper()]
        if stage.get("use_vlm_input"):
            parts.append("VLM")
        if stage.get("gpu_mode"):
            parts.append(",".join(f.replace("MNN_GPU_", "") for f in stage["gpu_mode"]))
        parts.append(f"prec={stage.get('precision', 'low')}")
        return (f"\n{'━' * 60}\n"
                f"  {C.BOLD}Stage [{idx}/{total}]{C.RESET}  {bc}{C.BOLD}{stage['name']}{C.RESET}\n"
                f"  {C.DIM}{' | '.join(parts)}{C.RESET}\n"
                f"{'━' * 60}")

    def run(self, stage: dict, output_dir: pathlib.Path, idx: int, total: int) -> None:
        target = self.config.target_dir
        model_name = self.config.model_name()
        print(self._format_header(stage, idx, total))

        stage_dir = output_dir / model_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        cache_file = stage.get("cache_file", "")
        if cache_file:
            self.device.push_cache(cache_file)

        base_cmd = self.build_command(stage)

        vlm_images: List[str] = []
        if stage.get("use_vlm_input", False):
            all_images = self.config.inputs.get("images", [])
            vlm_idx = stage.get("vlm_image_index", None)
            if vlm_idx is not None and vlm_idx < len(all_images):
                vlm_images.append(f"{target}/images/{pathlib.Path(all_images[vlm_idx]).name}")
            elif all_images:
                vlm_images = [f"{target}/images/{pathlib.Path(p).name}" for p in all_images]

        if vlm_images:
            tpl = self.config.inputs.get("vlm_prompt_template", "")
            all_output = ""
            for img_idx, dev_img_path in enumerate(vlm_images):
                img_label = pathlib.Path(dev_img_path).stem
                self.logger.info(f"  {C.BLUE}Image [{img_idx+1}/{len(vlm_images)}]{C.RESET}: {img_label}")
                img_cmd = list(base_cmd) + ["--vlm_image", dev_img_path]
                if tpl:
                    img_cmd += ["--vlm_template", tpl]
                shell_cmd = (f"export LD_LIBRARY_PATH={target}:$LD_LIBRARY_PATH && "
                           + " ".join(shlex.quote(c) for c in img_cmd))
                self.logger.debug(f"Shell: {shell_cmd}")
                t0 = time.time()
                ret = self.device.shell(shell_cmd, capture=True, silent=True)
                elapsed = time.time() - t0
                print(ret.stdout, end="")
                all_output += ret.stdout
                if ret.stderr and ret.stderr.strip():
                    (stage_dir / f"{img_label}_stderr.txt").write_text(ret.stderr)
                self.logger.ok(f"{img_label} completed in {elapsed:.1f}s")
            (stage_dir / "output_stdout.txt").write_text(all_output)
        else:
            prompt_text = ""
            prompt_idx = stage.get("prompt_index", None)
            if prompt_idx is not None:
                prompts = self.config.inputs.get("prompts", [])
                if prompt_idx < len(prompts):
                    prompt_text = prompts[prompt_idx]
            elif not stage.get("prompt_tokens", 0):
                prompts = self.config.inputs.get("prompts", [])
                if prompts:
                    prompt_text = prompts[0]

            cmd = list(base_cmd)
            if prompt_text:
                cmd += ["--prompt_text", prompt_text]
            shell_cmd = (f"export LD_LIBRARY_PATH={target}:$LD_LIBRARY_PATH && "
                       + " ".join(shlex.quote(c) for c in cmd))
            self.logger.debug(f"Shell: {shell_cmd}")
            t0 = time.time()
            ret = self.device.shell(shell_cmd, capture=True, silent=True)
            elapsed = time.time() - t0
            print(ret.stdout, end="")
            (stage_dir / "output_stdout.txt").write_text(ret.stdout)
            if ret.stderr and ret.stderr.strip():
                (stage_dir / "output_stderr.txt").write_text(ret.stderr)
            self.logger.ok(f"Completed in {elapsed:.1f}s")

        if stage.get("update_cache", False) and cache_file:
            self.device.pull_cache(cache_file)
            self.logger.ok(f"Cache updated: {cache_file}")
