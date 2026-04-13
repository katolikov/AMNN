"""Main benchmark pipeline orchestrator and stage overview printer."""

from __future__ import annotations
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

from .constants import ROOT, C
from .config import Config
from .device import DeviceManager
from .builder import Builder
from .converter import ModelConverter
from .stage_runner import StageRunner
from .logger import Logger

TOTAL_STEPS = 8


def print_stages_overview(stages: List[Dict[str, Any]]) -> None:
    if not stages:
        return
    backend_colors = {"cpu": C.GREEN, "opencl": C.BLUE, "vulkan": C.MAGENTA, "metal": C.CYAN, "cuda": C.YELLOW}
    names = [s["name"] for s in stages]
    col_w = max(max(len(n) for n in names), 12)

    def cell(val, w=col_w):
        return f"{val:<{w}}"

    sep_w = 18 + (col_w + 3) * len(stages) + 1
    print(f"\n  {C.BOLD}Stage Overview{C.RESET}")
    print(f"  {'─' * sep_w}")

    hdr = f"  {'Parameter':<18}"
    for s in stages:
        bc = backend_colors.get(s.get("backend", "cpu"), C.WHITE)
        hdr += f"│ {bc}{C.BOLD}{cell(s['name'])}{C.RESET} "
    print(hdr + "│")
    print(f"  {'─' * 18}" + ("┼" + "─" * (col_w + 2)) * len(stages) + "┤")

    rows = [
        ("Backend",      lambda s: s.get("backend", "cpu")),
        ("Precision",    lambda s: s.get("precision", "low")),
        ("Memory",       lambda s: s.get("memory", "low")),
        ("Power",        lambda s: s.get("power", "normal")),
        ("Threads",      lambda s: str(s.get("threads", "-")) if s.get("backend", "cpu") == "cpu" else "-"),
        ("GPU mode",     lambda s: ",".join(f.replace("MNN_GPU_", "") for f in s.get("gpu_mode", [])) or "-"),
        ("Attn mode",    lambda s: str(s.get("attention_mode", s.get("quant_qkv", 0)))),
        ("Warmup",       lambda s: str(s.get("warmup_rounds", 2))),
        ("Measure",      lambda s: str(s.get("measure_rounds", 5))),
        ("Max gen tok",  lambda s: str(s.get("max_gen_tokens", 64))),
        ("VLM input",    lambda s: "yes" if s.get("use_vlm_input") else "-"),
        ("OP profile",   lambda s: "ON" if s.get("enable_op_profile") else "-"),
        ("MLLM backend", lambda s: s.get("mllm_backend", "-") or "-"),
        ("Cache file",   lambda s: pathlib.Path(s["cache_file"]).name if s.get("cache_file") else "-"),
    ]

    for label, fn in rows:
        row = f"  {C.DIM}{label:<18}{C.RESET}"
        for s in stages:
            val = fn(s)
            row += f"│ {C.DIM + cell(val) + C.RESET if val == '-' else cell(val)} "
        print(row + "│")

    print(f"  {'─' * 18}" + ("┴" + "─" * (col_w + 2)) * len(stages) + "┘")
    print()


class BenchmarkPipeline:
    """Main orchestrator — drives the full benchmark pipeline."""

    def __init__(self, config_path: str, debug: bool = False):
        self.logger = Logger(debug=debug)
        self.config = Config(config_path, self.logger)
        self.device = DeviceManager(self.config, self.logger)
        self.builder = Builder(self.config, self.logger)
        self.converter = ModelConverter(self.config, self.logger, self.device)
        self.runner = StageRunner(self.config, self.device, self.logger)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self):
        print(f"\n{C.BOLD}╔══════════════════════════════════════════════════════════╗{C.RESET}")
        print(f"{C.BOLD}║          MNN LLM/VLM Benchmark Orchestrator             ║{C.RESET}")
        print(f"{C.BOLD}╚══════════════════════════════════════════════════════════╝{C.RESET}\n")

        self.logger.info(f"Config: {C.BOLD}{self.config.path.name}{C.RESET}")
        self.logger.info(f"Timestamp: {C.BOLD}{self.timestamp}{C.RESET}")
        stages = self.config.stages
        self.logger.info(f"Stages: {len(stages)} enabled")
        if self.logger.debug_mode:
            self.logger.debug(f"Config: {json.dumps(self.config.raw, indent=2)[:500]}...")

        # 1. Resolve inputs
        self.logger.step(1, TOTAL_STEPS, "Resolve Inputs")
        self.config.resolve_images()

        # 2. Model
        self.logger.step(2, TOTAL_STEPS, "Model Conversion")
        model_dir = self.converter.get_model_dir()

        # 3. Build
        self.logger.step(3, TOTAL_STEPS, "Cross-Compilation (Android NDK)")
        binary = self.builder.build()

        # 4. Deploy
        self.logger.step(4, TOTAL_STEPS, "Deploy to Device")
        if self.config.device_id:
            self.device.create_session()
            self.logger.info(f"Device: {C.BOLD}{self.config.device_id}{C.RESET}")
        self.device.push_all(binary, model_dir)

        # 5. Clocks
        self.logger.step(5, TOTAL_STEPS, "Clock Management")
        self.device.apply_clocks()

        # 6. Run stages
        self.logger.step(6, TOTAL_STEPS, f"Execute Benchmarks ({len(stages)} stages)")
        print_stages_overview(stages)
        self.device.clear_logcat()

        # Output base: benchmark_results/<benchmark_dir>/ if set, otherwise benchmark_results/
        bench_dir = self.config.raw.get("benchmark_dir", "")
        output_root = self.config.output_base
        if bench_dir:
            output_root = output_root / bench_dir

        t_total = time.time()
        try:
            for i, stage in enumerate(stages, 1):
                # Thermal cooldown before each stage
                self.device.wait_for_cooldown(stage_name=stage["name"])

                stage_output = output_root / stage["name"] / self.timestamp
                stage_output.mkdir(parents=True, exist_ok=True)
                self.config.save_to(stage_output)
                self.runner.run(stage, stage_output, idx=i, total=len(stages))
        finally:
            self.device.restore_clocks()
        total_elapsed = time.time() - t_total

        # 7. Collect
        self.logger.step(7, TOTAL_STEPS, "Collect Results")
        combined_dir = output_root / "_combined" / self.timestamp
        combined_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_to(combined_dir)
        self.device.capture_logcat(combined_dir)
        self.device.pull_artifacts(combined_dir)

        # 8. Reports
        reports = self.config.reports
        if reports.get("markdown") or reports.get("excel"):
            self.logger.step(8, TOTAL_STEPS, "Generate Reports")
            self._generate_reports(combined_dir)
        else:
            self.logger.step(8, TOTAL_STEPS, "Generate Reports")
            self.logger.info(f'{C.DIM}No reports configured (add "reports": {{"markdown": true, "excel": true}}){C.RESET}')

        # Summary
        print(f"\n{C.BOLD}{C.GREEN}{'═' * 60}{C.RESET}")
        print(f"  {C.GREEN}{C.BOLD}{C.CHECK} All done!{C.RESET}  Total time: {C.BOLD}{total_elapsed:.0f}s{C.RESET}")
        for stage in stages:
            sd = output_root / stage["name"] / self.timestamp
            print(f"  {C.DIM}{stage['name']}:{C.RESET} {C.CYAN}{sd}{C.RESET}")
        print(f"{C.BOLD}{C.GREEN}{'═' * 60}{C.RESET}\n")

    def _generate_reports(self, output_dir: pathlib.Path):
        try:
            report_script = ROOT / "generate_report.py"
            if report_script.exists():
                result = subprocess.run(
                    [sys.executable, str(report_script), str(output_dir)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line.startswith("Saved:"):
                            self.logger.ok(line)
                else:
                    self.logger.warn(f"Report generation failed: {result.stderr[:200]}")
            else:
                self.logger.warn("generate_report.py not found")
        except Exception as e:
            self.logger.warn(f"Report generation error: {e}")
