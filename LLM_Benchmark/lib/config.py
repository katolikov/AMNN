"""Benchmark configuration loader and validator."""

from __future__ import annotations
import json
import os
import pathlib
from typing import Any, Dict, List

from .constants import IMAGE_EXTS
from .logger import Logger


class Config:
    def __init__(self, path: str, logger: Logger):
        self.path = pathlib.Path(path).resolve()
        self.logger = logger
        if not self.path.exists():
            logger.fatal(f"Config not found: {self.path}")
        with open(self.path) as f:
            self.raw: Dict[str, Any] = json.load(f)
        self._migrate_deprecated()
        logger.debug(f"Config loaded: {self.path}")
        logger.debug(f"Stages: {[s['name'] for s in self.stages]}")

    def _migrate_deprecated(self):
        for stage in self.raw.get("stages", []):
            if "quant_qkv" in stage and "attention_mode" not in stage:
                stage["attention_mode"] = stage.pop("quant_qkv")
                self.logger.warn(f"Stage '{stage.get('name', '?')}': 'quant_qkv' deprecated, migrated to 'attention_mode'")

    @property
    def build(self) -> Dict[str, Any]: return self.raw.get("build", {})
    @property
    def model(self) -> Dict[str, Any]: return self.raw.get("model", {})
    @property
    def inputs(self) -> Dict[str, Any]: return self.raw.setdefault("inputs", {})
    @property
    def device(self) -> Dict[str, Any]: return self.raw.get("device", {})
    @property
    def outputs(self) -> Dict[str, Any]: return self.raw.get("outputs", {})
    @property
    def reports(self) -> Dict[str, Any]: return self.raw.get("reports", {})
    @property
    def stages(self) -> List[Dict[str, Any]]:
        return [s for s in self.raw.get("stages", []) if s.get("enabled", True)]

    @property
    def device_id(self) -> str: return self.device.get("adb_device_id", "")
    @property
    def target_dir(self) -> str: return self.device.get("target_dir", "/data/local/tmp/mnn_bench")
    @property
    def output_base(self) -> pathlib.Path:
        return pathlib.Path(self.outputs.get("host_output_dir", "./benchmark_results"))

    def model_name(self) -> str:
        for key in ("converted_mnn_dir", "source_path"):
            val = self.model.get(key, "").strip().rstrip("/")
            if val:
                name = pathlib.Path(val).name
                for suffix in ("-MNN", "-mnn", "-GGUF", "-gguf"):
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                return name
        return self.model.get("model_type", "unknown_model")

    def resolve_images(self):
        inputs = self.inputs
        explicit = list(inputs.get("images", []))
        img_dir = inputs.get("image_dir", "")
        if img_dir and os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if pathlib.Path(f).suffix.lower() in IMAGE_EXTS:
                    full = os.path.join(img_dir, f)
                    if full not in explicit:
                        explicit.append(full)
        inputs["images"] = explicit
        self.logger.info(f"Total images: {len(explicit)}")

    def save_to(self, output_dir: pathlib.Path):
        (output_dir / "benchmark_config.json").write_text(json.dumps(self.raw, indent=2))
