"""Model conversion — HuggingFace/GGUF to MNN format.

Supports two modes:
  1. Pre-converted: ``converted_mnn_dir`` points to existing MNN model
  2. Auto-convert:  ``source_path`` + ``conversion.enabled=true``
     Phase 1 (host):   llmexport.py does PyTorch → ONNX export
     Phase 2 (device):  MNNConvert runs ONNX → MNN on the target Android device
"""

from __future__ import annotations
import glob
import os
import pathlib
import shlex
import subprocess
import sys
from typing import List, Optional, TYPE_CHECKING

from .constants import ROOT, MNN_ROOT, C
from .config import Config
from .logger import Logger

if TYPE_CHECKING:
    from .device import DeviceManager


class ModelConverter:
    def __init__(self, config: Config, logger: Logger,
                 device: Optional["DeviceManager"] = None):
        self.config = config
        self.logger = logger
        self.device = device  # needed for on-device ONNX→MNN conversion

    def get_model_dir(self) -> pathlib.Path:
        """Detect model input type and convert if needed.

        Priority:
          1. converted_mnn_dir → use directly (no conversion)
          2. onnx_dir          → ONNX→MNN on device only
          3. source_path       → PyTorch→ONNX on host, then ONNX→MNN on device
        """
        model_cfg = self.config.model

        # Path 1: Pre-converted MNN model
        converted = model_cfg.get("converted_mnn_dir", "")
        if converted and os.path.isdir(converted):
            self.logger.ok(f"Using pre-converted MNN model: {C.CYAN}{converted}{C.RESET}")
            return pathlib.Path(converted)

        # Path 2: Pre-exported ONNX model (skip PyTorch export, only convert on device)
        onnx_dir = model_cfg.get("onnx_dir", "")
        if onnx_dir and os.path.isdir(onnx_dir):
            onnx_files = list(pathlib.Path(onnx_dir).rglob("*.onnx"))
            if onnx_files:
                self.logger.ok(f"Using ONNX model: {C.CYAN}{onnx_dir}{C.RESET} ({len(onnx_files)} files)")
                self._convert_on_device(pathlib.Path(onnx_dir))
                return pathlib.Path(onnx_dir)

        # Path 3: Source model (HuggingFace/local PyTorch) → full conversion
        conv = model_cfg.get("conversion", {})
        if not conv.get("enabled", False):
            self.logger.fatal("No converted_mnn_dir, onnx_dir, or conversion.enabled=true")
            return pathlib.Path()

        return self._convert(model_cfg, conv)

    # ── Phase 1: Host-side export (PyTorch → ONNX + weights) ──────────

    def _convert(self, model_cfg: dict, conv: dict) -> pathlib.Path:
        dst_rel = self.config.outputs.get("host_model_export_dir", "./exported_models")
        # llmexport.py runs from MNN_ROOT, so resolve dst relative to it
        dst = (MNN_ROOT / dst_rel).resolve()
        dst.mkdir(parents=True, exist_ok=True)

        export_script = MNN_ROOT / conv.get("export_script", "transformers/llm/export/llmexport.py")
        if not export_script.exists():
            self.logger.fatal(f"Export script not found: {export_script}")

        source_path = model_cfg.get("source_path", "")
        self.logger.info(f"Exporting {C.BOLD}{source_path}{C.RESET} {C.ARROW} {C.CYAN}{dst}{C.RESET}")

        # Try to find a host MNNConvert first (from build/ or build_host/)
        host_convert = self._find_host_mnnconvert()

        # If no host converter, export ONNX only and convert on device
        export_fmt = conv.get("export_format", "mnn")
        if not host_convert and export_fmt == "mnn":
            self.logger.info("No host MNNConvert found — exporting ONNX, will convert on device")
            export_fmt = "onnx"

        cmd: List[str] = [
            sys.executable, str(export_script),
            "--path",        source_path,
            "--type",        model_cfg.get("model_type", "auto"),
            "--dst_path",    dst_rel,
            "--export",      export_fmt,
            "--quant_bit",   str(conv.get("quant_bit", 4)),
            "--quant_block", str(conv.get("quant_block", 64)),
            "--embed_bit",   str(conv.get("embed_bit", 16)),
        ]
        if host_convert:
            cmd += ["--mnnconvert", str(host_convert)]
        if conv.get("visual_quant_bit"):
            cmd += ["--visual_quant_bit", str(conv["visual_quant_bit"])]
        if conv.get("sym", False):
            cmd.append("--sym")
        if conv.get("skip_visual", False):
            cmd.append("--skip_visual")
        for extra in conv.get("extra_args", []):
            cmd.append(extra)

        self.logger.debug(f"Export cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(MNN_ROOT), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            self.logger.err(f"Export failed:\n{result.stdout[-500:]}")
            raise RuntimeError("Model export failed")
        self.logger.ok(f"Host export done: {C.CYAN}{dst}{C.RESET}")

        # Phase 2: If we exported ONNX only, convert on device
        if export_fmt == "onnx" and conv.get("export_format", "mnn") == "mnn":
            self._convert_on_device(dst)

        # GGUF weight injection (if needed)
        gguf_path = model_cfg.get("gguf_path", "")
        if gguf_path:
            self._inject_gguf(gguf_path, dst)

        return dst

    # ── Phase 2: On-device ONNX → MNN conversion ─────────────────────

    def _convert_on_device(self, model_dir: pathlib.Path):
        """Push ONNX files to device, run MNNConvert, pull MNN files back."""
        if not self.device:
            self.logger.fatal("On-device conversion requires a DeviceManager (set device= in constructor)")
            return

        # Find Android MNNConvert binary
        android_convert = self._find_android_mnnconvert()
        if not android_convert:
            self.logger.fatal(
                "No Android MNNConvert found. Add -DMNN_BUILD_CONVERTER=ON to cmake_flags "
                "or build it separately with: mkdir build_converter_android && cd build_converter_android && "
                "cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake "
                "-DANDROID_ABI=arm64-v8a -DMNN_BUILD_CONVERTER=ON && make MNNConvert -j8")
            return

        target = self.config.target_dir
        conv_dir = f"{target}/converter"

        self.logger.info("Converting ONNX → MNN on device...")

        # Push MNNConvert binary + deps
        self.device.shell(f"mkdir -p {conv_dir}", silent=True)
        self.device.push(str(android_convert), f"{conv_dir}/MNNConvert")
        self.device.shell(f"chmod +x {conv_dir}/MNNConvert", silent=True)

        # Push ALL shared lib deps (search recursively from build dir)
        convert_build_root = android_convert.parent
        all_libs = list(convert_build_root.rglob("*.so"))
        for dep in all_libs:
            self.device.push(str(dep), f"{conv_dir}/{dep.name}", silent=True)
        self.logger.debug(f"Pushed {len(all_libs)} shared libs for MNNConvert")

        # Push ONNX files + external data (may be in onnx/ subdirectory)
        onnx_files = list(model_dir.rglob("*.onnx"))
        data_files = list(model_dir.rglob("*.onnx.data"))
        for f in onnx_files + data_files:
            self.device.push(str(f), f"{conv_dir}/{f.name}", silent=True)
        self.logger.ok(f"Pushed {len(onnx_files)} ONNX files + MNNConvert to device")

        # Build MNNConvert flags from config
        conv_cfg = self.config.model.get("conversion", {})
        mnn_convert_opts = conv_cfg.get("mnn_convert_options", {})

        # Convert each ONNX file on device
        for onnx_file in onnx_files:
            mnn_name = onnx_file.stem + ".mnn"
            extra_flags = ""
            # Boolean flags
            for flag in ["transformerFuse", "saveStaticModel", "fp16",
                        "saveExternalData", "forTraining", "hqq",
                        "keepInputFormat", "detectSparseSpeedUp",
                        "groupConvNative", "allowCustomOp"]:
                if mnn_convert_opts.get(flag, flag == "transformerFuse"):  # transformerFuse defaults ON
                    extra_flags += f" --{flag}"
            # Value flags
            for flag in ["optimizePrefer", "optimizeLevel", "weightQuantBits",
                        "weightQuantBlock", "bizCode", "batch",
                        "convertMatmulToConv", "useGeluApproximation",
                        "targetVersion"]:
                if flag in mnn_convert_opts:
                    extra_flags += f" --{flag} {mnn_convert_opts[flag]}"

            convert_cmd = (
                f"export LD_LIBRARY_PATH={conv_dir}:$LD_LIBRARY_PATH && "
                f"cd {conv_dir} && "
                f"./MNNConvert -f ONNX "
                f"--modelFile {onnx_file.name} "
                f"--MNNModel {mnn_name}"
                f"{extra_flags}"
            )
            self.logger.debug(f"Device convert: {convert_cmd}")
            ret = self.device.shell(convert_cmd, capture=True, silent=True)
            if ret.returncode != 0:
                self.logger.err(f"On-device conversion failed for {onnx_file.name}:")
                self.logger.err(ret.stdout[:300] if ret.stdout else ret.stderr[:300] if ret.stderr else "no output")
                raise RuntimeError(f"On-device ONNX→MNN conversion failed: {onnx_file.name}")
            self.logger.ok(f"Converted: {onnx_file.name} → {mnn_name}")

        # Pull converted MNN files back to host model dir (alongside other export artifacts)
        for onnx_file in onnx_files:
            mnn_name = onnx_file.stem + ".mnn"
            local_path = str(model_dir / mnn_name)
            self.device.pull(f"{conv_dir}/{mnn_name}", local_path)
            if os.path.exists(local_path):
                self.logger.ok(f"Pulled: {mnn_name} ({os.path.getsize(local_path) / 1_048_576:.1f} MB)")
            # Also pull .weight file if generated
            weight_path = str(model_dir / f"{mnn_name}.weight")
            self.device.pull(f"{conv_dir}/{mnn_name}.weight", weight_path)

        # Cleanup converter dir on device
        self.device.shell(f"rm -rf {conv_dir}", silent=True)

    # ── Helper: find binaries ─────────────────────────────────────────

    def _find_host_mnnconvert(self) -> Optional[pathlib.Path]:
        """Find a host-runnable (macOS/Linux) MNNConvert binary."""
        for path in [
            MNN_ROOT / "build" / "MNNConvert",
            MNN_ROOT / "build_host" / "MNNConvert",
        ]:
            if path.exists():
                self.logger.debug(f"Found host MNNConvert: {path}")
                return path
        return None

    def _find_android_mnnconvert(self) -> Optional[pathlib.Path]:
        """Find an Android ARM64 MNNConvert binary."""
        candidates = [
            MNN_ROOT / "build_converter_android" / "MNNConvert",
        ]
        # Also search in any build_android_* dirs
        for d in ROOT.glob("build_android_*"):
            candidates.append(d / "MNNConvert")

        for c in candidates:
            if c.exists():
                self.logger.debug(f"Found Android MNNConvert: {c}")
                return c
        return None

    # ── GGUF injection ────────────────────────────────────────────────

    def _inject_gguf(self, gguf_path: str, dst: pathlib.Path):
        gguf_script = MNN_ROOT / "transformers" / "llm" / "export" / "gguf2mnn.py"
        if not gguf_script.exists():
            self.logger.fatal(f"gguf2mnn.py not found: {gguf_script}")
            return

        self.logger.info(f"Injecting GGUF weights from {C.BOLD}{gguf_path}{C.RESET}")
        cmd = [sys.executable, str(gguf_script),
               "--mnn_json", str(dst / "llm.mnn.json"),
               "--gguf_path", gguf_path,
               "--dst_path", str(dst)]
        result = subprocess.run(cmd, cwd=str(MNN_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.err(f"GGUF injection failed:\n{result.stderr[-300:]}")
            raise RuntimeError("GGUF injection failed")
        self.logger.ok("GGUF weights injected")
