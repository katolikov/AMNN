"""Cross-compilation for Android NDK."""

from __future__ import annotations
import subprocess
import time
import pathlib

from .constants import ROOT, MNN_ROOT, C
from .config import Config
from .logger import Logger


class Builder:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def _build_dir(self) -> pathlib.Path:
        build = self.config.build
        build_type = build.get("build_type", "Release").lower()
        parts = ["build_android", build_type]
        if build.get("use_vulkan_image", False):
            parts.append("vulkan_image")
        return ROOT / "_".join(parts)

    def build(self) -> pathlib.Path:
        build_cfg = self.config.build
        build_dir = self._build_dir()
        build_dir.mkdir(parents=True, exist_ok=True)

        build_type = build_cfg.get("build_type", "Release")
        generator = build_cfg.get("generator", "")
        jobs = build_cfg.get("jobs", 8)

        self.logger.info(f"Build type: {C.BOLD}{build_type}{C.RESET}  |  Generator: {C.BOLD}{generator or 'Make'}{C.RESET}")

        cmake_path = build_cfg.get("cmake_path", "cmake")
        ndk_path = build_cfg.get("ndk_path", "")
        cmake_args = [
            cmake_path, str(MNN_ROOT),
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_TOOLCHAIN_FILE={ndk_path}/build/cmake/android.toolchain.cmake",
            "-DMNN_BUILD_LLM=ON", "-DMNN_LOW_MEMORY=ON",
            "-DMNN_BUILD_OPENCV=ON", "-DMNN_IMGCODECS=ON", "-DMNN_BUILD_AUDIO=ON",
            "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON",
            "-DMNN_BUILD_BENCH_SUITE=ON",
            f"-DBENCH_SUITE_DIR={ROOT}",
        ]
        if generator:
            cmake_args += ["-G", generator]
        for flag in build_cfg.get("cmake_flags", []):
            cmake_args.append(flag)

        # Auto-add MNN_GPU_TIME_PROFILE if any stage has op profiling enabled
        needs_gpu_profile = any(s.get("enable_op_profile", False) for s in self.config.stages)
        has_gpu_profile = any("MNN_GPU_TIME_PROFILE" in f for f in build_cfg.get("cmake_flags", []))
        if needs_gpu_profile and not has_gpu_profile:
            cmake_args.append("-DMNN_GPU_TIME_PROFILE=ON")
            self.logger.info("  Auto-enabled MNN_GPU_TIME_PROFILE (op profiling stage detected)")

        # Detect if CMake cache has different flags than what we need — force reconfigure
        cache_file = build_dir / "CMakeCache.txt"
        if cache_file.exists():
            cache_text = cache_file.read_text()
            needs_reconfigure = False
            # Check GPU profiling flag consistency
            if needs_gpu_profile and "MNN_GPU_TIME_PROFILE:BOOL=ON" not in cache_text:
                needs_reconfigure = True
            # Check SEP_BUILD consistency
            if "MNN_SEP_BUILD:BOOL=OFF" not in cache_text and "MNN_SEP_BUILD" not in cache_text:
                needs_reconfigure = True
            if needs_reconfigure:
                cache_file.unlink()
                self.logger.info("  Cleared CMake cache (build flags changed, forcing full reconfigure)")

        self.logger.info(f"  Configuring CMake ({len(cmake_args)} args)...")
        self.logger.debug(f"CMake args: {cmake_args}")
        result = subprocess.run(cmake_args, cwd=str(build_dir), capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.err(f"CMake configure failed:\n{result.stderr[-500:]}")
            raise RuntimeError("CMake configure failed")
        self.logger.ok("CMake configured")

        self.logger.info(f"  Building with {jobs} parallel jobs...")
        build_cmd = [cmake_path, "--build", str(build_dir), "--target", "llm_benchmark", "-j", str(jobs)]
        self.logger.cmd(" ".join(build_cmd))
        t0 = time.time()
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(result.stderr[-1000:])
            raise RuntimeError("Build failed")

        binary = build_dir / "llm_benchmark"
        if not binary.exists():
            for candidate in build_dir.rglob("llm_benchmark"):
                if candidate.is_file():
                    binary = candidate
                    break

        size_mb = binary.stat().st_size / 1_048_576
        self.logger.ok(f"Built llm_benchmark ({size_mb:.1f} MB) in {elapsed:.0f}s")
        return binary
