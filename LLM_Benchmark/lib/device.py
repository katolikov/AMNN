"""ADB device communication, file push/pull, cache, clock and thermal management."""

from __future__ import annotations
import atexit
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from .constants import C, LOGCAT_BUFFER_SIZE
from .config import Config
from .logger import Logger


class DeviceManager:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.adbk = shutil.which("adbk") or shutil.which("adb") or "adb"
        self._sessions_created: set = set()
        self._cleanup_registered = False
        self._clocks_applied = False

    def _adb_cmd(self) -> List[str]:
        base = [self.adbk]
        if self.config.device_id:
            base += ["-s", self.config.device_id]
        return base

    def adb(self, *args: str, check=True, capture=False, silent=False) -> subprocess.CompletedProcess:
        cmd = self._adb_cmd() + list(args)
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
        if not silent:
            self.logger.cmd(cmd_str)
        self.logger.debug(f"ADB: {cmd_str}")
        return subprocess.run(cmd, check=check, capture_output=capture, text=True, errors="replace")

    def shell(self, cmd_str: str, root=False, **kw) -> subprocess.CompletedProcess:
        inner = cmd_str
        if root:
            # Android su syntax: su 0 sh -c 'command'
            # (NOT Linux-style 'su -c command' which Android su misparses)
            inner = f"su 0 sh -c {shlex.quote(cmd_str)}"
        return self.adb("shell", inner, **kw)

    def push(self, src: str, dst: str, silent=False) -> None:
        self.logger.debug(f"Push: {src} -> {dst}")
        self.adb("push", src, dst, silent=silent)

    def pull(self, src: str, dst: str) -> None:
        self.logger.debug(f"Pull: {src} -> {dst}")
        self.adb("pull", src, dst, check=False, silent=True)

    # ── Session management ──
    def create_session(self):
        if "adbk" not in self.adbk:
            return
        dev = self.config.device_id
        if not dev:
            return
        current_user = os.environ.get("USER", "unknown")
        ret = subprocess.run(f"{self.adbk} --status", shell=True, capture_output=True, text=True)
        for line in ret.stdout.splitlines():
            if dev in line and "Session" in line and current_user in line:
                self.logger.info(f"Reusing adbk session for {C.CYAN}{dev}{C.RESET}")
                return
        ret = subprocess.run(f"{self.adbk} -s {dev} --create-session", shell=True, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f"adbk --create-session failed: {ret.stderr.strip()}")
        self._sessions_created.add(dev)
        self.logger.ok(f"Created adbk session for {C.CYAN}{dev}{C.RESET}")
        if not self._cleanup_registered:
            atexit.register(self._cleanup_sessions)
            self._cleanup_registered = True

    def _cleanup_sessions(self):
        for dev in list(self._sessions_created):
            subprocess.run(f"{self.adbk} -s {dev} --delete-session", shell=True, capture_output=True, text=True)

    # ── Cache management ──
    def push_cache(self, local_path: str):
        if not os.path.exists(local_path):
            self.logger.warn(f"Cache file not found: {local_path}")
            return
        target = f"{self.config.target_dir}/tmp"
        self.shell(f"mkdir -p {target}", silent=True)
        self.push(local_path, f"{target}/mnn_cachefile.bin")
        self.logger.ok(f"Pushed cache: {pathlib.Path(local_path).name}")

    def pull_cache(self, local_path: str):
        target = f"{self.config.target_dir}/tmp/mnn_cachefile.bin"
        self.pull(target, local_path)
        if os.path.exists(local_path):
            self.logger.ok(f"Pulled cache: {pathlib.Path(local_path).name}")
        else:
            self.logger.warn("Cache file not found on device")

    # ── Clock management ──
    def apply_clocks(self):
        settings = self.config.device.get("clock_settings", {})
        if not settings:
            self.logger.info(f"{C.DIM}No clock settings configured{C.RESET}")
            return
        need_root = self.config.device.get("clock_root", True)
        for path, val in settings.items():
            self.shell(f"echo {val} > {path}", root=need_root, check=False, silent=True)
        self._clocks_applied = True
        self.logger.ok(f"Clock settings applied ({len(settings)} entries, root={'yes' if need_root else 'no'})")

    def restore_clocks(self):
        if not self._clocks_applied:
            return
        need_root = self.config.device.get("clock_root", True)
        restore = self.config.device.get("clock_restore", {})
        for path, val in restore.items():
            self.shell(f"echo {val} > {path}", root=need_root, check=False, silent=True)
        self.logger.ok("Clocks restored")

    # ── Thermal cooldown ──
    def read_temperature(self, sensor_path: str) -> Optional[int]:
        """Read temperature from a sysfs sensor on device.  Returns °C or None."""
        ret = self.shell(f"cat {sensor_path}", capture=True, check=False, silent=True)
        if ret.returncode != 0 or not ret.stdout.strip():
            return None
        try:
            raw = float(ret.stdout.strip())
            # Some sensors report milli-degrees (e.g. 42000 = 42°C)
            if raw > 1000:
                return int(raw / 1000)
            return int(raw)
        except ValueError:
            return None

    def wait_for_cooldown(self, stage_name: str = "") -> None:
        """Block until device temperature drops below the configured threshold.

        Config keys (under "device"):
            thermal_sensor   – sysfs path  (default: /sys/kernel/gpu/gpu_tmu)
            thermal_target   – target °C   (default: 25)
            thermal_timeout  – max wait s  (default: 300)
            thermal_poll     – poll interval s (default: 5)
        """
        sensor = self.config.device.get("thermal_sensor", "/sys/kernel/gpu/gpu_tmu")
        target = self.config.device.get("thermal_target", 25)
        if not sensor or target <= 0:
            return

        timeout = int(self.config.device.get("thermal_timeout", 300))
        poll_s  = int(self.config.device.get("thermal_poll", 5))

        # Initial read
        temp = self.read_temperature(sensor)
        if temp is None:
            self.logger.warn(f"Cannot read thermal sensor: {sensor}")
            return
        if temp <= target:
            self.logger.ok(f"GPU temp {temp}°C ≤ {target}°C — ready")
            return

        # Need to wait — show progress bar
        label = f"before {stage_name}" if stage_name else ""
        self.logger.info(
            f"GPU temp {C.YELLOW}{temp}°C{C.RESET} > {target}°C — "
            f"cooling down {label}"
        )

        BAR_WIDTH = 30
        t0 = time.time()
        start_temp = temp

        while temp > target:
            elapsed = time.time() - t0
            if elapsed >= timeout:
                self.logger.warn(
                    f"Thermal timeout ({timeout}s) — proceeding at {temp}°C"
                )
                return

            # Progress: how far from start_temp → target
            if start_temp > target:
                progress = min(1.0, (start_temp - temp) / (start_temp - target))
            else:
                progress = 1.0
            filled = int(BAR_WIDTH * progress)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            remaining = (elapsed / max(progress, 0.01)) * (1.0 - progress) if progress > 0.05 else 0
            eta_str = f"~{int(remaining)}s" if remaining > 0 else "..."

            sys.stdout.write(
                f"\r  {C.CYAN}🌡  [{bar}] {temp}°C → {target}°C  "
                f"({int(elapsed)}s elapsed, ETA {eta_str}){C.RESET}  "
            )
            sys.stdout.flush()

            time.sleep(poll_s)
            temp = self.read_temperature(sensor)
            if temp is None:
                sys.stdout.write("\n")
                self.logger.warn("Lost thermal sensor reading, proceeding")
                return

        # Done
        elapsed = time.time() - t0
        sys.stdout.write(
            f"\r  {C.GREEN}🌡  [{'█' * BAR_WIDTH}] {temp}°C ≤ {target}°C  "
            f"(cooled in {int(elapsed)}s){C.RESET}" + " " * 20 + "\n"
        )
        sys.stdout.flush()
        self.logger.ok(f"GPU cooled to {temp}°C in {int(elapsed)}s")

    # ── Push all files ──
    def push_all(self, binary: pathlib.Path, model_dir: pathlib.Path):
        target = self.config.target_dir
        self.shell(f"rm -rf {target} && mkdir -p {target}/tmp {target}/model {target}/images", silent=True)

        self.push(str(binary), f"{target}/llm_benchmark")
        self.shell(f"chmod +x {target}/llm_benchmark", silent=True)
        self.logger.push("llm_benchmark", binary.stat().st_size)

        total_model, count = 0, 0
        for f in sorted(model_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(model_dir)
                dst = f"{target}/model/{rel}"
                self.shell(f"mkdir -p {str(pathlib.Path(dst).parent)}", silent=True)
                self.push(str(f), dst, silent=True)
                total_model += f.stat().st_size
                count += 1
        self.logger.push(f"model/ ({count} files)", total_model)

        images = self.config.inputs.get("images", [])
        for img_path in images:
            if os.path.exists(img_path):
                self.push(str(img_path), f"{target}/images/{pathlib.Path(img_path).name}", silent=True)
        if images:
            self.logger.push(f"images/ ({len(images)} files)")

        build_dir = binary.parent
        libs = list(build_dir.rglob("*.so"))
        total_libs = sum(l.stat().st_size for l in libs)
        for lib in libs:
            self.push(str(lib), f"{target}/{lib.name}", silent=True)
        self.logger.push(f"{len(libs)} shared libs", total_libs)
        self.logger.ok(f"All files pushed to {target}")

    # ── Logcat ──
    def clear_logcat(self):
        self.adb("logcat", "-c", silent=True, check=False)
        self.logger.ok("Logcat buffer cleared")

    def capture_logcat(self, output_dir: pathlib.Path):
        self.adb("logcat", "-G", LOGCAT_BUFFER_SIZE, silent=True, check=False)
        self.logger.ok(f"Logcat buffer resized to {LOGCAT_BUFFER_SIZE}")

        full_path = output_dir / "logcat_capture.txt"
        ret = self.adb("logcat", "-d", capture=True, silent=True, check=False)
        full_path.write_text(ret.stdout)
        self.logger.ok(f"Full logcat: {ret.stdout.count(chr(10))} lines -> {full_path.name}")

        mnnjni_path = output_dir / "logcat_mnnjni.txt"
        mnnjni_lines = [l for l in ret.stdout.splitlines() if "MNNJNI" in l]
        mnnjni_path.write_text("\n".join(mnnjni_lines))
        self.logger.info(f"  MNNJNI:  {len(mnnjni_lines)} lines -> {mnnjni_path.name}")

        crash_kw = ["FATAL", "tombstone", "SIGSEGV", "SIGABRT", "backtrace"]
        crashes = [l for l in ret.stdout.splitlines() if any(k in l for k in crash_kw)]
        if crashes:
            (output_dir / "logcat_crashes.txt").write_text("\n".join(crashes))
            self.logger.warn(f"  CRASHES:  {len(crashes)} lines")
        else:
            self.logger.info(f"  CRASHES:  none {C.GREEN}OK{C.RESET}")

        error_kw = ["error", "Error", "ERROR", "fail", "Fail", "FAIL"]
        mnn_errors = [l for l in mnnjni_lines if any(k in l for k in error_kw) and "CPU Group" not in l and "Cache" not in l]
        if mnn_errors:
            (output_dir / "logcat_mnn_errors.txt").write_text("\n".join(mnn_errors))
        self.logger.info(f"  MNN ERRORS: {'none ' + C.GREEN + 'OK' + C.RESET if not mnn_errors else str(len(mnn_errors)) + ' lines'}")

    def pull_artifacts(self, output_dir: pathlib.Path):
        dst = output_dir / "device_tmp"
        dst.mkdir(parents=True, exist_ok=True)
        self.pull(f"{self.config.target_dir}/tmp/", str(dst))
