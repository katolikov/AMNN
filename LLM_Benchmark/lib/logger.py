"""Centralized logging with optional debug mode."""

from __future__ import annotations
import sys
from datetime import datetime
from .constants import C


class Logger:
    def __init__(self, debug: bool = False):
        self.debug_mode = debug

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def info(self, msg: str) -> None:
        print(f"  {C.DIM}{self._ts()}{C.RESET}  {msg}", flush=True)

    def ok(self, msg: str) -> None:
        print(f"  {C.DIM}{self._ts()}{C.RESET}  {C.GREEN}{C.CHECK}{C.RESET} {msg}", flush=True)

    def warn(self, msg: str) -> None:
        print(f"  {C.DIM}{self._ts()}{C.RESET}  {C.YELLOW}⚠{C.RESET}  {C.YELLOW}{msg}{C.RESET}", flush=True)

    def err(self, msg: str) -> None:
        print(f"  {C.DIM}{self._ts()}{C.RESET}  {C.RED}{C.CROSS}{C.RESET} {C.RED}{msg}{C.RESET}", flush=True)

    def step(self, num: int, total: int, title: str) -> None:
        bar = f"{C.BOLD}{C.CYAN}[{num}/{total}]{C.RESET}"
        print(f"\n{bar} {C.BOLD}{title}{C.RESET}")
        print(f"{'─' * 60}")

    def cmd(self, cmd_str: str) -> None:
        if len(cmd_str) > 120:
            cmd_str = cmd_str[:117] + "..."
        print(f"  {C.DIM}  $ {cmd_str}{C.RESET}", flush=True)

    def push(self, label: str, size_bytes: int = 0) -> None:
        size_s = ""
        if size_bytes > 1_048_576:
            size_s = f" {C.DIM}({size_bytes / 1_048_576:.1f} MB){C.RESET}"
        elif size_bytes > 1024:
            size_s = f" {C.DIM}({size_bytes / 1024:.0f} KB){C.RESET}"
        elif size_bytes > 0:
            size_s = f" {C.DIM}({size_bytes} B){C.RESET}"
        print(f"    {C.DIM}{C.ARROW}{C.RESET} {label}{size_s}", flush=True)

    def debug(self, msg: str) -> None:
        if self.debug_mode:
            print(f"  {C.DIM}[DEBUG {self._ts()}] {msg}{C.RESET}", flush=True)

    def fatal(self, msg: str) -> None:
        self.err(msg)
        sys.exit(1)
