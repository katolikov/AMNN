"""Shared constants, colors, and the stage flag registry."""

from __future__ import annotations
import pathlib
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent   # LLM_Benchmark/
MNN_ROOT = ROOT.parent                                  # MNN repo root
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
LOGCAT_BUFFER_SIZE = "64M"

# ── ANSI Colors ──
_USE_COLOR = sys.stdout.isatty()

class C:
    RESET   = "\033[0m"   if _USE_COLOR else ""
    BOLD    = "\033[1m"   if _USE_COLOR else ""
    DIM     = "\033[2m"   if _USE_COLOR else ""
    RED     = "\033[91m"  if _USE_COLOR else ""
    GREEN   = "\033[92m"  if _USE_COLOR else ""
    YELLOW  = "\033[93m"  if _USE_COLOR else ""
    BLUE    = "\033[94m"  if _USE_COLOR else ""
    MAGENTA = "\033[95m"  if _USE_COLOR else ""
    CYAN    = "\033[96m"  if _USE_COLOR else ""
    WHITE   = "\033[97m"  if _USE_COLOR else ""
    CHECK   = "✓" if _USE_COLOR else "OK"
    CROSS   = "✗" if _USE_COLOR else "FAIL"
    ARROW   = "→" if _USE_COLOR else "->"

# ── Stage Flag Registry ──
# Adding a new benchmark flag = adding ONE entry here.
# (stage_key, cli_flag, type, default, condition_fn_or_None)
STAGE_FLAGS: List[tuple] = [
    # Core
    ("backend",         "--backend",         str,   "cpu",     None),
    ("precision",       "--precision",       str,   "low",     None),
    ("memory",          "--memory",          str,   "low",     None),
    ("power",           "--power",           str,   "normal",  None),
    ("attention_mode",  "--attention_mode",  int,   0,         None),
    # CPU-specific
    ("threads",         "--threads",         int,   4,         lambda s: s.get("backend", "cpu") == "cpu"),
    # Benchmark params
    ("warmup_rounds",   "--warmup",          int,   2,         None),
    ("measure_rounds",  "--rounds",          int,   5,         None),
    ("prompt_tokens",   "--prompt_tokens",   int,   128,       None),
    ("max_gen_tokens",  "--max_gen_tokens",  int,   64,        None),
    # Engine options
    ("use_mmap",        "--use_mmap",        bool,  False,     None),
    ("max_all_tokens",  "--max_all_tokens",  int,   0,         None),
    ("dynamic_quant",   "--dynamic_quant",   int,   0,         None),
    ("kvcache_limit",   "--kvcache_limit",   int,   -1,        None),
    # Boolean flags
    ("enable_op_profile", "--op_profile",    bool,  False,     None),
    ("no_tuning",       "--no_tuning",       bool,  False,     None),
]
