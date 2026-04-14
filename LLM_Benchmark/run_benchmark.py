#!/usr/bin/env python3
"""
MNN LLM/VLM Benchmark Suite — Python Orchestrator
===================================================

Class-based architecture for easy extensibility.
All classes live in ``lib/`` — this file is just the entry point.

Usage::
    python run_benchmark.py benchmark_config.json           # run benchmarks
    python run_benchmark.py app_config.json --build-app     # build Android app
"""

from __future__ import annotations

import argparse
import sys

from lib import BenchmarkPipeline, AppBuilder


def main():
    parser = argparse.ArgumentParser(description="MNN LLM/VLM Benchmark Suite")
    parser.add_argument("config",
                       help="Path to config JSON (benchmark config or app config)")
    parser.add_argument("--debug-log", action="store_true",
                       help="Enable verbose debug logging")
    parser.add_argument("--build-app", action="store_true",
                       help="Build the Android app using <config> as the app config")
    parser.add_argument("--icon", default=None,
                       help="Path to a PNG icon for the Android app (used with --build-app)")
    args = parser.parse_args()

    if args.build_app:
        builder = AppBuilder(args.config, icon_path=args.icon, debug=args.debug_log)
        builder.build()
        return

    pipeline = BenchmarkPipeline(args.config, debug=args.debug_log)
    pipeline.run()


if __name__ == "__main__":
    main()
