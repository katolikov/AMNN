#!/usr/bin/env python3
"""
MNN LLM/VLM Benchmark Suite — Python Orchestrator
===================================================

Class-based architecture for easy extensibility.
All classes live in ``lib/`` — this file is just the entry point.

Usage::
    python run_benchmark.py config.json              # normal mode
    python run_benchmark.py config.json --debug       # verbose debug output
"""

from __future__ import annotations

import argparse
import sys

from lib import BenchmarkPipeline


def main():
    parser = argparse.ArgumentParser(description="MNN LLM/VLM Benchmark Suite")
    parser.add_argument("config", nargs="?", default="benchmark_config.json",
                       help="Path to benchmark config JSON")
    parser.add_argument("--debug-log", action="store_true",
                       help="Enable verbose debug logging")
    args = parser.parse_args()
    pipeline = BenchmarkPipeline(args.config, debug=args.debug_log)
    pipeline.run()


if __name__ == "__main__":
    main()
