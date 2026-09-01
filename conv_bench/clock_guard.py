#!/usr/bin/env python3
"""GPU clock/thermal telemetry: settle on temperature, and certify a batch from what the clock
ACTUALLY did rather than from a sample taken beside it.

Three things this replaces:

  * Fixed sleeps. Every section sleeps 20-40s between shapes whether the device needs it or not.
    /sys/kernel/gpu/gpu_tmu is the GPU die temperature and is world-readable on this device, so we
    can wait for the actual condition instead of guessing at it -- usually far less than 35s, and
    occasionally more, which is the case a fixed sleep gets wrong in the dangerous direction.

  * sample_clock(). It runs a ~6 second GPU workload purely to observe the clock, which both costs
    time and heats the part it is measuring. gpu_clock_stats is a cumulative time-per-frequency
    table; differencing two snapshots gives the time-weighted mean frequency over exactly the
    interval between them, for free and without touching the GPU.

  * Prose warnings about throttling. A batch now carries the mean frequency it ran at and the
    fraction of time at nominal, so "this batch is not comparable" becomes a recorded fact.

Pinning the clock needs root, which this device does not have (`adb root` is refused on production
builds, there is no `su`, and /sys/kernel/gpu/gpu_governor is root-owned). pin() is implemented and
degrades to a clear no-op report, so a rooted or engineering-build device gets it for free.
"""
from __future__ import annotations

import re
import subprocess
import time

GPU = "/sys/kernel/gpu"


class GpuTelemetry:
    def __init__(self, serial: str):
        self.s = serial
        self.nominal = self._max_clock_khz()

    # ------------------------------------------------------------------ raw reads
    def _sh(self, cmd: str, timeout: int = 30) -> str:
        try:
            return subprocess.run(f"adb -s {self.s} shell '{cmd}'", shell=True, text=True,
                                  capture_output=True, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            return ""

    def _max_clock_khz(self) -> int:
        try:
            return int(self._sh(f"cat {GPU}/gpu_max_clock").strip() or 0)
        except ValueError:
            return 0

    def temperature_c(self) -> float | None:
        """GPU die temperature in Celsius, or None if the node is unreadable."""
        raw = self._sh(f"cat {GPU}/gpu_tmu").strip()
        try:
            return float(raw)
        except ValueError:
            return None

    def freq_stats(self) -> dict[int, int]:
        """{frequency_khz: cumulative_microseconds_at_that_frequency}."""
        out, res = self._sh(f"cat {GPU}/gpu_clock_stats"), {}
        for line in out.splitlines():
            m = re.match(r"\s*(\d+)\s+(\d+)\s*$", line)
            if m:
                res[int(m.group(1))] = int(m.group(2))
        return res

    # ------------------------------------------------------------------ settling
    def settle(self, target_c: float = 40.0, max_wait: int = 120, poll: int = 5,
               plateau_delta: float = 0.5) -> dict:
        """Wait until the GPU is cool enough to measure on.

        Returns when the temperature is at or below target_c, or when it stops falling (a device
        idling at 45C in a warm room will never reach 40, and waiting the full timeout for that
        every time would be worse than the fixed sleep this replaces)."""
        t0, last, readings = time.time(), None, []
        while time.time() - t0 < max_wait:
            t = self.temperature_c()
            if t is None:
                return dict(ok=False, why="gpu_tmu unreadable", waited=time.time() - t0)
            readings.append(t)
            if t <= target_c:
                return dict(ok=True, temp=t, waited=round(time.time() - t0, 1),
                            reason="reached target")
            if last is not None and (last - t) < plateau_delta:
                return dict(ok=True, temp=t, waited=round(time.time() - t0, 1),
                            reason=f"plateaued at {t:.1f}C (still falling < {plateau_delta}C/poll)")
            last = t
            time.sleep(poll)
        return dict(ok=True, temp=readings[-1] if readings else None,
                    waited=round(time.time() - t0, 1), reason="max_wait reached")

    # ------------------------------------------------------------------ pinning
    def pin(self, governor: str = "performance") -> dict:
        """Pin the GPU clock. Needs root; reports plainly when it cannot."""
        before = self._sh(f"cat {GPU}/gpu_governor").strip()
        err = self._sh(f"echo {governor} > {GPU}/gpu_governor 2>&1")
        after = self._sh(f"cat {GPU}/gpu_governor").strip()
        if after == governor:
            return dict(pinned=True, governor=after, was=before)
        return dict(pinned=False, governor=after, was=before,
                    why=(err.strip() or "write rejected") +
                        " -- pinning needs root; this build refuses `adb root` and has no `su`, so "
                        "every measurement runs under the governor and must be certified after "
                        "the fact via Watchdog instead")


class Watchdog:
    """Certify one batch from the clock it actually ran at.

        with Watchdog(tel) as w:
            ...measure...
        w.result  ->  {'mean_mhz':.., 'pct_at_nominal':.., 'valid':.., 'temp_start':.., ...}

    Differencing gpu_clock_stats over the batch is exact and free: it reports where the time
    genuinely went, not where a probe happened to catch the clock. A batch that spent most of its
    time below nominal is flagged -- its internal A/B comparisons remain sound (arms are
    interleaved), but its absolute microseconds are not comparable with anything else."""

    # Validity is judged on the MEAN frequency, not on the fraction of time at exactly nominal.
    # A real batch measured 62% at nominal but a mean of 954 MHz -- 97% of nominal, i.e. a 3%
    # effect, far inside the noise floor. Failing that batch would reject good data: the governor
    # dips a step or two constantly, and what a measurement actually experiences is the mean.
    def __init__(self, tel: GpuTelemetry, min_mean_ratio: float = 0.95):
        self.tel, self.min_mean_ratio = tel, min_mean_ratio
        self.result: dict = {}

    def __enter__(self):
        self._t0 = time.time()
        self._stats0 = self.tel.freq_stats()
        self._temp0 = self.tel.temperature_c()
        return self

    def __exit__(self, *exc):
        stats1 = self.tel.freq_stats()
        temp1 = self.tel.temperature_c()
        delta = {f: stats1.get(f, 0) - self._stats0.get(f, 0) for f in stats1}
        delta = {f: us for f, us in delta.items() if us > 0}
        total = sum(delta.values())
        if not total:
            self.result = dict(valid=None, why="no clock-stats movement (idle or node unreadable)",
                               temp_start=self._temp0, temp_end=temp1)
            return False
        mean_khz = sum(f * us for f, us in delta.items()) / total
        nominal = self.tel.nominal or max(delta)
        pct_nom = 100.0 * sum(us for f, us in delta.items() if f >= nominal * 0.98) / total
        ratio = mean_khz / nominal if nominal else 0.0
        self.result = dict(
            valid=ratio >= self.min_mean_ratio,
            mean_mhz=round(mean_khz / 1000, 1),
            nominal_mhz=round(nominal / 1000, 1),
            mean_ratio=round(ratio, 3),
            pct_at_nominal=round(pct_nom, 1),     # context only; not the validity test
            temp_start=self._temp0, temp_end=temp1,
            seconds=round(time.time() - self._t0, 1),
            why=None if ratio >= self.min_mean_ratio else
                f"mean {mean_khz/1000:.0f} MHz is {100*(1-ratio):.0f}% below nominal "
                f"({nominal/1000:.0f} MHz); absolute times from this batch are inflated")
        return False
