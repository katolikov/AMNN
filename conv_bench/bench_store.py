#!/usr/bin/env python3
"""Result store for the conv benchmark: records WHAT a number was measured against, and
refuses the comparisons that are not meaningful.

Every retracted claim in this investigation came from a sound measurement compared against the
wrong thing. Three distinct instances:

  * Section 5 measures every arm (its "MNN default" included) under MNN_NO_WINOGRAD=1, to isolate
    hardcoding against a like-for-like kernel. On the small cores MNN actually DEPLOYS Winograd,
    so that baseline is a path which never runs -- "hardcoding wins 34%" re-measured as 0%.
  * The same conv reads 33 us in section 17 and 99 us in section 4. Both are right inside their
    own section and meaningless against each other; the report explained this in prose and the
    prose did not survive contact with the reader.
  * A leftover {DEV}/output directory flipped ModuleBasic into output-dumping mode, inflating one
    baseline 4.6x (22.5 us recorded as 104 us) and manufacturing a "-62%" win out of nothing.

None of those is a measurement bug, so no amount of care while measuring prevents them. They are
prevented here instead, structurally:

  1. Every row stores `baseline_arm` / `baseline_env`, and every rendered table prints the baseline
     env. A baseline that is not the deployed configuration is labelled in the table, not a footnote.
  2. Comparisons are addressed by batch. There is deliberately NO api that accepts two batch ids;
     cross-batch comparison raises CrossBatchComparison rather than rendering.
  3. Degenerate measurements (a conv cannot take 1 us) mark the row invalid and make the batch
     raise on close, so a broken cell cannot be quietly averaged into a table.

Pure stdlib, no device access. Storage is sqlite so a run is queryable after the fact.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# A 3x3 conv over any shape in this project is tens of microseconds. Anything at or below this is
# not a fast kernel, it is a measurement that did not happen -- an empty parse, a degenerate loop
# count, or the tool running in the wrong mode. See the {DEV}/output failure above.
MIN_PLAUSIBLE_US = 5.0

# Fallback noise floor, used only where variance_probe.py has not measured the configuration.
# It is a GUESS, and a demonstrably wrong one for some configs: 34->32@48x64 in image mode moved
# 71.3us -> 46.2us across batches (54%) on a run whose clock held and whose correctness passed.
# Prefer a measured per-config floor; see load_noise_floors().
NOISE_FLOOR_PCT = 6.0

# {"<conv>|<arm>": spread_pct} from conv_bench/noise_floors.json, if it has been generated.
_FLOORS: dict[str, float] = {}


def load_noise_floors(path="conv_bench/noise_floors.json") -> int:
    """Load measured across-batch spreads. Returns how many configs were loaded.

    Across-batch spread is the right quantity for deciding whether a difference is real. Reps
    inside one batch are interleaved and therefore agree with each other even when the batch as a
    whole is off -- which is exactly how a 54% error passed every check the suite had."""
    global _FLOORS
    p = Path(path)
    if not p.exists():
        return 0
    data = json.loads(p.read_text())
    _FLOORS = {k: v["spread_pct"] for k, v in data.get("floors", {}).items()}
    return len(_FLOORS)


def noise_floor(conv: str, arm: str) -> float:
    """Floor for one configuration: its measured across-batch spread, else the global fallback.

    Never returns less than the fallback -- a config that happened to repeat tightly five times is
    not thereby proven stable to 1%."""
    return max(NOISE_FLOOR_PCT, _FLOORS.get(f"{conv}|{arm}", 0.0))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started       REAL,
    finished      REAL,
    device        TEXT,
    shape_family  TEXT,
    harness       TEXT,
    clock_start   REAL,
    clock_end     REAL,
    clock_valid   INTEGER,
    notes         TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id      TEXT PRIMARY KEY,
    run_id        TEXT,
    section       TEXT,
    label         TEXT,
    reps          INTEGER,
    interleaved   INTEGER,
    baseline_arm  TEXT,
    created       REAL
);
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      TEXT,
    run_id        TEXT,
    conv          TEXT,
    arm           TEXT,
    env           TEXT,
    mode          INTEGER,
    us            REAL,
    spread        REAL,
    n             INTEGER,
    baseline_arm  TEXT,
    baseline_env  TEXT,
    deployed      INTEGER,
    valid         INTEGER,
    invalid_why   TEXT
);
CREATE TABLE IF NOT EXISTS correctness (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    conv          TEXT,
    arm           TEXT,
    env           TEXT,
    mode          INTEGER,
    cosine        REAL,
    passed        INTEGER
);
CREATE INDEX IF NOT EXISTS ix_results_batch ON results(batch_id);
CREATE INDEX IF NOT EXISTS ix_results_conv  ON results(conv);
"""


class CrossBatchComparison(Exception):
    """Raised when two numbers from different interleaved batches would be compared.

    Absolute microseconds are only comparable inside one batch: arms there are interleaved with a
    rotating order, so thermal drift moves both sides together. Across batches -- different
    sections, different runs, different thermal states -- the same conv legitimately reads 33 us
    and 99 us. Comparing those produces a confident, wrong number."""


class DegenerateMeasurement(Exception):
    """Raised on batch close when any recorded cell is not a physically possible measurement."""


class MissingBaseline(Exception):
    """Raised when a batch is rendered or compared without declaring which arm is its baseline."""


class OversizedBatch(Exception):
    """Raised when a batch holds so many arms that drift within it invalidates the comparison."""


class UncheckedKernel(Exception):
    """Raised when timings are reported for a (conv, arm, mode) with no passing correctness gate."""


def is_deployed(env: str, mode: int) -> bool:
    """True when this arm is what MNN actually runs with no interference.

    `env` empty means no MNN_* flags; mode 68 is the buffer/WIDE default the model ships with.
    Anything else -- a forced kernel, MNN_NO_WINOGRAD, image mode -- is an intervention, and a
    baseline that is an intervention must be labelled as one wherever it is rendered."""
    return not env.strip() and mode == 68


class Batch:
    """One interleaved measurement. Arms inside it are mutually comparable; nothing else is."""

    def __init__(self, store: "ResultStore", run_id: str, section: str, label: str,
                 reps: int, interleaved: bool, tolerate_invalid: bool):
        self.store, self.run_id = store, run_id
        self.batch_id = uuid.uuid4().hex[:12]
        self.section, self.label, self.reps = section, label, reps
        self.interleaved, self.tolerate_invalid = interleaved, tolerate_invalid
        self._baseline_arm: str | None = None
        self._rows: list[dict] = []

    # ------------------------------------------------------------------ recording
    def baseline(self, arm: str) -> None:
        """Declare which arm every other arm in this batch is measured against."""
        self._baseline_arm = arm

    def record(self, conv: str, arm: str, us: float, env: str = "", mode: int = 68,
               spread: float | None = None, n: int = 1, samples=None) -> dict:
        """Record one cell. Traps fire here; the batch raises on close if any cell is invalid."""
        if samples:
            samples = [s for s in samples if s]
            us = statistics.median(samples) if samples else 0.0
            n = len(samples)
            if len(samples) > 1:
                spread = max(samples) - min(samples)
        valid, why = True, None
        if us is None or us <= 0:
            valid, why = False, "no measurement parsed (empty output, or the run produced no conv kernel)"
        elif us < MIN_PLAUSIBLE_US:
            valid, why = False, (f"{us:.2f}us is below the {MIN_PLAUSIBLE_US}us plausibility floor "
                                 f"-- a conv cannot be this fast; the run almost certainly did not "
                                 f"execute the timed loops")
        row = dict(conv=conv, arm=arm, env=env, mode=mode, us=float(us or 0.0),
                   spread=spread, n=n, valid=valid, invalid_why=why)
        self._rows.append(row)
        return row

    def correctness(self, conv: str, arm: str, cosine: float, env: str = "", mode: int = 68) -> bool:
        """Record a correctness gate for one (conv, arm, mode). Timings for a triple with no
        passing gate cannot be rendered -- see UncheckedKernel."""
        passed = bool(cosine is not None and cosine > 0.99)
        self.store._conn.execute(
            "INSERT INTO correctness (run_id, conv, arm, env, mode, cosine, passed)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.run_id, conv, arm, env, mode, cosine, int(passed)))
        return passed

    # ------------------------------------------------------------------ close
    def _flush(self) -> None:
        if self._baseline_arm is None:
            raise MissingBaseline(
                f"batch {self.label!r} recorded {len(self._rows)} rows without calling "
                f".baseline(arm). Without it there is no way to know what the numbers were "
                f"measured against -- which is the exact defect this store exists to prevent.")
        base = next((r for r in self._rows if r["arm"] == self._baseline_arm), None)
        if base is None:
            raise MissingBaseline(
                f"batch {self.label!r} declares baseline {self._baseline_arm!r} but never "
                f"recorded it. Arms present: {sorted({r['arm'] for r in self._rows})}")
        b_env, b_mode = base["env"], base["mode"]
        c = self.store._conn
        c.execute("INSERT INTO batches (batch_id, run_id, section, label, reps, interleaved,"
                  " baseline_arm, created) VALUES (?,?,?,?,?,?,?,?)",
                  (self.batch_id, self.run_id, self.section, self.label, self.reps,
                   int(self.interleaved), self._baseline_arm, time.time()))
        for r in self._rows:
            c.execute(
                "INSERT INTO results (batch_id, run_id, conv, arm, env, mode, us, spread, n,"
                " baseline_arm, baseline_env, deployed, valid, invalid_why)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.batch_id, self.run_id, r["conv"], r["arm"], r["env"], r["mode"], r["us"],
                 r["spread"], r["n"], self._baseline_arm, b_env,
                 int(is_deployed(b_env, b_mode)), int(r["valid"]), r["invalid_why"]))
        c.commit()
        bad = [r for r in self._rows if not r["valid"]]
        if bad and not self.tolerate_invalid:
            lines = "\n".join(f"    {r['conv']} / {r['arm']}: {r['invalid_why']}" for r in bad)
            raise DegenerateMeasurement(
                f"batch {self.label!r} produced {len(bad)} impossible cell(s):\n{lines}\n"
                f"  They are stored and marked invalid. Fix the cause before trusting this batch; "
                f"pass tolerate_invalid=True only to inspect a known-broken run.")


class ResultStore:
    def __init__(self, path="conv_bench/results.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self.run_id: str | None = None

    # ------------------------------------------------------------------ runs
    def begin_run(self, device: str, shape_family: str, harness: str = "",
                  clock_start: float | None = None, notes: str = "") -> str:
        self.run_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO runs (run_id, started, device, shape_family, harness, clock_start, notes)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.run_id, time.time(), device, shape_family, harness, clock_start, notes))
        self._conn.commit()
        return self.run_id

    def finish_run(self, clock_end: float | None = None, max_drop_pct: float = 10.0) -> bool:
        """Close the run and decide whether its clock held. Returns clock_valid.

        A run whose clock fell is not discarded -- its within-batch comparisons are still sound,
        because arms are interleaved. It is flagged so absolute microseconds can be treated
        accordingly."""
        row = self._conn.execute("SELECT clock_start FROM runs WHERE run_id=?",
                                 (self.run_id,)).fetchone()
        start = row["clock_start"] if row else None
        valid = 1
        if start and clock_end:
            valid = int(100.0 * (start - clock_end) / start <= max_drop_pct)
        self._conn.execute("UPDATE runs SET finished=?, clock_end=?, clock_valid=? WHERE run_id=?",
                           (time.time(), clock_end, valid, self.run_id))
        self._conn.commit()
        return bool(valid)

    # A batch is only meaningful if every arm in it saw the same thermal state. full_sweep.py put
    # 215 launches (~2h) into one "interleaved" batch; drift across that window was larger than the
    # 10-20% effects being measured, and its verdicts contradicted three shorter runs that agreed
    # with each other. Rotation across reps cannot fix a batch that long. This is the same
    # cross-batch error the store prevents, occurring INSIDE a batch because it was sized wrong.
    MAX_ARMS_PER_BATCH = 40

    @contextmanager
    def batch(self, section: str, label: str, reps: int = 1, interleaved: bool = True,
              tolerate_invalid: bool = False, max_arms: int | None = None):
        b = Batch(self, self.run_id, section, label, reps, interleaved, tolerate_invalid)
        yield b
        cap = max_arms or self.MAX_ARMS_PER_BATCH
        arms = {r["arm"] for r in b._rows}
        if len(arms) > cap:
            raise OversizedBatch(
                f"batch {label!r} holds {len(arms)} arms (cap {cap}). A batch must be short enough "
                f"that thermal drift within it is negligible, or its arms are not comparable and "
                f"the interleaving is decorative. Split it -- one batch per probe model is the "
                f"pattern that worked -- or pass max_arms= if you have certified the clock held.")
        b._flush()

    # ------------------------------------------------------------------ queries
    def rows(self, batch_id: str, conv: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM results WHERE batch_id=?"
        args: list = [batch_id]
        if conv:
            q += " AND conv=?"; args.append(conv)
        return list(self._conn.execute(q + " ORDER BY conv, us", args))

    def batches(self, run_id: str | None = None) -> list[sqlite3.Row]:
        if run_id:
            return list(self._conn.execute("SELECT * FROM batches WHERE run_id=? ORDER BY created",
                                           (run_id,)))
        return list(self._conn.execute("SELECT * FROM batches ORDER BY created"))

    def find(self, conv: str, arm: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM results WHERE conv=? AND arm=? ORDER BY id", (conv, arm)))

    def compare(self, conv: str, arm_a: str, arm_b: str) -> dict:
        """Compare two arms on one conv. Raises unless they share a batch.

        There is no variant of this that takes two batch ids. That is the point: the API cannot
        express the comparison that produced this session's retracted numbers."""
        a_rows, b_rows = self.find(conv, arm_a), self.find(conv, arm_b)
        if not a_rows or not b_rows:
            missing = arm_a if not a_rows else arm_b
            raise LookupError(f"no result for {conv!r} / {missing!r}")
        shared = ({r["batch_id"] for r in a_rows} & {r["batch_id"] for r in b_rows})
        if not shared:
            raise CrossBatchComparison(
                f"{conv}: {arm_a!r} and {arm_b!r} were never measured in the same batch.\n"
                f"  {arm_a!r} in {sorted({r['batch_id'] for r in a_rows})}\n"
                f"  {arm_b!r} in {sorted({r['batch_id'] for r in b_rows})}\n"
                f"  Absolute times are only comparable within one interleaved batch. Measure them "
                f"together instead of differencing these.")
        bid = sorted(shared)[0]
        a = next(r for r in a_rows if r["batch_id"] == bid)
        b = next(r for r in b_rows if r["batch_id"] == bid)
        if not (a["valid"] and b["valid"]):
            bad = a if not a["valid"] else b
            raise DegenerateMeasurement(f"{conv} / {bad['arm']}: {bad['invalid_why']}")
        pct = 100.0 * (b["us"] - a["us"]) / a["us"] if a["us"] else 0.0
        # The claim has to clear the NOISIER of the two configurations: a stable baseline does not
        # make a difference real if the arm it is compared against swings 50% between batches.
        floor = max(noise_floor(conv, arm_a), noise_floor(conv, arm_b))
        return dict(batch_id=bid, conv=conv, a=arm_a, b=arm_b, a_us=a["us"], b_us=b["us"],
                    pct=pct, significant=abs(pct) >= floor, floor=floor,
                    baseline_env=a["baseline_env"], deployed=bool(a["deployed"]))

    def stability(self, conv: str, arm: str) -> dict:
        """Across-batch spread for ONE arm -- the only legitimate cross-batch operation.

        compare() refuses to difference two arms from different batches because that conflates the
        arms with the batches. Asking how much a SINGLE arm moves between batches does not: the arm
        is held fixed and the batch is the variable, which is precisely the measurement wanted."""
        rows = [r for r in self.find(conv, arm) if r["valid"] and r["us"]]
        vals = [r["us"] for r in rows]
        if len(vals) < 2:
            return dict(conv=conv, arm=arm, n=len(vals), spread_pct=None,
                        why="needs at least two batches")
        med = statistics.median(vals)
        return dict(conv=conv, arm=arm, n=len(vals), median=med, lo=min(vals), hi=max(vals),
                    spread_pct=100.0 * (max(vals) - min(vals)) / med,
                    batches=sorted({r["batch_id"] for r in rows}))

    def require_correctness(self, conv: str, arm: str, mode: int = 68) -> None:
        row = self._conn.execute(
            "SELECT passed, cosine FROM correctness WHERE conv=? AND arm=? AND mode=?"
            " ORDER BY id DESC LIMIT 1", (conv, arm, mode)).fetchone()
        if row is None:
            raise UncheckedKernel(f"{conv} / {arm} (mode {mode}) has no correctness gate; its "
                                  f"timings must not be reported")
        if not row["passed"]:
            raise UncheckedKernel(f"{conv} / {arm} (mode {mode}) FAILED its correctness gate "
                                  f"(cosine {row['cosine']:.4f}); its timings are meaningless")

    # ------------------------------------------------------------------ rendering
    def render(self, batch_id: str, check_correctness: bool = False) -> str:
        """Markdown table for one batch. Always prints the baseline env, and says plainly when the
        baseline is not the deployed configuration."""
        bat = self._conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if bat is None:
            raise LookupError(f"no batch {batch_id!r}")
        rows = self.rows(batch_id)
        if not rows:
            return f"### {bat['label']}\n\n_(no rows)_\n"
        base_env = rows[0]["baseline_env"]
        deployed = bool(rows[0]["deployed"])
        out = [f"### {bat['label']}", ""]
        out.append(f"_baseline_: `{bat['baseline_arm']}` "
                   + (f"with env `{base_env.strip()}`" if base_env.strip() else "with no flags")
                   + f" · {bat['reps']} reps · "
                   + ("interleaved" if bat["interleaved"] else "**NOT interleaved**"))
        if not deployed:
            out.append("")
            out.append("> ⚠️ **This baseline is not the deployed configuration.** Percentages below "
                       "are against an intervention, not against what MNN runs by default. They "
                       "answer a kernel-vs-kernel question, not a shipping question.")
        out += ["", "| conv | arm | µs | vs baseline |", "|---|---|---|---|"]
        by_conv: dict[str, list] = {}
        for r in rows:
            by_conv.setdefault(r["conv"], []).append(r)
        for conv, rs in by_conv.items():
            base = next((r for r in rs if r["arm"] == bat["baseline_arm"]), None)
            for r in sorted(rs, key=lambda x: x["us"]):
                if not r["valid"]:
                    out.append(f"| {conv} | {r['arm']} | — | **INVALID** — {r['invalid_why']} |")
                    continue
                if check_correctness:
                    self.require_correctness(conv, r["arm"], r["mode"])
                if base and base["valid"] and base["us"] and r["arm"] != bat["baseline_arm"]:
                    pct = 100.0 * (r["us"] - base["us"]) / base["us"]
                    fl = max(noise_floor(conv, r["arm"]),
                             noise_floor(conv, bat["baseline_arm"]))
                    d = (f"{pct:+.0f}%" if abs(pct) >= fl
                         else f"{pct:+.0f}% (noise, floor {fl:.0f}%)")
                else:
                    d = "baseline" if r["arm"] == bat["baseline_arm"] else "—"
                out.append(f"| {conv} | {r['arm']} | {r['us']:.1f} | {d} |")
        return "\n".join(out) + "\n"

    def close(self):
        self._conn.close()
