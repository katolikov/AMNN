#!/usr/bin/env python3
"""Render a run's results as three recommendation tables.

gpuMode is set per Interpreter, so a model gets ONE memory mode even when its convs disagree. A
single "best per conv" table therefore answers a question nobody can act on directly: it mixes
buffer and image winners that cannot coexist. These three answer the questions that map onto a
decision:

    1. BUFFER   -- if this model runs in buffer mode, what is the best kernel for each conv?
    2. IMAGE    -- if it runs in image mode, likewise.
    3. OVERALL  -- the best arm regardless of mode, against the deployed baseline. Useful for
                   sizing the prize, and for convs that sit in their own submodel.

Every percentage is measured against the baseline named in the table and checked against that
configuration's own measured noise floor; anything smaller is reported as noise, not a win.
"""
from __future__ import annotations

import bench_store as BS

BUFFER, IMAGE = 68, 132


def _rows_by_conv(st, run_id):
    out = {}
    for b in st._conn.execute("SELECT batch_id, baseline_arm FROM batches WHERE run_id=?"
                              " ORDER BY created", (run_id,)):
        for r in st.rows(b["batch_id"]):
            if r["valid"]:
                out.setdefault(r["conv"], []).append(r)
    return out


def _floor(conv):
    return max(BS.noise_floor(conv, "buffer"), BS.noise_floor(conv, "image"))


def _table(title, convs, pick, baseline_arm, note=""):
    """pick(rows) -> the candidate rows for this table; baseline_arm names the reference."""
    lines = [f"\n{title}", "-" * len(title)]
    if note:
        lines.append(note)
    lines.append(f"{'conv':<20}{'baseline':>10}{'best':>10}  {'arm':<24}{'gain':>7}  verdict")
    wins = 0
    for conv in sorted(convs):
        rows = pick(convs[conv])
        base = next((r for r in rows if r["arm"] == baseline_arm), None)
        if not base or not rows:
            continue
        best = min(rows, key=lambda r: r["us"])
        gain = 100 * (best["us"] - base["us"]) / base["us"] if base["us"] else 0.0
        fl = _floor(conv)
        sig = abs(gain) >= fl and best["arm"] != baseline_arm
        wins += sig
        arm = best["arm"] if sig else baseline_arm
        shown = best["us"] if sig else base["us"]
        lines.append(f"{conv:<20}{base['us']:>10.1f}{shown:>10.1f}  {arm:<24}"
                     f"{gain:>6.0f}%  {'USE IT' if sig else 'keep default'}")
    lines.append(f"({wins} of {len(convs)} convs improve on the default in this mode)")
    return "\n".join(lines)


def render(st, run_id, floors_path=None):
    if floors_path:
        BS.load_noise_floors(floors_path)
    convs = _rows_by_conv(st, run_id)
    if not convs:
        return "(no results for this run)"
    out = []
    out.append(_table(
        "1. BUFFER MODE  (gpuMode 68)", convs,
        lambda rs: [r for r in rs if r["mode"] == BUFFER], "buffer default",
        "the best kernel per conv if the model runs in buffer mode"))
    out.append(_table(
        "2. IMAGE MODE  (gpuMode 132)", convs,
        lambda rs: [r for r in rs if r["mode"] == IMAGE], "image default",
        "the best kernel per conv if the model runs in image mode.\n"
        "NOTE: MNN_CONV_FORCE/SPEC/HARD are read only by the buffer backend, so image mode has no\n"
        "per-kernel choice -- the arms here differ by algorithm (Winograd) only."))
    out.append(_table(
        "3. OVERALL  (best arm, either mode)", convs,
        lambda rs: rs, "buffer default",
        "against the DEPLOYED baseline (buffer, no flags). gpuMode is per Interpreter, so a\n"
        "buffer and an image winner in this table cannot both be used unless the convs live in\n"
        "different submodels -- confirm at whole-model wall-clock before shipping."))
    return "\n".join(out)
