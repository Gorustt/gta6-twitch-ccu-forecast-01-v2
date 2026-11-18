"""Forecast GTA VI launch-day Twitch viewers, hour by hour (best version).

Reads every ``*.in`` case in the input directory and writes one ``*.out``
forecast per case: for hours 2..24, five quantiles of that hour's viewer
count. Python 3.13, standard library only, offline, deterministic.

Method (rolling-midnight comparables, two median models blended in the
mid-day dip window):

1. Every comparable past launch contributes its release-day hourly curve as
   log-ratios to its own hour-1 print, ``lr_h = ln(v_h / v_1)``.
2. Reference class: launches whose unlock note says the launch rolled out
   at midnight local time ("rolling" / "midnight local"). Simultaneous
   global unlocks have a different shape and are excluded.
3. Model A (level regression): per hour h, OLS ``lr_h = a_h + b_h *
   ln(v_1)`` over the rolling comparables, evaluated at the target's
   hour-1 print. The day's shape is driven by the hour-1 level: across
   comparables, corr(ln(v_1), peak log-ratio) is about -0.94 -- a small
   hour 1 means most of the audience has not unlocked yet, so the day
   rises steeply; a huge hour 1 means the audience is already there.
4. Model B (peak-ratio + shape template): the peak log-ratio over hours
   2..24 is regressed on ln(v_1); each hour's shape is the kernel-weighted
   mean of ``lr_h / peak`` over the rolling comparables nearest in ln(v_1)
   (Gaussian kernel, bandwidth 1.0 in log space). The median is
   ``peak_pred * shape_h``. This model follows steep-rise launches much
   better in the mid-day dip: on the practice set it cuts Spider-Man 2's
   loss from 0.219 to 0.180, but it overshoots on flatter launches, so it
   is only trusted where the dip lives.
5. Blend: for hours 8..15 (the mid-day trough and recovery) the whole
   quantile row is the log-space 50/50 average of the two models' rows
   (median and spread together); everywhere else it is model A alone.
   Each model's spread is its own per-hour residual standard deviation,
   mapped to the 5/25/50/75/95% levels with standard-normal quantiles.
6. Guards: the regression input is capped half a log unit past the largest
   comparable hour 1 (GTA VI's hour 1 will exceed past launches); if fewer
   than 6 rolling comparables are usable, all usable comparables are used;
   if none are usable, the house persistence baseline is emitted so the
   case stays valid.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = range(2, 25)
DIP_HOURS = set(range(8, 16))
BLEND = 0.5  # weight on model A inside the dip window; 1 - BLEND on B
CAP_EXTRA = 0.5  # log units past the largest comparable hour 1
SHAPE_BW = 1.0  # Gaussian kernel bandwidth (log space) for model B shape


def _parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _ols(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _curves(case):
    """Usable comparable launch curves: {v1, lr: {hour: log-ratio}}.

    A comparable counts when its release day has a positive hour-1 print
    and at least 20 of 24 hourly prints (missing hours are filled by
    log-linear interpolation, at most a 4-hour gap). The GTA VI category's
    own pre-release history is not a launch and is skipped.
    """
    out = []
    for comp in case.get("comparables", []):
        if "Grand Theft Auto VI category before release" in comp.get("name", ""):
            continue
        t0 = _parse(comp["day_start"])
        pts = {}
        for row in comp.get("hourly", []):
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            dh = (_parse(row["from"]) - t0).total_seconds() / 3600.0
            if 0 <= dh < 24:
                pts[int(dh) + 1] = float(viewers)
        if 1 not in pts or len(pts) < 20:
            continue
        have = sorted(pts)
        full = {}
        for h in range(1, 25):
            if h in pts:
                full[h] = pts[h]
                continue
            lo = max((x for x in have if x < h), default=None)
            hi = min((x for x in have if x > h), default=None)
            if lo is None or hi is None or hi - lo > 4:
                continue
            frac = (h - lo) / (hi - lo)
            full[h] = math.exp(
                math.log(pts[lo])
                + (math.log(pts[hi]) - math.log(pts[lo])) * frac
            )
        if len(full) < 23:
            continue
        v1 = full[1]
        out.append(
            {
                "v1": v1,
                "lr": {h: math.log(full[h] / v1) for h in full},
                "unlock": comp.get("unlock", ""),
            }
        )
    return out


def _rolling(curves):
    """Rolling-midnight launches only; fall back to all curves if too few."""
    sub = [
        c
        for c in curves
        if "rolling" in c["unlock"].lower()
        or "midnight local" in c["unlock"].lower()
    ]
    return sub if len(sub) >= 6 else curves


def _model_b_rows(curves, x0, p1):
    """Model B as full quantile rows (same normal-quantile spread as model
    A, but from model B's own residual sigma). Returns None when the peak
    regression is not usable (peak prediction too small to scale a shape
    by)."""
    peaks = {id(c): max(c["lr"][h] for h in HOURS) for c in curves}
    xs = [math.log(c["v1"]) for c in curves]
    ys = [peaks[id(c)] for c in curves]
    a, b = _ols(xs, ys)
    peak_pred = a + b * x0
    if peak_pred < 0.05:
        return None
    rows = {}
    for h in HOURS:
        wsum = ssum = 0.0
        for c in curves:
            w = math.exp(-0.5 * ((math.log(c["v1"]) - x0) / SHAPE_BW) ** 2)
            pk = peaks[id(c)]
            shape = c["lr"][h] / pk if abs(pk) > 0.05 else 0.0
            wsum += w * shape
            ssum += w
        shape_h = wsum / ssum if ssum > 0 else 0.0
        med = peak_pred * shape_h
        # Spread: peak-prediction error only, at each training curve's OWN
        # shape -- i.e. the shape is treated as known given the scale, and
        # only the peak level is uncertain. This makes dip-hour intervals
        # narrower than model A's; it is the measured-best choice on the
        # practice set (see HANDOFF-V2), not a claim that shapes are exact.
        resid = []
        for c in curves:
            pk = peaks[id(c)]
            own = c["lr"][h] / pk if abs(pk) > 0.05 else 0.0
            resid.append(c["lr"][h] - (a + b * math.log(c["v1"])) * own)
        sig = statistics.pstdev(resid) if len(resid) > 1 else 0.3
        rows[h] = [p1 * math.exp(med + z * sig) for z in Z]
    return rows


def _model_a_rows(curves, x0, p1):
    """Model A as full quantile rows."""
    rows = {}
    for h in HOURS:
        xs = [math.log(c["v1"]) for c in curves]
        ys = [c["lr"][h] for c in curves]
        a, b = _ols(xs, ys)
        med = a + b * x0
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sig = statistics.pstdev(resid) if len(resid) > 1 else 0.3
        rows[h] = [p1 * math.exp(med + z * sig) for z in Z]
    return rows


def _house(p1):
    """House persistence baseline: every hour stays at the hour-1 print,
    spread widening over the day. Used only when no comparable is usable."""
    rows = []
    for h in HOURS:
        s = 0.15 * math.sqrt(h - 1)
        rows.append(
            {"hour": h, "q": [round(p1 * math.exp(z * s), 2) for z in Z]}
        )
    return {"hours": rows}


def _forecast(case):
    known = case.get("known") or []
    p1 = float(known[0]["viewers"]) if known else 1.0
    if p1 < 1.0:
        p1 = 1.0
    curves = _rolling(_curves(case))
    if not curves:
        return _house(p1)
    xmax = max(math.log(c["v1"]) for c in curves)
    x0 = min(math.log(p1), xmax + CAP_EXTRA)
    rows_a = _model_a_rows(curves, x0, p1)
    rows_b = _model_b_rows(curves, x0, p1)
    rows = []
    for h in HOURS:
        if rows_b is not None and h in DIP_HOURS:
            # log-space 50/50 average of the two models' full quantile
            # rows: blends the median and the spread together
            qs = [
                math.exp(BLEND * math.log(rows_a[h][i])
                         + (1.0 - BLEND) * math.log(rows_b[h][i]))
                for i in range(5)
            ]
        else:
            qs = list(rows_a[h])
        qs = [round(min(max(q, 1.0), 1e9), 2) for q in qs]
        # keep the five quantiles non-decreasing after rounding/clamping
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = _forecast(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
