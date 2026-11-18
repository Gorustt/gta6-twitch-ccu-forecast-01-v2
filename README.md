[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Gorustt/gta6-twitch-ccu-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Gorustt/gta6-twitch-ccu-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# Forecast GTA VI's launch-day Twitch viewers hour by hour

Best-version entry for the [make.fun bounty gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) (agent-fun/program/v1,
Python 3.13, standard library only, offline, deterministic). It replaces the
baseline entry (rolling-midnight OLS, LOO mean 0.1615) with a two-model
blend that handles the mid-day dip of steep-rise launches.

## Method

Each comparable past launch contributes its release-day hourly curve as
log-ratios to its own hour-1 print, `lr_h = ln(v_h / v_1)`, h = 2..24.
Reference class: launches whose unlock note says the launch rolled out at
midnight local time ("rolling" / "midnight local"); simultaneous global
unlocks are excluded.

- **Model A — level regression.** Per hour, OLS `lr_h = a_h + b_h *
  ln(v_1)` over the rolling comparables, evaluated at the target's hour-1
  print. The day's shape is driven by the hour-1 level (across
  comparables, corr(ln(v_1), peak log-ratio) is about -0.94): a small hour
  1 means most of the audience has not unlocked yet, so the day rises
  steeply; a huge hour 1 means the audience is already there.
- **Model B — peak ratio + shape template.** The peak log-ratio over hours
  2..24 is regressed on ln(v_1); each hour's shape is the kernel-weighted
  mean of `lr_h / peak` over the rolling comparables nearest in ln(v_1)
  (Gaussian kernel, bandwidth 1.0 in log space). Median: `peak_pred *
  shape_h`. This model follows steep-rise launches in the mid-day dip far
  better than model A (Spider-Man 2's loss 0.219 -> 0.180 on its own), but
  it overshoots on flatter launches, so it is only trusted where the dip
  lives.
- **Blend.** For hours 8..15 (the mid-day trough and recovery) the whole
  quantile row is the log-space 50/50 average of the two models' rows;
  everywhere else the forecast is model A alone. Each model's spread is
  its own per-hour residual standard deviation, mapped to the 5/25/50/75/
  95% levels with standard-normal quantiles. Model B's spread counts
  peak-prediction error only (each training curve at its own shape), which
  narrows dip-hour intervals; that is the measured-best choice on the
  practice set, not a claim that shapes are exact.
- **Guards.** The regression input is capped half a log unit past the
  largest comparable hour 1 (GTA VI's hour 1 will exceed past launches);
  fewer than 6 rolling comparables falls back to all usable comparables;
  no usable comparables emits the house persistence baseline so the case
  stays valid.

## Local validation (leave-one-launch-out, 4 practice cases)

Answers re-fetched from the same SullyGnome source the kit documents; every
case's hour-1 print matches exactly. House baseline scores 0.387097 on the
same answers (RULES.txt says ~0.387), validating the scorer. Variants were
tested on the identical protocol (quadratic log-log 0.1666, two-regime OLS
0.1643, peak+shape alone 0.1685, local-linear 0.174-0.191, tier-clean OLS
0.1648, empirical-quantile spread 0.1551, full-day ensemble 0.1564); the
dip-window blend below was the best measured mean.

| case | loss | typical miss | 90% coverage |
| ---- | ---- | ------------ | ------------ |
| p00 Marvel's Wolverine | 0.1647 | 29.2% | 0.96 |
| p01 Red Dead Redemption 2 | 0.1213 | 19.2% | 0.96 |
| p02 God of War Ragnarök | 0.1126 | 15.6% | 1.00 |
| p03 Marvel's Spider-Man 2 | 0.1986 | 33.9% | 0.78 |
| **mean** | **0.1493** | | |

Mean 0.1493 beats the baseline entry's 0.1615 and the house baseline
(0.3871), and improves the worst case (Spider-Man 2: loss 0.2195 -> 0.1986,
miss 39.5% -> 33.9%, coverage unchanged at 0.78) with no big regression
elsewhere (largest move: RDR2 0.1188 -> 0.1213). Honest caveats: the mean
is measured on only 4 practice cases, so differences under ~0.01 are
noise; the practice board is dev-only and the final is a single real case
(GTA VI itself), so this is not a promised payout. The two large hour-1
launches score 0.121 / 0.113, and GTA VI's hour 1 will be in the large
regime.

Determinism: two runs per case produce byte-identical outputs. Runtime:
~75-95 ms per case (limit 8 s). Output files ~1.5 KB (limit 65,536 B).
A synthetic final-like case (hour-1 = 1,500,000, beyond the largest
comparable) runs valid with a sensible decaying median (h24/h1 = 0.54).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case.
