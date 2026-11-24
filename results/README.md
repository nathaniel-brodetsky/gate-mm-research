# Results

Outputs of registered tests, committed unchanged, so that every number in the hypothesis files and
reports can be traced to a file and a code commit. Reports in `docs/results/` are generated from these
files with `python -m mmlab.reporting.h001_h002_report`.

Evaluation code (`mmlab.evaluation`, `mmlab.microstructure`) is the same at 35449bd and b0881fb.

Reproducibility: rerunning a command on the same dataset gives the same integers and strings, but
floats can differ in their last bits, because polars adds in parallel and the order of additions
changes between runs. Two runs of the development evaluation differed by at most 4.1e-14 relative
(2026-11-24). Check a rerun with `python -m mmlab.evaluation.compare new.json results/<file>
--ignore role`, which accepts float differences up to 1e-12 absolute or 1e-9 relative.

| File | Command | Code commit |
|---|---|---|
| `h001_h002/development.json` | `python -m mmlab.evaluation.h001_h002 data/processed/U001 --period development` | 35449bd |
| `h001_h002/development_no_spyx.json` | the same with `--exclude SPYX_USDT` (D-001 amendment 2) | 35449bd |
| `h001_h002/validation.json` | `python -m mmlab.evaluation.h001_h002 data/processed/U001 --period validation` | b0881fb |
| `h001_h002/validation_no_spyx.json` | the same with `--exclude SPYX_USDT` | b0881fb |

`h003_explore/explore.csv` is the output of `python -m mmlab.microstructure.conditional data/processed/U001` (exploration for H-003, periods before 2026-11-01 only).

`h003/exploration_dry_run.json` is `python -m mmlab.evaluation.h003 data/processed/U001 --period exploration`: the H-003 test run on the data used to pick its threshold, to check the code before November; it decides nothing.

`lab/pools.json`, `lab/realism_empirical.json`, `lab/checks_empirical.json` and
`lab/compare_empirical.json` repeat the lab with the resampling generator (`python -m
mmlab.evaluation.lab pools`; the pools themselves stay in `data/interim/lab_pools/`).

`lab/daypools.json`, `lab/realism_days.json`, `lab/checks_days.json` and `lab/compare_days.json`
repeat it with the calendar-day generator (`python -m mmlab.evaluation.lab pools --kind days`; the
pools stay in `data/interim/lab_daypools/`). `lab/tails_days.json` is `python -m mmlab.evaluation.tails
results/lab/daypools.json --compare results/lab/compare_days.json --dataset data/processed/U001`: the
heaviest days of every pair. `lab/pooled.json`, `lab/pooled_empirical.json` and `lab/pooled_days.json`
are `python -m mmlab.evaluation.lab pooled` on `lab/calibration.json`, `lab/pools.json` and
`lab/daypools.json`: the pooled median rule of H-003 under several truths. `lab/levels.json` and
`lab/levels_days.json` are `python -m mmlab.evaluation.lab pooled --reps 4000` on `lab/calibration.json`
and `lab/daypools.json`: the same with the random-pairs null and the p-value of every period, for the
calibrated level.

`h004_explore/quote_pause.json` is `python -m mmlab.microstructure.quote_pause data/processed/U001`:
quote pauses on unusual prices or trading, exploration period, research pairs (stage 3, step 3a).

`lab/realism.json` is `python -m mmlab.evaluation.lab realism data/processed/U001` on that calibration:
real daily statistics ranked among synthetic periods.

`lab/compare.json` is `python -m mmlab.evaluation.lab compare` on the same calibration: the
candidate day tests of `mmlab.evaluation.day_tests` side by side.

`lab/calibration.json` and `lab/checks.json` are `python -m mmlab.evaluation.lab calibrate` on `data/processed/U001` (exploration period) and `python -m mmlab.evaluation.lab run` on that calibration: how the statistical methods behave on synthetic data like ours.

`ws_probe/probe-20261123T220212Z.txt` is the output of `python -m mmlab.data_ingestion.gate_ws_probe
summarize` on the 10-minute recording of 2026-11-23 (code at 58e6296); the raw recording stays in
`data/raw/ws_probe/`.

The dataset `data/processed/U001` is not committed. It was built on 2026-11-23 with
`python -m mmlab.market_data.build_dataset configs/universe_U001.json data/processed/U001 --start
2026-08-01 --end 2026-11-01`: 15 pairs before 35449bd and 9 with it (that commit changed retries and
memory use, not the output). SUKU_USDT rebuilt at 35449bd is identical to its first build (top of
book, trades and hourly checks).
