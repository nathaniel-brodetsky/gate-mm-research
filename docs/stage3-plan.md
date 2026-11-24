# Stage 3 — Research program

Status: started 2026-11-24. Owner's brief: run every hypothesis and every method we have, record every
result. No orders, no keys: research on public data only.

## Rules that keep many tests honest

1. A method's results are trusted only after the method has been checked on synthetic data calibrated
   to ours, where the truth is known (the lab).
2. Hypotheses are registered in batches before their test data exists. A batch is tested at once, with
   the false discovery rate controlled across the batch, by a method the lab has shown to hold its size.
3. Exploration and test data never overlap. Exploration: 2026-08-01 – 2026-10-31, and November once
   it has been opened for H-001 – H-003. Next test window: December 2026 (deals published
   2027-01-01). The U-001 holdout pairs stay closed until a hypothesis is decided on the research pairs.
4. Everything is recorded: every variant including failures, the code commit and the data it ran on.

## Steps

1. Synthetic lab for the statistical methods *(done 2026-11-24, `results/lab/`)*. The generator
   (`mmlab.stochastic_models.markout_panel`) is calibrated per pair on the exploration period; the lab
   (`mmlab.evaluation.lab`) measures the coverage and size of the day bootstrap (30 and 61 days), the
   false discovery rate of Benjamini–Hochberg across 24 dependent pairs, the coverage of the bootstrap
   over pairs, and the power of the registered H-003 rule in November.
   Findings (synthetic months calibrated on the exploration period, 400 repetitions per check):
   - The registered day bootstrap (stationary blocks of mean 3 days, percentile interval) is too
     liberal. With a true theta of 0 it rejects at the 5% level in 9.5% of 30-day months for the
     median pair (7.9% over 61 days), and in up to 30.2% for DEEP_USDT. Its 95% interval covers the
     truth in 87.5% of 30-day months for the median pair, 52.5% for DEEP_USDT. On trades at quoted
     half-spreads of at least 80 bp: 11.7% rejections, 82.8% coverage.
   - Benjamini–Hochberg at q = 0.10 on those p-values, 24 pairs, no effect anywhere: at least one false
     discovery in 59.3% of 30-day months and 41.0% of 61-day periods. With half the pairs at +10 bp
     the false discovery rate is 11.6% (30 days) and 9.6% (61 days), power 66.5% and 73.5%.
   - The bootstrap over pairs of the median holds: coverage 93.0–94.8%, lower end above 0 under a
     true median of 0 in 2.4–3.9% of cases (nominal 2.5%).
   - The registered H-003 rule in a synthetic November accepts in 50.7% of months if every pair's
     August–October effect persists, 44.9% if it halves and 4.0% if it is absent.
2. If the lab finds a method miscalibrated, *(done 2026-11-24: `mmlab.evaluation.day_tests` holds five
   candidates; `python -m mmlab.evaluation.lab compare` runs them on the same synthetic months, with
   Benjamini–Hochberg and Benjamini–Yekutieli on top; `results/lab/compare.json`)* compare alternatives on the same synthetic data (for
   example a studentized or wider-block bootstrap, a t-test on daily values with a HAC variance,
   sign-flip tests over days, a correction of the false discovery rate for dependence) and register
   the one that holds its size for the next batch.
   Findings (400 synthetic months per pair and setting; size = rejections at 5% with theta 0):
   - No candidate holds 5% over 30 days on data like ours. The bootstrap-t is best: median size 7.6–7.7%
     (worst pair 21–23%), interval coverage 90.2–90.7%. Over 61 days: 6.5% (worst 16.3–16.5%),
     coverage 92.1–92.6%. On trades at quoted half-spreads of at least 80 bp (30 days): 8.5% with the
     Politis–White block (worst 11.3%), coverage 88.2%. The registered percentile bootstrap gives
     9.1% and 7.9%, the Newey–West t test 10.1% and 7.6%.
   - Studentizing costs some power: rejection at +10 bp, all trades, 82% against 90% over 30 days
     and 96.5% against 98.5% over 61 days.
   - Across 24 pairs with no effect anywhere, a false discovery after Benjamini–Hochberg on bootstrap-t
     p-values happens in 35–37% of 30-day months and 26–29% of 61-day periods; after
     Benjamini–Yekutieli in 19–20% and 11–12% (target 10%). With half the pairs at +10 bp,
     Benjamini–Yekutieli on bootstrap-t keeps the false discovery rate at 5.2–5.3% (30 days) and
     4.0–4.2% (61 days), with power 46–47% and 57–58%.
   - The calibration behind these numbers: trade-level noise has very heavy tails (fitted Student df
     4.0–4.7 on all trades, up to 9.4 on wide-spread trades); daily trade counts are strongly over-dispersed (gamma shape 0.03–3.4, except XT_USDT and
     SAITO_USDT); SUKU_USDT and DEEP_USDT hit the AR limit of 0.9, which produces the worst cases.
   - Proposed standard, to be fixed in decision D-002 once step 2b confirms the generator: per-pair
     tests use the bootstrap-t with the Politis–White block and Benjamini–Yekutieli across pairs, over
     test windows of at least 61 days; claims about all pairs together use the median with a bootstrap
     over pairs, which holds its size.
2b. Realism of the generator *(this step)*: `python -m mmlab.evaluation.lab realism` ranks five real
    statistics of each pair (daily theta spread, lag-1 correlation and tails, share of days with
    trades, variation of daily counts) among 1,000 synthetic periods. Ranks piling up near 0 or 1
    show what the lab gets wrong, and how far its conclusions can be trusted.
3. Synthetic market with known regime changes: do topological features (persistence of sliding-window
   embeddings of mid, spread and flow) detect them better and earlier than rolling volatility? Depth
   snapshots of the order book are built only if they do.
4. Feature library on the archive: quoted spread, imbalance, depth (if built), volatility, trade flow
   and its sign autocorrelation, time since the last trade, price impact per unit volume (Kyle's
   lambda), and topological features if step 3 passes.
5. Exploration on the exploration data, then one registered batch H-004… before 2027-01-01, tested on
   December.
6. A queue-aware event-driven simulator (fills from the archive's make/take deltas, never on a price
   touch) for whatever survives step 5, with results as a function of latency. Public data resolve the
   book to 20 ms at best (WebSocket) and 100 ms in the archive, so 100 and 400 ms can be simulated and
   1 and 10 ms cannot.
