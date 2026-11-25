# Lead-lag from the wider market (stage 3, step 3c)

Status: closed 2026-11-25 by its rule (findings at the end).

## Why

The average maker on the illiquid pairs loses to adverse selection: round trips of a new maker lose
26–29 bp before fees (stage 3 report, section 3), and the pair's own state before a trade moves theta
by 3 bp at most (step 4a). On an illiquid pair the informed taker is often someone who has seen the
price move elsewhere. The cheapest test of that uses the wider market on the same venue: spot BTC_USDT
on Gate, whose trades archive is complete. No other venue's data is used.

## Data and exploration

- `python -m mmlab.market_data.reference_grid BTC_USDT data/processed/reference` reads the monthly
  deals of August–October 2026 and writes a one-second as-of grid: the price of the last trade before
  each second. A missing month is refused rather than bridged.
- `python -m mmlab.microstructure.leadlag_explore` takes every trade of the research pairs in the
  exploration period (2026-08-01 – 2026-10-31) and the log change of BTC_USDT
  - as of 1 s before the trade, over the last 1, 5, 10, 30 and 60 s (tradable);
  - ex post, from just before the trade to about 60 s after it (not tradable: a ceiling for any
    signal about BTC),

  signed so that positive means BTC moved the taker's way. Per bucket of the move, theta (volume-
  weighted RS(60 s) minus the maker fee) as in step 4a, for all trades and for the H-003 subset; per
  feature, the effect of not quoting while the signed move exceeds 3 bp: kept share of the volume and
  change in theta, median over pairs.

## Rule (written before the run)

1. **Ceiling.** If not quoting when BTC moves more than 3 bp the taker's way in the minute after the
   trade (known only ex post) gains less than 3 bp of theta for the median pair on all trades, lead-lag
   from BTC is closed: even perfect knowledge of BTC's next minute would not matter.
2. **Candidate.** Otherwise, an as-of lag whose rule gains at least 3 bp for the median pair while
   keeping at least half of the volume, on all trades or on the H-003 subset, becomes a candidate for
   the batch H-004 (registered before its test data exists, tested on December under D-002). Ten
   as-of cells are screened: exploration, not inference.
3. Neither outcome makes spot market making viable by itself: on all trades theta is −10.45 bp for the
   median pair. Three bp is the largest effect of the pair's own state and more than the perpetual
   maker fee (0.02%), the scale at which a signal matters on stage 4.

If the ceiling is open but no as-of lag qualifies, BTC moves too fast for a maker 1 s behind; other
references (ETH_USDT, the same token on a liquid venue) would be the owner's decision.

## Findings (2026-11-25, `results/h004_explore/leadlag_btc.json`)

Exploration period, research pairs; effect of not quoting while BTC's signed move exceeds 3 bp,
medians over pairs [archive].

| feature | all trades (24 pairs, theta −10.45 bp): kept, gain | H-003 subset (20 pairs, theta +26.55 bp): kept, gain |
|---|---|---|
| ref_1s | 0.996, +0.03 bp (16/24 improved) | 0.997, +0.05 bp (12/20) |
| ref_5s | 0.977, +0.04 bp (13/24) | 0.981, +0.58 bp (13/20) |
| ref_10s | 0.952, +0.04 bp (14/24) | 0.964, +0.36 bp (11/20) |
| ref_30s | 0.873, +0.24 bp (14/24) | 0.894, −0.32 bp (8/20) |
| ref_60s | 0.807, +0.17 bp (14/24) | 0.834, −1.29 bp (7/20) |
| ex post, next 60 s | 0.813, +2.02 bp (20/24) | 0.830, +5.71 bp (16/20) |

- **The rule closes lead-lag from BTC.** Even knowing BTC's next minute, not quoting the 19% of the
  volume when BTC then moved the taker's way gains 2.02 bp for the median pair on all trades, below
  the 3 bp ceiling. What BTC did before the trade is worth at most 0.24 bp; no as-of lag qualifies on
  either subset.
- The adverse selection on these pairs does not come from BTC's moves; more likely from each token's
  own flow, news or its price on other venues. Testing the last needs other venues' data: the owner's
  decision.
- On wide spreads the ex-post ceiling is larger (+5.71 bp), but signals known before the trade give
  at most +0.58 bp there, and the longer lags lower theta.
