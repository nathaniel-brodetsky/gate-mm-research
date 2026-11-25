# Stage 3 — Report: passive market making on illiquid Gate spot pairs

- **Written:** 2026-11-25, from the steps recorded in [stage3-plan.md](stage3-plan.md)
- **Data:** Gate global spot archive, 24 research pairs of U-001 ([D-001](decisions/D-001-research-universe.md)),
  exploration period 2026-08-01 – 2026-10-31. November is still closed (H-003 is decided on it after
  2026-12-01); the holdout pairs are closed.
- **Scope:** research on public data only. No account, no keys, no orders.
- **Status of the numbers:** everything below is exploration or lab output. Nothing here is a test of
  a registered hypothesis except where H-001 to H-003 are named.

## Summary

1. On these 24 pairs, at Gate's VIP 0 fees, a new maker who quotes at or one tick inside the best price
   loses money in every setting tested, before fees as well as after. Round trips lose 46–49 bp net
   (26–29 bp before fees) on all trades and 48–82 bp net (28–62 bp before fees) on wide spreads; no pair
   is positive on all trades in any setting.
2. The positive average realized spread of wide-spread trades (the subject of H-003, +35 bp in
   exploration) belongs to liquidity that was already in the book. A maker who joins the back of the
   queue gets fills 40–49 bp worse than the average trade; one who steps one tick in front gets +4 to
   +11 bp against the mid 60 s later, and loses it while getting out.
3. The result does not depend on the details of the simulation: both bounds of the queue model agree
   within 0.6 bp, and latencies from 100 to 1,000 ms within 0.3 bp on all orders.
4. The loss of the average trade is mostly the fee: the liquidity that trades captures about 0 bp of
   realized spread at 60 s, and the maker fee is 10 bp. Days of extreme prices of single tokens make the
   result risky and per-pair tests unreliable, but are not why it is negative; neither pauses on unusual
   activity nor the as-of state of the book and the flow change it by more than about 3 bp.
5. Statistics: per-pair day tests reject a true 0 in 10–16% of windows of 30–61 days for the median pair
   (up to 52%), because losses come in rare heavy days. Decisions therefore use the median over pairs at
   a level calibrated in a synthetic lab rebuilt from real days ([D-002](decisions/D-002-testing-standard.md)).

## 1. Testing standard

The lab (`mmlab.evaluation.lab`) checks statistical rules on synthetic periods whose truth is known.
Three generators were built; only the one that rebuilds periods from real calendar days, in blocks,
shifted to a known theta, reproduces the real daily statistics (median ranks 0.48–0.68, one miss in 24
pairs and five statistics). Under it:

| Rule (true theta 0, nominal level) | 30 days | 61 days |
|---|---|---|
| Registered day bootstrap per pair, median pair (5%) | 15.8% (worst 51.5%) | 13.4% |
| Bootstrap-t with Politis–White block per pair (5%) | 13.4% | 10.4% |
| Benjamini–Hochberg across 24 pairs, any false discovery | 93.5% | 88.2% |
| Median over pairs, every pair at 0 (2.5%) | 7.1% | 5.0% |

The size of the per-pair tests follows the skewness of each pair's days (Spearman −0.82 across pairs),
not the length of slow regimes. D-002 fixes the standard for H-004 onwards: decide on the median over
pairs (bootstrap over pairs), at a level calibrated so that every null truth (every pair at 0; real
spread around a median of 0, fixed or random pairs) is accepted in at most 2.5% of periods. For the
realized spread at 60 s minus the fee: 0.5% (30 days) and 1% (61 days) on all trades, 2.5% and 1.5% on
half-spreads of at least 80 bp. H-003's registered level, 2.5% on wide spreads over 30 days, is the
calibrated one.

## 2. Where the average trade's loss comes from

- **Heavy days** (step 2e). On all trades the days are skewed to the loss side in 18 of 24 pairs. The
  worst day of 18 pairs had a price range at least three times the pair's median day (up to 478%
  against 11%), 23 of the 24 worst days had no data problems, and the worst dates coincide across pairs
  no more than chance would make them: they are events of single tokens.
- **Quote pauses** (step 3a). 24 rules that stop quoting while the price range or the number of trades
  over 1–30 minutes is 2–10 times its usual level move the median pair's theta by −0.5 to +0.7 bp. Even
  dropping each pair's three worst days after the fact adds only 3.4 bp to a median of −10.5 bp.
- **State before the trade** (step 4a). Top-of-book imbalance, signed taker flow, time since the last
  trade and the relative price range move theta by about 3 bp at most; every bucket on all trades lies
  between −13.7 and −6.9 bp.
- **Fees.** The traded liquidity's median realized spread at 60 s is about 0 bp before the fee (H-001:
  +0.37 bp in development, −1.09 bp in validation), against a VIP 0 maker fee of 0.1%. Gate's negative
  maker fees need at least 0.15% of the platform's monthly spot maker volume or more than USD 20 million
  traded in 30 days; the 24 pairs together traded at most about USD 0.9 million a day in July.

## 3. What a new maker gets

The execution simulator (`mmlab.execution_simulator`) fills virtual orders only from archived trades,
never on a price touch: an order joins at the end of the 100 ms batch it reaches the exchange in,
behind everything displayed at its price; trades at its price consume the queue first; a trade through
its price fills it up to that trade's size. A one-pass engine is checked against an independent
per-order reference on random books.

Orders of 10 USDT every 60 s on both sides, 300 ms latency, pessimistic queue; medians over pairs with at
least 30 fills (or round trips) on 5 days; theta against the mid 60 s later, after the 0.1% maker fee:

| | All trades | Half-spread ≥ 80 bp |
|---|---|---|
| Average trade (for comparison) | −10.5 bp | +34.8 bp |
| Back of the queue, order kept 300 s | −27.4 bp | −24.1 bp |
| Back of the queue, 100 USDT | −47 to −49 bp | −56 to −86 bp |
| One tick inside, kept 300 s / 60 s | −23.2 / −19.8 bp | +4.1 / +10.8 bp |
| Closing across the spread 60 s after a fill | −46 to −64 bp | −124 to −139 bp |
| Round trip, closing passively first (net) | −46.3 to −48.7 bp | −47.8 to −82.4 bp |
| Round trip, before fees | −26.3 to −28.7 bp | −27.8 to −62.3 bp |
| Pairs with a positive round trip | 0 of 24 | 1–3 of 14–15 |

Passive exits kept 30 minutes close about half of the size, after 19–30 minutes; the price keeps
moving against fresh inventory meanwhile. Quoting all 24 pairs this way would have lost 192–580 USDT a
day at 10 USDT per order.

## 4. Conclusions

- Passive market making on illiquid Gate spot pairs is not viable for a participant at VIP 0 fees with
  the strategies tested: joining the queue, stepping one tick inside the spread, closing passively with
  a fallback across the spread. The conclusion holds across the queue model, latency, order lifetime
  and size.
- H-003 will be read accordingly: an acceptance on 2026-12-01 will say that the liquidity in place at
  wide spreads earned its fee on average, not that a new maker can earn it (recorded in the hypothesis
  file before November was opened).
- Speed is not the missing ingredient at the horizons studied: 100 ms instead of 1,000 ms changes the
  result by less than 0.3 bp on all orders.

## 5. Limitations

- Three months, 24 pairs of one universe, one fee tier. Events heavier or more frequent than those of
  August–October are not in the lab.
- The simulator sees only the top of the book and treats every position on its own (no netting, no
  inventory-dependent quotes, no signals). A strategy that skews its quotes by inventory and nets
  positions could do better; it would have to close a gap of 26–62 bp before fees.
- The model of the virtual order's effect on others is absent: its presence changes nothing in the
  archive. At 10 USDT this is small; at 100 USDT fills were already much worse.

## 6. Open decisions (owner)

1. **Direction.** Candidates, with the evidence above: USDT-M perpetuals as a separate track (VIP 0
   maker fee 0.02% against 0.1% on spot, https://www.gate.com/announcements/article/43029, read
   2026-11-25; but spot round trips lose even before fees, so the fee alone does not decide);
   inventory-aware quoting with signals on spot (low prior); research on the taker side (most of a move's
   impact comes within 100 ms, H-002; low prior at 264 ms round trip to the API); or no further market
   making research on this venue.
2. **Legal availability.** Secondary sources report that Gate's global platform stopped registering new
   EU/EEA users in March 2025 and serves them through Gate Europe, authorised under MiCA in Malta, which
   does not cover derivatives. Whether the owner can use Gate's global spot or perpetuals has to be
   checked in Gate's user agreement (section 2.5) and with the owner's residence before any trading
   stage; research on public data is not affected.
3. **H-003** is decided on November after the archive publishes it (2026-12-01, about 05:00 UTC), as
   registered ([stage2-plan.md](stage2-plan.md), step 10).

## 7. Reproducibility

Every number above comes from a committed results file; [results/README.md](../results/README.md) lists
each file with the command that produced it, and the commit messages name the code commit. The modules
added in this stage:

| Module | Purpose |
|---|---|
| `mmlab.stochastic_models.markout_panel` | generators of synthetic markout panels (parametric, trade resampling, calendar days) |
| `mmlab.evaluation.lab`, `day_tests`, `tails` | the synthetic lab, candidate day tests, heaviest days |
| `mmlab.feature_engineering.asof_activity`, `trade_state` | as-of signals on a one-second grid and per trade |
| `mmlab.microstructure.quote_pause`, `state_explore` | explorations of pauses and of the state before a trade |
| `mmlab.execution_simulator.queue`, `fill_explore`, `round_trip` | queue-aware fills, fill markouts, round trips |

## 8. Addenda 2026-11-25

**Topological features.** Step 3b of the plan, skipped before this report, was run in the synthetic lab
([topology-plan.md](topology-plan.md), `results/topology/regime_lab.json`): persistence diagrams of
trailing price windows (swings and Rips loops of the delay embedding) against seven simple statistics,
on regimes with known switches. By the rule written before the run they fail: for reverting against
continuing moves they add nothing (AUC gain −0.001 and +0.002 at 60 and 300 s), for jumps +0.005 and
+0.012, below the 0.02 required, and they see no switch sooner; in the cycle control they add +0.014
and +0.040, so they work where there is a loop to see. None of the conclusions above changes.
Modules: `mmlab.feature_engineering.window_stats`, `mmlab.topology.persistence`, `regime_lab`.

**Lead-lag from BTC** (step 3c, [leadlag-plan.md](leadlag-plan.md),
`results/h004_explore/leadlag_btc.json`): not quoting while spot BTC_USDT has moved more than 3 bp the
taker's way gains at most +0.24 bp for the median pair; even knowing BTC's next minute gains +2.02 bp,
below the 3 bp the rule required. The adverse selection on these pairs does not come from BTC's moves.
Modules: `mmlab.market_data.reference_grid`, `mmlab.microstructure.leadlag_explore`.
