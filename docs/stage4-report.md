# Stage 4 — Report: passive market making on illiquid Gate USDT perpetuals

- **Written:** 2026-11-27, from the steps recorded in [stage4-plan.md](stage4-plan.md)
- **Data:** Gate's futures archive, 24 research contracts of U-002 ([D-003](decisions/D-003-perp-universe.md)),
  exploration period 2025-07-01 – 2025-10-31. The test period (November–December 2025) and the holdout
  contracts are closed and stay so.
- **Scope:** research on public data only. No account, no keys, no orders.
- **Status of the numbers:** exploration and simulator output, read by rules written before each run.
  Nothing here is a test of a registered hypothesis.

## Summary

1. On these 24 contracts a new maker who quotes one contract one tick inside the spread loses money
   before fees. Round trips: −12.9 bp gross, −18.9 bp net at VIP 0 (maker 0.02%, taker 0.05%), −16.7 bp
   with the fees shown in the contract objects (maker −0.01%, taker 0.075%); 0 of 24 contracts positive,
   with either queue model.
2. Netting the inventory does not save it. The maker's own fills are worth −6.7 bp against the mid 300 s
   later, before any exit: −8.7 bp after one maker fee at VIP 0 and −5.7 bp with the contract fields'
   rebate; 6 and 7 of 24 contracts positive.
3. Perpetuals are the same market as spot with a lower fee. The average trade's realized spread at 60 s
   is −1.4 bp (spot: about 0); its theta is −3.4 bp at a maker fee of 0.02% against −10.5 bp on spot at
   0.1%. A new maker's round trips lose half as much as on spot (gross −12.9 against −26 to −29 bp), but
   the sign is the same, and it is set before fees.
4. Positive markouts appear only where the spread is wide (half-spread from 10 bp, about 6% of the
   volume) and in the least active stratum (6 contracts, 5 of them delisted inside the window). As on
   spot, they are earned by liquidity that is already in the book: a maker who joins it does not get them
   back through the round trip.
5. A universe drawn from Gate's list of today's contracts would have been biased: 219 of the 587
   contracts that traded at the start of the window are delisted since, and the least active the most
   (46 of 87 with 100–1,000 trades a day).

## 1. Data

| Step | Result [archive] |
|---|---|
| Archive map, 2024-01 – 2026-10 (file sizes at 4 hours a day) | futures order books nearly complete in 2024–2025 (at least 96% of hours full in 23 of 24 months for BTC_USDT); 2026 degrades (41.1% in October); outages are venue-wide |
| Hour-by-hour coverage, GPS_USDT, July–December 2025 | 96.2–99.1% usable hours a month; one hour missing |
| Candidates (Tardis.dev's list, delisted included) | 587 USDT perpetuals listed before June 2025 and available on 2025-07-01; 219 delisted since; all of Gate's older contracts are in the list |
| U-002 (D-003) | four strata of June 2025 median daily trades in [100, 10,000); 24 research contracts, 7 of them delisted inside the exploration period |
| Dataset | all 24 built; the hours missing for delisted contracts end on their delisting dates |

## 2. The average trade

Medians over contracts of volume-weighted measures per contract [backtest]:

| | All 24 | Stratum 0 | Strata 1–3 |
|---|---|---|---|
| Quoted half-spread | 3.63 bp | 20.05 bp | 2.10 to 2.99 bp |
| RS(60 s) before fees | −1.44 bp | +12.69 bp | −3.59 to −0.45 bp |
| Theta at maker 0.02% | −3.44 bp (7 of 24 positive) | +10.69 bp | −5.59 to −2.45 bp |
| Theta at maker −0.01% | −0.44 bp | +13.69 bp | −2.59 to +0.55 bp |
| Exit at 60 s (crossing the spread) | −9.52 bp | −16.17 bp | −8.37 to −5.98 bp |

By quoted half-spread, RS(60 s) before fees: under 5 bp −4.5 bp (71% of the volume), 5–10 bp −0.2 bp
(12%), 10–20 bp +4.5 bp (5%), 20–40 bp +13.6 bp, 40–80 bp +44.0 bp, 80 bp and more +57.2 bp (together
about 1%). Without the last 14 days before each delisting, the realized spread and theta of all trades
move by less than 0.3 bp (the exit by 1.1 bp).

## 3. What a new maker gets

The stage 3 simulator: every 60 s an order of one contract on each side, one tick inside the spread,
kept 60 s; each filled piece closed by a passive order one tick inside the other side, kept 300 s, then
across the spread; 300 ms latency; queue position from the archive, fills only from archived trades
[simulator].

| | All 24 | By stratum |
|---|---|---|
| Round trips, gross | −12.9 bp | −15.9 to −11.7 bp |
| Round trips, net at VIP 0 | −18.9 bp (0 of 24 positive) | −21.6 to −18.0 bp |
| Round trips, net with the contract fields | −16.7 bp (0 of 24) | −18.8 to −15.8 bp |
| Closed passively; median holding | 29%; 95 s | |
| Optimistic queue model | within 0.1 bp of the above | |
| Own fills against the mid 60 / 300 s later | −6.8 / −6.7 bp | stratum 0 +8.3 / +7.3 bp; strata 1–3 −8.0 to −5.9 bp |
| Own fills at 300 s, net of one maker fee: VIP 0 / contract fields | −8.7 bp (6 of 24) / −5.7 bp (7 of 24) | stratum 0 +5.3 / +8.3 bp (5 of 6) |

## 4. Conclusions

- **Passive market making on illiquid Gate USDT perpetuals is not viable for a new maker at VIP 0**, by
  the rules written before each step (3b, 3c, 3c-2): no candidate for a registered batch. With the fees
  shown in the contract objects it is not viable either: round trips lose before fees, and own fills
  lose after a rebate of 1 bp.
- With stage 3, passive market making on Gate's illiquid markets is closed in both products at the fees
  available to an account without market maker status. The loss is adverse selection, not the fee:
  makers already in the book earn the spread; a maker who joins the queue is filled when the price is
  about to move against it.
- The least active contracts are the exception in the measures, not in the round trips: their own fills
  are positive (+7.3 bp at 300 s, 5 of 6), but the six are mostly contracts about to be delisted, too few
  for the rule (8), and crossing their spread costs 16 bp.

## 5. Limitations

- One quoting schedule: every 60 s, one tick inside, one contract, no signals, no inventory skew. The
  measures of the pair's own state (stage 3), BTC's moves (step 3c of stage 3) and topological features
  (step 3b of stage 3) found no signal worth more than about 3 bp.
- The own-fill markout leaves out inventory risk; it is a bound for a maker that nets its inventory, not
  a simulation of one.
- Sizes are in contracts; P&L in USDT needs the multiplier, which delisted contracts no longer have.
- Funding is left out (holding for minutes, charged every 8 hours).
- Which fees an account pays is not established from public data; the conclusions hold under both
  scenarios because the sign is set before fees (round trips) or survives a rebate (own fills).
- The candidate list is a third party's record (Tardis.dev); the archive misses 1–4% of hours.

## 6. Open decisions (owner)

1. **Closing passive market making on Gate.** Recommended: both products are closed by their rules.
2. **The least active perpetuals.** The positive own fills of stratum 0 could be tested on contracts not
   yet seen (the 3 holdout contracts of stratum 0 and the 6 eligible ones not drawn), under a new
   decision written before their data is read. The capacity is tiny (100–316 trades a day per contract)
   and most such contracts are delisted within months; low priority.
3. **Token-specific information.** BTC explains at most 2 bp of the adverse selection; the same token on
   other venues might explain more. That needs other venues' data, the owner's decision (sources, terms
   of use).
4. **Prerequisites of any trading stage**, unchanged: legal availability of Gate's products for the
   owner's residence and the account's actual fees.
5. **H-003** on spot is decided on November after 2026-12-01, as registered.

## 7. Reproducibility

Every number above comes from a committed results file listed in
[results/README.md](../results/README.md); the commit messages name the code commit. Modules added in
this stage:

| Module | Purpose |
|---|---|
| `mmlab.data_ingestion.gate_archive_map` | month-by-month map of the archive from file sizes |
| `mmlab.orderbook.gate_coverage` (usable hours) | hour-by-hour coverage |
| `mmlab.data_ingestion.perp_candidates`, `perp_activity` | candidates without survivorship, June 2025 activity |
| `mmlab.market_data.perp_universe` | U-002 under D-003 |
| `mmlab.market_data.build_dataset --biz futures_usdt` | futures top of book and trades |
| `mmlab.microstructure.perp_explore` | realized spread and theta by contract and half-spread |
| `mmlab.execution_simulator.perp_round_trip` | round trips and own-fill markouts of a new maker |
