# Stage 4 — USDT-M perpetuals (separate track)

Status: started 2026-11-25 by the owner's choice after the stage 3 report. Research on public data only;
spot and perpetuals are never mixed (own data, own universe, own hypotheses). Whether the owner may use
Gate's perpetuals at all is for the owner to check (stage 3 report, section 6) before any trading
stage.

## Why

Spot market making on illiquid Gate pairs lost in every setting tested ([stage 3 report](stage3-report.md)).
Perpetuals change one first-order parameter: the VIP 0 maker fee is 0.02% against 0.1% on spot
(https://www.gate.com/announcements/article/43029, read 2026-11-25). The spot round trips lost even before
fees, so a lower fee alone is not enough: the question is whether illiquid perpetuals have a different
microstructure (spreads, adverse selection, inventory risk).

## What is known (stage 1)

- Order book archive (`futures_usdt/orderbooks`) for October and November 2026 is not usable as a
  sample: GPS_USDT had updates in 41% of October's hours and 40% of complete hours in November 1–22;
  outages are global across contracts and fall in the most active weekday hours.
- Trade archives are complete where checked, also during order book outages.
- 1,037 USDT perpetuals were listed on 2026-11-23, 289 with 24 h quote volume of 20k–200k USDT.
- Futures sizes are in contracts; a contract's value needs its quanto multiplier.
- Fees: Gate's announcement gives 0.02% maker and 0.05% taker at VIP 0, but the public contract
  objects carry their own `maker_fee_rate` and `taker_fee_rate`, documented as "maker fee rate,
  negative values indicate rebates" (https://www.gate.com/docs/developers/apiv4/en/futures/); on
  2026-11-26 0G_USDT showed −0.0001 and 0.00075, and Gate's own 2025 example for BTC_USDT the same.
  Which rate an account pays is not established from public data; it decides the economics and is
  the owner's to check (e.g. the account's fee page) before any trading stage.

## Steps

1. Archive map *(done 2026-11-25, `results/perp_archive/futures_orderbooks.csv`,
   `spot_control.csv`)*: `python -m mmlab.data_ingestion.gate_archive_map` asked the archive for the
   size of each hourly order book file, without downloading it, at 01, 07, 13 and 19 UTC of every day
   from January 2024 to October 2026: futures BTC_USDT (outages are venue-wide), futures GPS_USDT
   (illiquid, audited in stage 1), and spot BTC_USDT as a control of the method. "Thin" is a file under
   10% of the market's median size, as a snapshot without updates looks. Sizes cannot show mid-hour
   gaps or broken chains.
   Findings [archive]:
   - Futures BTC_USDT: a file in every sampled hour of 2024 and 2025 but one (2025-06), and at least
     96.0% of the hours full in 23 of those 24 months (2024-05: 87.9%).
   - 2026 degrades: 11 and 6 sampled hours without a file in January and February; full hours 85.8% in
     April, 67.5% in June, 83.1% in July, 90.8% in September and 41.1% in October (stage 1 measured 41%
     of October's hours with updates for GPS_USDT by replaying the files: the two methods agree).
   - GPS_USDT has files from mid-January 2025 (its listing); from February to December 2025 at least
     97.3% of its sampled hours are full. Its missing and thin hours match BTC_USDT's month by month
     (2026-01: 113 files each; 2026-06: 39 thin each): outages are venue-wide.
   - Spot BTC_USDT: at least 97.5% full in every month but 2024-02 (87.9%).
   - The candidate window for perpetuals is 2025: February–December. Step 2 checks the six most recent
     months, July–December 2025, before anything is built on them.
2. Hour-by-hour coverage of the window.
   2a. *(done 2026-11-26, `results/perp_archive/coverage/GPS_USDT-2025-*.csv`)* `python -m
       mmlab.orderbook.gate_coverage --quiet` downloaded and replayed every hourly order book file of
       futures GPS_USDT from 2025-07-01 to 2026-01-01, month by month. An hour is usable when its file
       is clean, its last row comes at least 50 minutes after its snapshot and the next hour chains to
       it. Rule, written before the run: a month is usable when at least 95% of its hours are; the
       window goes on if at least three consecutive months are usable.
       Findings [archive]: every month is usable. Usable hours: July 99.1%, August 98.4%, September
       98.2%, October 96.2%, November 97.5%, December 99.1%; one hour missing (July); 6 to 28 chain
       breaks and 1 to 21 truncated hours a month, most in October. The window July–December 2025 goes
       on to step 2b. The update ids lost at the breaks (93,087 to 1,457,842 a month) are not read as
       GPS_USDT updates: whether the ids count per contract is not established.
   2b. A universe U-002 of illiquid perpetuals for the window, registered before its order book data
       is examined (D-003). Gate's API lists only the contracts trading today; a universe chosen from
       it would leave out contracts delisted since (survivorship bias).
       2b-1. *(done 2026-11-26, `results/perp_universe/candidates.csv`)* `python -m
             mmlab.data_ingestion.perp_candidates` read Tardis.dev's free list of the symbols it
             recorded on Gate futures, with the dates they were available (delisted ones included), and
             Gate's list of today's contracts (pages of 100; a page of 1,000 is refused); both answers
             are kept as raw JSON. Candidates: USDT perpetuals listed before 2025-06-01 and still
             available on 2025-07-01, with the size of their June 2025 trades file (headers only).
             Findings: Tardis lists 1,802 symbols, Gate 1,042 USDT contracts today. 587 candidates, of
             which 219 (37%) are no longer listed: a universe drawn from today's list would have lost
             them. Every Gate contract launched before June 2025 is in Tardis's list. 570 candidates
             have a June 2025 trades file (2.01 GB together: 12 under 0.1 MB, 253 of 0.1–1 MB, 265 of
             1–10 MB, 38 of 10–100 MB, 2 above), 17 have none.
       2b-2. *(done 2026-11-26, `results/perp_universe/activity_2025-06.csv`)* `python -m
             mmlab.data_ingestion.perp_activity` read the June 2025 trades of the 570: trades per
             calendar day, active days, median daily trades and, where Gate still gives the
             multiplier, median daily quote volume. The trade count needs no multiplier, so it is the
             measure a rule can apply to delisted contracts too.
             Findings: 569 were active on at least 28 days. Median daily trades: 10–100: 2 contracts
             (2 delisted since), 100–1,000: 87 (46), 1,000–10,000: 392 (151), above: 88 (18). The
             least active contracts are the most often delisted.
       2b-3. *(done 2026-11-26, `configs/universe_U002.json`)* [D-003](decisions/D-003-perp-universe.md):
             four strata of median daily trades in [100, 10,000), 6 research and 3 holdout contracts
             each by a salted hash, delisted contracts kept; exploration July–October 2025, held-out
             test November–December 2025. `python -m mmlab.market_data.perp_universe` wrote the
             selection after the rule was pushed.
             Findings: eligible per stratum 15, 71, 228 and 162 (8, 38, 102 and 49 of them delisted
             since); excluded: 88 above 10,000 trades a day, 17 without a June file, 4 stable or
             pegged, 1 active on fewer than 28 days, 1 below 100. 16 of the 36 selected contracts are
             delisted since: all six research contracts of the least active stratum (PYR, PEIPEI,
             AMP, BTT, ALU, NAVX), two of the second (XNO, REQ), none of the third, two of the fourth
             (PUFFER, RDAC); 6 of the 12 holdout contracts. Flag (D-001 policy, not removed): holdout
             USTC_USDT is TerraClassicUSD, a former algorithmic stablecoin that lost its peg in 2022.
3. Data and exploration on the research contracts.
   3a. *(this step)* `python -m mmlab.market_data.build_dataset --biz futures_usdt` reads the hourly
       futures order books and the monthly futures `trades` files: top of book and trades per contract,
       sizes in contracts, the same validity rules as for spot. Periods reaching 2025-11-01 are
       refused without --open-time-holdout (D-003), and the holdout contracts stay locked. Built for
       the 24 research contracts over the exploration period, July–October 2025 (seven of them are
       delisted inside it: D-003, Amendment 1). Measures in bp
       (markouts, realized spread) need no contract multiplier; order sizes in USDT do, and Gate gives
       it only for contracts still listed.
   3b. *(done 2026-11-27, `results/perp_explore/markouts.json`)* `python -m
       mmlab.microstructure.perp_explore` on the research contracts over the
       exploration period: per contract, realized spread at 10, 60 and 300 s (mid), at 60 s against the
       microprice and the far side (exit), the quoted half-spread, and theta under the two fee
       scenarios (maker 0.02%; the contract field's −0.01%); by bucket of quoted half-spread; all
       trades and without each delisted contract's last 14 days. Medians over contracts with at least
       30 trades on 5 days, overall and by stratum. How it will be read, written before the run: a
       subset (all trades or a bucket of half-spread) is a candidate for a registered batch only if
       its median theta at maker 0.02% is positive over at least 8 counted contracts; the round trips
       of a new maker (step 3c) then check it before anything is registered. On spot the same median
       was −10.45 bp at a 0.1% fee (stage 3).
       Findings (medians over contracts) [backtest]:
       - All trades, 24 contracts: quoted half-spread 3.63 bp; RS at 10/60/300 s −0.97/−1.44/−2.36 bp;
         exit at 60 s −9.52 bp; theta −3.44 bp at maker 0.02% (7 of 24 positive), −0.44 bp at −0.01%.
         The average trade's realized spread sits where spot's did (+0.37 and −1.09 bp in H-001);
         theta is closer to 0 only through the lower fee.
       - Stratum 0 (6 contracts, 5 delisted inside the window): half-spread 20.05 bp, RS60 +12.69 bp,
         theta +10.69 bp (5 of 6), exit −16.17 bp. Strata 1–3: theta −5.59 to −2.45 bp.
       - By quoted half-spread (RS60 before fees; median share of volume): under 5 bp −4.45 (71.4%),
         5–10 −0.16 (12.2%), 10–20 +4.46 (5.1%), 20–40 +13.64 (0.9%), 40–80 +44.04 (0.1%), 80 and more
         +57.18 (0.1%, 20 contracts).
       - Without the last 14 days before a delisting: within 0.3 bp overall.
       - By the rule, the buckets from 10 bp are candidates (median theta at 0.02% positive over 20–24
         contracts), about 6% of the volume; strata are not subsets under the rule. As on spot (H-003),
         a positive markout at a wide spread is measured against a mid nobody trades at; step 3c decides.
   3c. *(done 2026-11-27, `results/perp_explore/round_trips_{pessimistic,optimistic}.json`)*
       `python -m mmlab.execution_simulator.perp_round_trip`: the stage 3 round trips
       of a new maker on the research contracts over the exploration period. Every 60 s an order of
       one contract (the smallest Gate accepts) on each side, one tick inside the spread, kept 60 s;
       each filled piece closed by a passive order one tick inside the other side, kept 300 s, then
       across the spread. Net bp of entry value under the two fee scenarios (VIP 0: maker 0.02%, taker
       0.05%; contract fields: maker −0.01%, taker 0.075%), for the pessimistic and the optimistic
       queue model, all round trips and without the last 14 days before a delisting. Funding is left
       out (minutes of holding, charged every 8 hours). How it will be read, written before the run:
       a new maker's quoting is a candidate for a registered batch only if the median net at VIP 0
       is positive over at least 8 counted contracts with the pessimistic queue; the optimistic queue
       is a bound, not a basis.
       Findings (medians over contracts) [simulator]:
       - Pessimistic queue, 24 contracts: gross −12.86 bp, net −18.90 bp at VIP 0 and −16.73 bp
         under the contract fields; 0 of 24 positive under either; 29% of the size closed passively,
         median holding 95 s. Every stratum is negative (gross −15.87 to −11.66 bp, net at VIP 0
         −21.60 to −18.03 bp).
       - Optimistic queue: within 0.1 bp of the pessimistic one. Without the run-up to delistings:
         gross −12.44 bp, net −18.90 bp.
       - By the rule there is no candidate. Gross is negative before any fee, so no fee tier changes the
         sign. On spot the same round trips gave gross −26 to −29 bp and net −46 to −49 bp.
       - Limitation: every filled piece is closed on its own. A maker that nets its buys against its
         sells crosses the spread less; step 3c-2 measures that bound.
   3c-2. *(this step)* `python -m mmlab.execution_simulator.perp_round_trip --measure fills`: the
       markout of the same maker's own entry fills against the mid 60 and 300 s later, before any
       exit: what a maker that nets its inventory would make per unit of fill value, before inventory
       risk; net is the 300-s markout minus one maker fee. How it will be read, written before the run:
       if the median contract's net at VIP 0 is not positive (pessimistic queue, at least 8 counted
       contracts), passive market making on these perpetuals is closed at VIP 0 as it was on spot; the
       contract-field fee is reported beside it.
   3d. A batch registered under D-002 before November–December 2025 is read, or the perpetuals track
       is closed like spot.
4. If the window fails, the options are the owner's: a WebSocket recorder (forward data, months to
   collect) or a paid history (Tardis.dev lists Gate futures since 2020-07-01; prices not checked).
