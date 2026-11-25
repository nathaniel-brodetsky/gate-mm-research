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
   2a. *(this step)* `python -m mmlab.orderbook.gate_coverage --quiet` downloads and replays every
       hourly order book file of futures GPS_USDT from 2025-07-01 to 2026-01-01, month by month. An hour
       is usable when its file is clean, its last row comes at least 50 minutes after its snapshot and
       the next hour chains to it. Rule, written before the run: a month is usable when at least 95% of
       its hours are; the window goes on to step 2b if at least three consecutive months are usable.
   2b. A universe U-002 of illiquid perpetuals, registered before its order book data is examined
       (D-003). Gate's API lists only the contracts trading today, so a universe chosen from it in 2025
       would leave out contracts delisted since: survivorship bias, which D-003 must address (e.g.
       delisted contracts found in the trades archive or in Gate's delisting announcements).
3. The dataset builder adapted to the futures formats (signed sizes, contract multipliers, funding),
   then the stage 3 tools: markouts, the lab and D-002 levels, the queue-aware fills and round trips
   with the perpetual fees (maker 0.02%, taker 0.05% at VIP 0) and funding.
4. If the window fails, the options are the owner's: a WebSocket recorder (forward data, months to
   collect) or a paid history (Tardis.dev lists Gate futures since 2020-07-01; prices not checked).
