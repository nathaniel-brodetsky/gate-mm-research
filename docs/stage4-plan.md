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

1. Archive map *(this step)*: `python -m mmlab.data_ingestion.gate_archive_map` asks the archive for
   the size of each hourly order book file without downloading it, for four hours of every day
   (01, 07, 13 and 19 UTC) from January 2024 to October 2026: futures BTC_USDT (outages are
   venue-wide), futures GPS_USDT (illiquid, audited in stage 1), and spot BTC_USDT as a control of the
   method. Per market and month: hours with a file and "thin" files (under 10% of the market's
   median size, as a snapshot without updates looks). Sizes cannot show mid-hour gaps or broken
   chains.
2. If a period of at least three consecutive months looks complete: hour-by-hour coverage with
   `mmlab.orderbook.gate_coverage` on a sample of illiquid contracts; the dataset builder adapted to
   the futures formats (signed sizes, contract multipliers, funding); a universe U-002 of illiquid
   perpetuals listed through the period, registered before its data is examined (D-003).
3. Then the stage 3 tools on that data: markouts, the lab and D-002 levels, the queue-aware fills and
   round trips with the perpetual fees and funding.
4. If no usable history exists, the options are the owner's: a WebSocket recorder (forward data,
   months to collect) or a paid history (Tardis.dev lists Gate futures since 2020-07-01; prices not
   checked).
