# Stage 2 — Research dataset and first microstructure facts

Status: started 2026-11-23. No trading strategy in this stage.

1. Pre-register the research universe (D-001) and hypotheses H-001, H-002. *(done)*
2. Run the selection; commit `configs/universe_U001.json`. *(done)*
3. Asset metadata for research pairs; top of book per batch with availability time and validity.
   *(done; SPYX_USDT flagged, D-001 amendment 2)*
4. Dataset builder: download and validate archives for the research pairs, store top of book and
   trades in Parquet with gap sentinels. *(done: 24 pairs, 2026-08-01 – 2026-10-31, 2208 hours each,
   none missing; see [results/README.md](../results/README.md))*
5. Descriptive facts and the tests of H-001 and H-002 on the development period, then validation.
   *(done: H-001 rejected, H-002 rejected; reports for [development](results/h001-h002-development.md)
   and [validation](results/h001-h002-validation.md))*
6. Plan for a WebSocket recorder; where it runs depends on H-002. *(done: probe below; the recorder
   is deferred. H-002 is rejected: most impact arrives within 1 s, about 70% within the first 100 ms
   batch, so a recorder would have to capture events finer than 100 ms, with exchange timestamps)*
   - Documented (Spot WebSocket v4, read 2026-11-23): `spot.trades` is real time and
     `create_time_ms` may carry decimals below 1 ms; `spot.book_ticker` updates every 10 ms;
     `spot.obu` sends 50 levels every 20 ms or 400 levels every 100 ms, with `U`/`u` sequencing.
     No connection limits and no server location are documented.
   - Consequence if confirmed: from public channels the order book cannot be resolved below 20 ms
     (best bid and ask: 10 ms), wherever the recorder runs.
   - `mmlab.data_ingestion.gate_ws_probe` checks this from the owner's machine on pairs outside
     U-001: message spacing, delay against exchange time, sequence breaks, disconnects, ping round
     trip, and the API's addresses against AWS ip-ranges.
   - Probe, 2026-11-23 22:02–22:12 UTC, BTC_USDT, LUNC_USDT, RAIN_USDT, owner's machine (clock
     NTP-synchronised, 2 ms off); summary in
     [results/ws_probe/](../results/ws_probe/probe-20261123T220212Z.txt):
     - All 8 addresses of `api.gateio.ws` are in AWS ap-northeast-1 (Tokyo) by AWS ip-ranges and
       reverse DNS. Gate does not document where its matching engine runs; Tardis.dev states that
       Gate's servers are in AWS ap-northeast-1.
     - WebSocket ping round trip p50 264 ms (min 258, max 534, 113 pings); TCP connect 262–310 ms.
       Stage 1 measured 391–412 ms; the route varies.
     - Exchange time to local receipt: p5 143–145 ms, p50 143–159 ms, p95 up to 227 ms.
     - `spot.obu`, 50 levels: BTC_USDT messages are p10 20 ms and p50 24 ms apart, which confirms the
       20 ms batching. On the two illiquid pairs the p10 spacing is 9–11 ms, so `t` is not on a fixed
       20 ms grid; what `t` marks (window end or last change) is open. No sequence breaks, one full
       book per stream, no disconnects.
     - `spot.book_ticker`: p10 spacing 6–11 ms. `spot.trades`: all 1,255 trades carry
       sub-millisecond `create_time_ms`.
   - Tardis.dev (paid) records Gate spot `obu` with 400 levels, i.e. every 100 ms (its 2026-11-01
     sample subscribes `ob.{pair}.400`), so its book is not finer than Gate's free archive; it adds
     `book_ticker` history since 2023-04-29.
   - Decision 2026-11-24 (owner): no forward recording for now; where a recorder would run is
     decided later. Research continues on the archive (100 ms batches).
7. Holdout for H-001 and H-002, opened once after 2026-12-01 (November trades): the 12 holdout pairs
   over 2026-08-01 – 2026-11-30 and the research pairs in November. It estimates the effects; the
   decisions stand. Order once November is published: (a) H-003 is decided on the research pairs in
   November; (b) H-001 and H-002 effects are estimated on the research pairs in November; (c) the
   holdout pairs are opened once for all three hypotheses. If H-003's test period is extended to
   December, (c) waits until H-003 is decided.
8. Exploration for H-003 on 2026-08-01 – 2026-10-31, which H-001 and H-002 have already used:
   realized spread by the book state a maker can see before a trade, i.e. the quoted half-spread q
   and the maker-side share rho of the best quantities (`mmlab.microstructure.conditional`). H-003
   is registered from it before 2026-12-01, so that November and the holdout pairs are opened once
   for H-001, H-002 and H-003 together and stay unseen by H-003 until its test. *(done: only quoted
   half-spreads of at least 80 bp show a positive median θ in both periods; H-003 registered
   2026-11-24)*
9. Implement the test of H-003 before November is published (2026-12-01).

Results are reported as a function of latency (1, 10, 100, 400 ms) wherever latency matters.

Implementation of the tests (step 5): `mmlab.microstructure.markouts` (as-of joins on availability
time, validity enforced) and `mmlab.evaluation.h001_h002` (block bootstrap over days, BH across pairs).
H-002's ratio S is computed on trades usable at both 1 s and 300 s, so that both horizons describe the
same trades.
