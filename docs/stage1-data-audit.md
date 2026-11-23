# Stage 1 — Gate market data audit

Status: **complete** · Audit date: 2026-11-23 (UTC) · Owner: Nathaniel Brodetsky

Evidence labels used below:

- **[docs]** stated in Gate documentation or announcements, not verified by us
- **[measured]** measured by us on archive files or live endpoints on the audit date
- **[secondary]** third-party source, not verified against a primary source
- **[hypothesis]** our interpretation, not yet tested

## 1. Conclusions

1. Gate publishes free hourly L2 order book archives (`orderbooks`) and monthly trade archives
   (`deals` spot, `trades` futures) for illiquid pairs too. [measured]
2. The archive format differs from Gate's WebSocket documentation and in one place from Gate's own
   archive description. Every rule below was verified on data (section 4).
3. **Spot archive: usable.** October 2026, LUNC_USDT and RAIN_USDT: 743 of 744 hours usable each. Both
   have one defect in the same hour (2026-10-13 16h), so spot incidents are venue-wide too. [measured]
4. **Futures order book archive: not usable as a research sample.** GPS_USDT complete hours: October
   2026 about 290 of 744, November 1–22 212 of 528 (40%). Outages are global across contracts and, in
   October, concentrated in weekday afternoons UTC, so the remaining sample is biased towards quiet
   hours. [measured]
5. Trade archives are complete where checked (no id gaps in October for LUNC_USDT, GPS_USDT), including
   during futures order book outages. [measured]
6. Book rows are 100 ms batches labelled with the **start** of the window. Book state labelled `t` may
   be used only from `t + 100 ms`. [measured]
7. Over 99.9% of `take` volume in the book is cancellations, not trades. Fills must come from trade
   files, not from book takes. [measured]
8. Forward data (shadow mode, forward tests) cannot come from the archive: books are published
   ~1 h late and trades only monthly. An own WebSocket recorder is required. [measured]
9. **Gate EU is a separate market with negligible capacity.** Own host and order books (Paris), 427
   pairs quoted in EUR and USDC only (no USDT). Median venue day Sep–Nov 2026: about 68k–152k USDC +
   16k–29k EUR; over the last 30 days 399 of 427 pairs had a median daily volume of zero, and the most
   active pair (ETH_USDC) a median of 61k USDC. There is no Gate EU archive (EU-only pairs return 404),
   so its history beyond daily candles and the last 30 days of trades exists only if we record it.
   [measured]
10. **Latency from the owner's network:** 391–412 ms TCP round trip to the global API, 64 ms to Gate EU.
    Results of later stages will be reported as a function of latency (1, 10, 100, 400 ms), since the
    strategy may be run by parties with co-located infrastructure. [measured]
11. **Public REST trades are limited to the last 30 days** on both venues ("for public trades older than
    30 days, download from historical data instead"). [measured]
12. **The global spot universe is large.** Over the last 30 days 643 pairs had a median daily volume of
    1k–10k and 722 pairs 10k–100k (quote currency, mostly USDT), versus 10 and 2 on Gate EU.
    Exchange-reported volume, not audited. [measured]
13. **Open blocker:** which venue a Romanian resident may use. Per primary sources Romania is not named
    in the global platform's restricted list, but that list is non-exhaustive, and MiCA allows only
    authorised providers to serve EU residents (Gate's EU entity is Gate EU). Not legal advice; the
    account onboarding decides. [docs] [secondary]

## 2. Venue and legal status (Romania)

| Item | Finding | Label |
|---|---|---|
| Global platform terms | Clause 2.5 Restricted Locations: US, Mainland China, Singapore, Canada, France, Germany, Hong Kong, Malaysia, Malta, Spain, Netherlands, UK, Austria, Japan, and others; "include but are not limited to"; updated without notice. Romania not named. Perpetuals and margin "not allowed in the jurisdictions in which they are prohibited" | [docs] gate.com/legal/user-agreement |
| MiCA | Since 2026-07-01 only authorised CASPs may serve EU clients (narrow reverse-solicitation exception) | [secondary] |
| EU entity | Gate Technology Limited (Malta), MFSA, authorised 2025-09-29, LEI 984500D6A0F945BB5A15; 6 MiCA services incl. trading platform, exchange crypto/fiat and crypto/crypto, order execution | [secondary] renderings of the ESMA register |
| Gate EU products | Spot, convert, fiat; USDT not tradable (one-way USDT to USDC conversion) | [docs] Gate Europe announcement 228081 |
| Gate EU API | `https://api.gateeu.com/api/v4`; data centre AWS Paris (eu-west-3); public endpoints 900 r/s per IP; orders 10 r/s per user and pair | [docs] search snippet of the Gate EU API page |

Not legal advice. Which platform and products are available is decided at account onboarding.

### 2.1 Venue comparison — measured 2026-11-23 ~14:57 UTC from the owner's network

Tool: `python -m mmlab.data_ingestion.gate_venues OUT_DIR` (public endpoints, no credentials).

| | Global (`api.gateio.ws`) | Gate EU (`api.gateeu.com`) |
|---|---|---|
| Network RTT, min TCP connect of 5 | 391–412 ms (3 runs) | 64 ms |
| HTTP requests | 0.5–2 s small, 15–22 s for full pair and ticker lists | 0.1–0.6 s |
| Public data without key | yes | pairs, tickers, order books yes; `/spot/time` needs a key; every request needs a `Timestamp` header |
| Tradable pairs | 2,157 (USDT 2,008, USDC 66, …) | 427 (EUR 214, USDC 213, no USDT) |
| 24 h quote volume | ≈ 2.17 bn USDT, 5.0 m USDC | ≈ 72.8 k USDC, 32.8 k EUR |
| Most active | BTC_USDT 1.0 bn, ETH_USDT 0.37 bn | USDC_EUR 24.8 k, BTC_USDC 21.2 k, QNT_USDC 15.4 k, AAVE_USDC 14.5 k, ETH_USDC 8.7 k; all others < 3 k |

Order books of the six most active pairs listed on both venues, fetched back to back: **different on
every pair** (different book ids, prices and sizes), so the venues do not share a matching engine.
Quoted spreads in that snapshot (single observation):

| Pair | Gate EU spread | Global spread (same USDC pair) |
|---|---|---|
| BTC_USDC | 10.9 bp | 2.5 bp |
| ETH_USDC | 2.8 bp | 10.7 bp |
| SOL_USDC | 2.6 bp | 13.1 bp |
| BNB_USDC | 5.2 bp | 15.7 bp |
| XRP_USDC | 6.1 bp | 11.1 bp |
| PENDLE_USDC | 21.9 bp | 95 bp |

[hypothesis] Gate EU majors are quoted tightly by an incumbent liquidity provider despite very small
traded volume; with a 0.10% maker fee per side (Gate EU fee not verified) a maker–maker round trip
costs about 20 bp, more than most of these spreads.

### 2.2 Gate EU trading history — daily candles, measured 2026-11-23

Tool: `python -m mmlab.data_ingestion.gate_rest_history OUT_DIR` (427 tradable pairs, closed daily
candles, up to 365 days). Only pairs tradable today are covered: pairs delisted earlier (for example
USDT pairs before July 2026) are missing, so early months are understated. [measured]

| Month | USDC volume (median day) | EUR volume (median day) | Largest pairs by share |
|---|---|---|---|
| 2025-11 (8 d) | 13.1 m (1.45 m) | 0 | BTC_USDC 67%, ETH_USDC 27% |
| 2025-12 | 68.5 m (12 k) | 0 | BTC_USDC 61%, ETH_USDC 36% |
| 2026-01 | 70.3 m (1.50 m) | 3 k | BTC_USDC 23%, ETH_USDC 9%, LTC_USDC 6% |
| 2026-02 | 52.0 m (1.97 m) | 0.2 k | BTC_USDC 16%, ETH_USDC 6%, LTC_USDC 6% |
| 2026-03 | 10.5 m (254 k) | 95 k | BTC_USDC 19%, STX_USDC 8% |
| 2026-04 | 7.3 m (154 k) | 38 k | BTC_USDC 9%, ETH_USDC 6% |
| 2026-05 | 1.4 m (40 k) | 1 k | BTC_USDC 17%, NEAR_USDC 10% |
| 2026-06 | 2.0 m (1.8 k) | 2.9 m (13) | BTC_EUR 19%, ETH_EUR 18% |
| 2026-07 | 0.28 m (5.9 k) | 0.17 m (4.5 k) | BTC_USDC 23%, USDC_EUR 22% |
| 2026-08 | 1.8 m (11.5 k) | 1.3 m (6.3 k) | USDC_EUR 23%, ETH_USDC 15% |
| 2026-09 | 3.6 m (68 k) | 0.8 m (16 k) | ETH_USDC 34%, BTC_USDC 22% |
| 2026-10 | 4.8 m (152 k) | 1.1 m (29 k) | ETH_USDC 26%, BTC_USDC 21% |
| 2026-11 (22 d) | 3.4 m (126 k) | 0.7 m (25 k) | ETH_USDC 33%, BTC_USDC 10% |

- Late 2025 volume came in bursts (BTC_USDC 10.3 m on 2025-12-12 alone). [measured] Whether this was
  promotional or organic is unknown. [hypothesis: launch incentives]
- July 2026, the end of the MiCA transition and the removal of USDT, is the low point. [measured]
- Last 30 days, pairs by median daily quote volume: 0 — 399; < 100 — 7; 100–1k — 9; 1k–10k — 10;
  10k–100k — 2; ≥ 100k — 0. Top medians: ETH_USDC 61.0k, BTC_USDC 16.8k, USDC_EUR 8.7k, INJ_USDC 7.0k,
  SOL_USDC 7.0k, AAVE_USDC 5.1k, QNT_USDC 4.8k, XRP_USDC 3.8k, PENDLE_USDC 3.4k, BTC_EUR 2.3k. [measured]
- Illustration, not an estimate: a maker taking a third of ETH_USDC median volume at a net 5–10 bp would
  earn about USD 10–20 per day, and that net edge is itself unproven.
- No Gate EU archive: BTC_EUR, ETH_EUR, USDC_EUR return 404 for `deals` and `orderbooks` (2026-09 – 11);
  BTC_USDC, listed on both venues, is present but is the global book. [measured]

### 2.3 Global spot trading history — daily candles, measured 2026-11-23

Same tool with `--venue global`: 2,157 tradable pairs, 704,133 closed daily candles, 12 pairs failed
(network). Volumes are as reported by the exchange; wash trading or fee programmes are not excluded,
so they are an upper bound for capacity work. [measured]

| Month | USDT volume (median day) | USDC volume (median day) |
|---|---|---|
| 2025-11 (8 d) | 18.4 bn (2.38 bn) | 270 m (35.1 m) |
| 2025-12 | 69.3 bn (2.29 bn) | 1,623 m (56.2 m) |
| 2026-01 | 77.0 bn (2.55 bn) | 1,089 m (35.2 m) |
| 2026-02 | 68.6 bn (2.21 bn) | 975 m (15.4 m) |
| 2026-03 | 60.3 bn (1.89 bn) | 372 m (11.3 m) |
| 2026-04 | 41.4 bn (1.38 bn) | 265 m (8.3 m) |
| 2026-05 | 46.5 bn (1.53 bn) | 211 m (7.0 m) |
| 2026-06 | 70.6 bn (2.28 bn) | 203 m (6.0 m) |
| 2026-07 | 38.7 bn (1.31 bn) | 139 m (4.2 m) |
| 2026-08 | 44.3 bn (1.03 bn) | 151 m (2.3 m) |
| 2026-09 | 53.3 bn (1.84 bn) | 235 m (7.5 m) |
| 2026-10 | 53.4 bn (1.70 bn) | 189 m (5.6 m) |
| 2026-11 (22 d) | 47.0 bn (2.05 bn) | 144 m (5.8 m) |

Pairs by median daily quote volume, last 30 days:

| Bucket | Global | Gate EU |
|---|---|---|
| 0 | 7 | 399 |
| < 100 | 125 | 7 |
| 100 – 1k | 189 | 9 |
| 1k – 10k | 643 | 10 |
| 10k – 100k | 722 | 2 |
| ≥ 100k | 458 | 0 |

Most active global pairs (median day, last 30 days): BTC_USDT 892 m, ETH_USDT 327 m, SOL_USDT 90 m,
USDC_USDT 60 m, XRP_USDT 50 m. [measured]

## 3. Data sources

### 3.1 Official archive — `download.gatedata.org`

URL scheme [docs, verified by probing]:

```
monthly: {biz}/{type}/{YYYYMM}/{market}-{YYYYMM}.csv.gz
hourly:  {biz}/orderbooks/{YYYYMM}/{market}-{YYYYMMDDHH}.csv.gz
         {biz}/orderbooks_slice/{YYYYMM}/{market}-{YYYYMMDDHH}.gz
biz: spot | futures_usdt | futures_btc        all periods UTC
```

| Type | Scope | First data found | Notes | Label |
|---|---|---|---|---|
| `orderbooks` | spot, futures | 2021-08 (BTC_USDT) | hourly; snapshot + 100 ms deltas | [measured] |
| `deals` / `trades` | spot / futures | 2021-08 (BTC_USDT) | monthly; docs say spot from 2023-01 | [measured] |
| `orderbooks_slice` | BTC_USDT only among pairs probed | 2021-08 | 20-level JSON snapshots, ~3,100/h, ids not monotonic; none for illiquid pairs | [measured] |
| `candlesticks_1m` | spot | — | absent from 2025-06 on, even BTC_USDT; build bars from trades | [measured] |
| `funding_applies` | futures | 2021-08 | every 8 h; 93 rows in Oct 2026 for GPS_USDT | [measured] |

Illiquid pairs appear from their first archived month; earlier months return 404 (listing date vs.
archive gap not distinguished). [measured]

Publication lag [measured, Last-Modified headers, 2026-10/11]:

| File | Published |
|---|---|
| spot `orderbooks` hour H | 40–50 min after H+1 |
| futures `orderbooks` hour H | 60–80 min after H+1 |
| `orderbooks_slice` | ~2 s after H+1 |
| monthly `deals` / `trades` | 1st of next month, 04:30–04:48 UTC |
| monthly `funding_applies` | 1st of next month, ~02:00 UTC |

Volume (gzip) [measured]: spot BTC_USDT ~17 MB/h (2026-11); futures BTC_USDT 8–40 MB/h (2026-10);
illiquid pairs 0.05–0.75 MB/h.

### 3.2 REST and WebSocket (documented, not tested)

- REST `https://api.gateio.ws/api/v4`; public endpoints 200 requests / 10 s per endpoint per IP; spot
  order book depth up to 100 levels. Matching engine in AWS ap-northeast-1 (Tokyo). [docs]
- Public spot WS: `spot.trades` (real time), `spot.book_ticker` (10 ms), `spot.order_book_update`
  (20 ms / 20 levels, 100 ms / 100 levels), `spot.obu` (50 levels / 20 ms, 400 levels / 100 ms). Local
  book sync via `U`/`u` ids and REST snapshot `with_id`. SBE endpoint available. [docs]
- Futures WS: `futures.obu` 50 levels / 20 ms, 400 levels / 100 ms. [docs]
- Connection and subscription limits: not documented beyond "one subscription per stream per
  connection". [docs]
- `GET /spot/trades` rejects windows older than 30 days with `INVALID_PARAM_VALUE` "for public trades
  older than 30 days, download from historical data instead", on both venues. [measured]
- Public REST tickers were reachable from the owner's network on 2026-11-23: 2,009 spot USDT pairs
  (524 with 24 h quote volume 20k–200k USDT), 1,037 USDT perpetuals (289 in that band). [measured]

### 3.3 Fees (global platform)

From 2026-04-09 [docs]: spot VIP 0 maker 0.100% / taker 0.100% (0.090% / 0.090% paid in GT); spot
maker 0% only from VIP 15. USDT perpetual taker VIP 0 0.050%; perpetual maker rates are not in the
announcement (third parties quote 0.020%) [secondary]. Gate EU fees: unknown. Account-level rates
must be read from `GET /spot/fee` with a read-only key.

Implication: a spot maker–maker round trip at VIP 0 costs 0.18–0.20%.

### 3.4 Third-party data (not purchased, prices not verified)

Tardis.dev: Gate spot since 2020-07-01 (high caps), all pairs since 2022-06-09; Gate futures since
2020-07-01; free CSV samples for the first day of each month. [secondary] Kaiko, CoinAPI,
CryptoCompare: not checked.

## 4. Verified archive formats

All rules below are implemented in `mmlab.data_ingestion.gate_orderbook_file` and
`mmlab.data_ingestion.gate_trades_file` and covered by tests.

### 4.1 `orderbooks`

```
spot:    ts, side, action, price, amount, begin_id, merged      side: 1 = ask, 2 = bid
futures: ts, action, price, size, begin_id, merged              sign of size: + bid, - ask
```

| Rule | Evidence [measured] |
|---|---|
| File starts with a full-depth snapshot (`action=set`, one id, ts on the hour, `merged=0`) | all files read; LUNC ~9,100 levels, GPS 200–350 |
| `make` / `take` amounts are **deltas** (make adds, take removes) | absolute reading crosses the book in 97–99.8% of batches; delta reading: 0 crossed, 0 negative levels in ~85k updates; level count after replaying hour H equals the H+1 snapshot exactly |
| `take` = trades **and** cancellations | trades explain 0.014% (GPS), 0.090% (LUNC), 0.053% (RAIN) of take volume |
| Id chain: first update = snapshot id + 1, next `begin_id` = `begin_id + merged` | 0 breaks over LUNC 24 h and GPS clean hours; `id + 1` rule breaks |
| Hour chaining: H+1 snapshot id = last covered id of H; file H runs up to ~0.5 s past the hour | replayed H equals H+1 snapshot level by level (LUNC 2026-10-15, all 24 hours) |
| Timestamps: 100 ms steps; a row labelled `t` holds changes in `[t, t+100 ms)` | of trades matched to takes: 97.5% (GPS), 97.9% (LUNC), 99.0% (RAIN) fall in that window |
| Side encoding | trades matched to takes on the expected side: GPS 1052/1054, LUNC 711/711, RAIN 912/915 |

### 4.2 Trades and funding

```
spot deals:       ts, trade_id, price, amount, side     side: 1 = TAKER BUY, 2 = TAKER SELL
futures trades:   ts, trade_id, price, size             sign of size: + taker buy, - taker sell
funding_applies:  ts, rate
```

- Gate's description "1: sell, 2: buy" for spot deals matches only 19% (LUNC) and 9% (RAIN) of trades
  when read as taker side, and ~100% when inverted; it describes the resting order. [measured]
- Trade timestamps have microseconds; ids are consecutive within a month (Oct 2026: LUNC 42,885,
  GPS 70,265 trades, 0 gaps, 0 out-of-order). [measured]

## 5. Completeness

Tool: `python -m mmlab.orderbook.gate_coverage` — replays every hour and chains it to the previous.

### 5.1 Spot

| Market | Period | Hours | Complete and chained | Defects | Label |
|---|---|---|---|---|---|
| LUNC_USDT | 2026-10 | 744 | 743 | 2026-10-13 16h: 3,046 update ids missing mid-hour → 653 crossed batches, 48 negative levels; resynced at 17h | [measured] |
| RAIN_USDT | 2026-10 | 744 | 742 (743 usable) | 2026-10-13 16h flagged, chain to 17h broken — same hour as LUNC_USDT | [measured] |

### 5.2 Futures (`futures_usdt`)

GPS_USDT, 2026-10 [measured]:

| Period (UTC) | Archive content |
|---|---|
| 10-01 00:00 – 10-08 05:00 (174 h) | snapshot only, no updates |
| 10-08 06:00 | no file |
| 10-08 – 10-21, weekdays | outage from ~13:00–14:50 to 16:00–20:00; none on 10-10/11, 10-17/18 (weekends) |
| 10-13 00:16–03:00, 10-15 02:17–04:00 | additional outages |
| 10-22 05:00 – 10-31 23:00 (235 h) | no files (404) |
| Total | updates present in 302 / 744 h (41%); ~11 of those cut mid-hour |

Outages are global [measured]:

- 2026-10-15: GPS, ZEST, DEMINGLI stop within the same second (02:17:20) and around 12:40–12:42.
- 2026-10-03: snapshot-only files for BTC_USDT (~175 KB vs 7.7–10.6 MB on 2026-10-10), ZEST, DEMINGLI.
- 2026-10-25 and 10-31: 404 for BTC_USDT, ZEST, DEMINGLI.

Missingness is not random: weekday outages start near 13:30 UTC (US equity open in October), the most
active hours. [measured timing; cause = hypothesis] Any futures backtest on this archive is biased
towards quiet periods.

GPS_USDT, 2026-11-01 – 2026-11-22 (528 h) [measured]: 212 hours complete and chained (40%); 242 hours
have no file (404); 60 chain breaks; no complete hour on 11-01 – 11-04. The hour-by-hour pattern of the
November outages has not been characterised yet.

"Complete" = internally clean, chained to the previous hour, with updates until at least 3,500 s past
the hour.

### 5.3 Other observations

- Book activity dwarfs trading: LUNC 2026-10-15 12h — 23,271 book updates vs 68 trades. [measured]
- DEMINGLI_USDT 2026-10-15 01h–02h: 0.92M and 1.14M updates per hour vs 1k–10k typical. [measured]
- One out-of-order timestamp in DEMINGLI 02h, just before the 02:17 outage. [measured]

## 6. Rules for later stages

1. Use only hours that are internally clean **and** chained; treat each hourly snapshot as a resync
   point; drop the remainder of an hour after an id break.
2. Book state labelled `t` is available from `t + 100 ms`.
3. Fills and markouts use trade files; book `take` rows are not trades.
4. Queue position inside a 100 ms batch is unobservable; any queue model is an estimate and needs
   sensitivity analysis.
5. Futures research needs an own recorder or a paid history (Tardis), and the archive's time-of-day
   bias must be stated wherever it is used.
6. Numbers, sides and prices are parsed as exact decimals; archive formatting is inconsistent
   (`1e-08`, `…0000000000000000`).

## 7. Open questions

1. Venue and product availability for the account (global vs Gate EU; spot vs perpetuals).
2. Gate EU: fee schedule, WebSocket endpoint and limits, who provides the tight quotes.
3. Global spot capacity: how much of the volume in the 1k–100k buckets is organic, and how much a
   maker could capture net of fees and adverse selection (stage 2 onwards).
4. Account-level fees (`GET /spot/fee`, read-only key).
5. Time-of-day pattern of the November futures outages.
6. Spot completeness across more pairs and months before any historical study; cause of the
   2026-10-13 16h spot incident.
7. WebSocket behaviour in practice: message loss, resync frequency, latency from Romania.

## 8. Reproduction

```bash
python -m mmlab.data_ingestion.gate_archive --help    # availability probe (HEAD)
python -m mmlab.orderbook.gate_coverage --help        # hourly completeness and chaining
python -m mmlab.orderbook.gate_replay --help          # replay and integrity checks of hourly files
python -m mmlab.orderbook.gate_reconcile --help       # trades vs book takes
python -m mmlab.data_ingestion.gate_venues --help     # global vs Gate EU public data and latency
python -m mmlab.data_ingestion.gate_rest_history --help   # venue-wide daily volume, trades API depth
```

Raw files are cached under `data/raw/gate/` (not in git).

## 9. Sources

- Historical data page: https://www.gate.com/developer/historical_quotes
- Archive announcement and field list: https://www.gate.com/announcements/article/21688
- REST API v4: https://www.gate.com/docs/developers/apiv4/en/
- Spot WebSocket v4: https://www.gate.com/docs/developers/apiv4/ws/en/
- Futures depth channel `futures.obu`: https://www.gate.com/announcements/article/44946
- Order book push changes: https://www.gate.com/announcements/article/44678
- Fee schedule from 2026-04-09: https://www.gate.com/announcements/article/50390
- Gate EU entity: https://www.gate.com/en-eu/about-us
- Global user agreement (clause 2.5, clause 4): https://www.gate.com/legal/user-agreement
- Gate Europe, USDT not tradable: https://www.gate.com/en-eu/announcements/article/228081
- Gate EU API (Paris data centre, limits): https://www.gate.com/fr-eu/docs/developers/apiv4/
  (read via search snippet; the page served to our fetcher was a marketing page)
- Tardis.dev Gate coverage: https://docs.tardis.dev/historical-data-details/gate-io ,
  https://docs.tardis.dev/historical-data-details/gate-io-futures
- Secondary sources on MiCA status and EU product scope: casptracker.eu, perpfinder.com, bitcoin.com
  reviews (2026), as cited in the stage-1 conversation.
