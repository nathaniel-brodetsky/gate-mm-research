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
9. **Open blocker:** which venue the account can trade (global gate.com vs Gate EU, spot only per
   secondary sources) and whether Gate EU shares the global order book. The archive describes the
   global venue. [secondary]

## 2. Venue and legal status (Romania)

| Item | Finding | Label |
|---|---|---|
| EU entity | Gate Technology Ltd (Malta), MiCA CASP authorisation by MFSA, 29 Sep 2025, passported in the EEA | [secondary] |
| EU product scope | Spot, convert, fiat; no perpetuals for EU residents | [secondary] |
| Global platform | User agreement lists several EEA states as restricted locations; list is non-exhaustive; Romania not named in the excerpts seen | [secondary] |
| Gate EU API | Separate host `api.gateeu.com`; docs not readable by our fetcher | [docs], unverified |
| Gate EU order book | Shared with global or separate: **unknown** | — |

Not legal advice. The account owner must confirm which platform and products are available to him.

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
2. Whether Gate EU shares the global order book (compare public books at the same instant).
3. Account-level fees (`GET /spot/fee`, read-only key) and Gate EU fee schedule.
4. Time-of-day pattern of the November futures outages.
5. Spot completeness across more pairs and months before any historical study; cause of the
   2026-10-13 16h spot incident.
6. WebSocket behaviour in practice: message loss, resync frequency, latency from Romania.

## 8. Reproduction

```bash
python -m mmlab.data_ingestion.gate_archive --help    # availability probe (HEAD)
python -m mmlab.orderbook.gate_coverage --help        # hourly completeness and chaining
python -m mmlab.orderbook.gate_replay --help          # replay and integrity checks of hourly files
python -m mmlab.orderbook.gate_reconcile --help       # trades vs book takes
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
- Tardis.dev Gate coverage: https://docs.tardis.dev/historical-data-details/gate-io ,
  https://docs.tardis.dev/historical-data-details/gate-io-futures
- Secondary sources on MiCA status and EU product scope: casptracker.eu, perpfinder.com, bitcoin.com
  reviews (2026), as cited in the stage-1 conversation.
