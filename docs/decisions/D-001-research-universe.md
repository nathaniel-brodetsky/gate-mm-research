# D-001: Research universe U-001

- **Decided:** 2026-11-23, before any order book or trade data of the selected pairs was examined
- **Implementation:** `mmlab.market_data.universe` (`UniverseRule` defaults are the rule)
- **Output:** `configs/universe_U001.json`, committed separately after the rule

## Why

Stage 1 showed that the global spot venue has 643 USDT pairs with a median daily volume of 1k–10k and
722 with 10k–100k, that its spot archive is complete, and that Gate EU has no usable volume. The research
object is therefore illiquid USDT pairs of the global spot venue.

## Rule

1. Quote currency USDT. Excluded bases: stablecoins and pegged commodity tokens (list in code) and
   leveraged tokens (`…3L`, `…5S`, …).
2. Listed before the selection window: at least one daily candle before 2026-07-01.
3. Selection window 2026-07-01 – 2026-07-31: a candle for every day, volume on at least 28 days.
4. Median daily quote volume over the window in [1,000; 100,000) USDT, four strata equal in log10:
   [1,000; 3,162), [3,162; 10,000), [10,000; 31,623), [31,623; 100,000).
5. Within each stratum, pairs are ordered by SHA-256 of `mmlab-U001-2026-11-23` + pair name. The first 6
   are the research set, the next 3 the holdout set: 24 research pairs, 12 holdout pairs.

The selection window precedes the development period (from 2026-08-01), so selection uses no data from
the periods it will be tested on.

## Known limitations

- **Survivorship.** Candles come only for pairs tradable on 2026-11-23. Pairs delisted between July and
  November cannot be selected, which removes exactly the pairs where delisting risk materialised. Their
  share is unknown; it must be estimated before any capacity or PnL claim.
- **Exchange-reported volume.** Wash trading or fee programmes are not excluded at selection.
- **Asset type.** Tokenized stocks or other assets with external trading hours are not identified by
  name and may be selected; they will be flagged when found, not silently removed.

## Holdout discipline

Holdout pairs are not loaded, plotted or summarised until a hypothesis has been decided on the research
pairs; then they are opened once per hypothesis, and that use is recorded in the hypothesis file.

## Amendment 1 — 2026-11-23, after the selection, before any order book or trade data was loaded

The research set contains SPYX_USDT, whose ticker suggests a tokenized equity (S&P 500 tracker), and
TEVA_USDT may be one too. Fixed now, before any data of these pairs is loaded:

- Decisions on H-001 and H-002 use all 24 research pairs as selected. No pair is replaced.
- Every result is also reported without the pairs that asset metadata
  (`configs/universe_U001_assets.json`, from `mmlab.data_ingestion.gate_assets`) identifies as
  tokenized equities or other assets with external trading hours. The excluded pairs are listed here,
  in a further amendment, before any result is computed.

## Amendment 2 — 2026-11-23, from asset metadata, before any order book or trade data was loaded

Asset metadata (`configs/universe_U001_assets.json`) names SPYX_USDT "SP500 xStock", a tokenized equity
index. TEVA_USDT is "Tevaera", a crypto-native token, and is not flagged. Every result is reported for
all 24 research pairs (the basis of decisions) and for the 23 pairs without SPYX_USDT.
