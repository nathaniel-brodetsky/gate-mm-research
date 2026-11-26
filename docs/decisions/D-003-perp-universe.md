# D-003: Research universe U-002 (USDT perpetuals)

- **Decided:** 2026-11-26, before any order book or trade of the window was examined for a selected
  contract, with one exception: the order books of futures GPS_USDT for July–December 2025 were
  replayed in stage 4, step 2a, for coverage counts only (no prices, spreads or markouts).
- **Implementation:** `mmlab.market_data.perp_universe` (`PerpRule` defaults are the rule)
- **Output:** `configs/universe_U002.json`, committed separately after the rule

## Why

Stage 4 found the futures order book archive usable for July–December 2025 (step 2a). Gate's API lists
only today's contracts, so the candidates come from Tardis.dev's list of the symbols it recorded on Gate
futures, with their availability dates (step 2b-1): 587 USDT perpetuals listed before 2025-06-01 and
available on 2025-07-01, of which 219 are no longer listed. The loss would not have been random: among
the contracts active on at least 28 days of June 2025, 46 of the 87 with 100–1,000 median daily trades
are delisted since, 151 of 392 with 1,000–10,000, and 18 of 88 with more. A universe drawn from today's
list would have lost half of the least active contracts, the research object.

A contract's size is in contracts, and its multiplier is known only for contracts still listed, so a
rule based on quote volume could not treat delisted contracts alike. The rule uses the number of trades.

## Rule

1. Candidates: USDT perpetuals listed before 2025-06-01 and available on 2025-07-01 according to
   Tardis.dev (`results/perp_universe/candidates.csv`). Excluded bases: stablecoins and pegged commodity
   tokens (D-001's list).
2. Selection month June 2025, before the window: active on at least 28 days
   (`results/perp_universe/activity_2025-06.csv`).
3. Median daily trades in June 2025 in [100, 10,000), four strata equal in log10: [100, 316.2),
   [316.2, 1,000), [1,000, 3,162.3), [3,162.3, 10,000).
4. Within each stratum, contracts are ordered by SHA-256 of `mmlab-U002-2026-11-26` + name. The first 6
   are the research set, the next 3 the holdout set: up to 24 research and 12 holdout contracts.

## Periods

- Selection: 2025-06-01 – 2025-06-30.
- Exploration: 2025-07-01 – 2025-10-31.
- Held-out test: 2025-11-01 – 2025-12-31. Unlike stage 2 the test period lies in the past, so its
  protection is procedural: no order book or trade of a universe contract in this period is loaded,
  plotted or summarised before a batch of hypotheses is registered under D-002. Every commit that
  touches these months says so.
- Holdout contracts are not loaded until a hypothesis has been decided on the research contracts; then
  they are opened once per hypothesis, as in D-001.

## Delisted contracts

They stay in the universe. A contract delisted inside the window contributes its data until it stops
trading. Every result is reported for all selected contracts (the basis of decisions) and for those
still trading on 2026-01-01.

## Known limitations

- **Trade count, not volume.** Contracts with many small trades and contracts with few large ones can
  share a stratum.
- **A third party's list.** Tardis.dev lists what it recorded; as a check, every contract Gate lists
  today that was launched before June 2025 is in that list.
- **Asset type.** Tokenized equities or other assets with external trading hours are not identified by
  name; they are flagged when found, not removed, as in D-001.
- **Fees.** Which maker and taker fees an account pays on these contracts is not established from public
  data (stage 4 plan); results are stated before fees and under named fee scenarios.

## Amendment 1 — 2026-11-26, after the selection, while the exploration data was being built and before any of it was read

Tardis.dev's availability dates (`results/perp_universe/candidates.csv`) show that seven research
contracts were delisted inside the exploration period: RDAC_USDT on 2025-07-08, PEIPEI_USDT on
2025-07-31, BTT_USDT on 2025-09-12, AMP_USDT and REQ_USDT on 2025-09-27, PYR_USDT and ALU_USDT on
2025-10-17. NAVX_USDT was delisted on 2025-11-14, inside the test period; XNO_USDT and PUFFER_USDT only
after the window. Fixed now:

- No contract is replaced. Decisions use the research set as selected; a contract enters a median over
  contracts only where it meets the minimums of the measure (trades on enough days), so RDAC_USDT (about
  a week of data) will mostly drop out by itself.
- The run-up to a delisting is a regime of its own (Gate announces delistings days ahead). Every result
  is also reported without each delisted contract's last 14 days before its delisting date.
- Contracts delisted before 2025-11-01 have no data in the test period: the test will rest on at most
  17 research contracts, NAVX_USDT until 2025-11-14.
