# D-002: Testing standard for hypotheses about the edge of quoting

- **Decided:** 2026-11-24, after the synthetic lab of stage 3 (steps 1–2g of `docs/stage3-plan.md`),
  before November 2026 or any later data were opened
- **Evidence:** `results/lab/` (`realism_days.json`, `checks_days.json`, `compare_days.json`,
  `tails_days.json`, `pooled*.json`, `levels.json`, `levels_days.json`)
- **Applies to:** hypotheses registered after this record (H-004 onwards). H-001 to H-003 keep their
  registered rules; the last section says how to read them under this standard.

## Why

Daily maker markouts on these pairs are lopsided. On all trades the days are skewed to the loss side in
18 of 24 pairs: losses come in rare days of extreme prices of a single token (on the worst day of 18 of
24 pairs the price range is at least three times the pair's median day, and 23 of the 24 worst days
have no data problems), and most windows of 30–61 days miss them.

Of the three generators of the lab, only the one that rebuilds periods from real calendar days
reproduces the real daily statistics. Under it, per-pair day tests reject a true theta of 0 in
10.4–15.8% of windows for the median pair (nominal 5%) and in up to 51.5% for the worst; the size
follows the skewness of the pair's days (Spearman −0.82 across pairs). No candidate test fixes this
within a window, and Benjamini–Hochberg or Benjamini–Yekutieli across 24 pairs find an effect where
there is none in 53–94% of periods.

The median over pairs with a bootstrap over pairs works, but not at its nominal level on all trades:
with every pair at 0 it accepts in 7.1% of 30-day periods at a nominal 2.5%, because a pair's window
estimate is usually above its mean when the loss day is missing.

## Rule

1. **Unit of inference.** A hypothesis about the edge of a quoting rule is a claim about the
   population of illiquid pairs from which the research pairs were drawn at random (D-001). The
   decision statistic is the median over pairs of each pair's statistic over the test window. The
   inference is a percentile bootstrap over pairs with B = 10,000 resamples and the one-sided p-value
   p = (1 + #{bootstrap medians ≤ 0}) / (B + 1); at least 8 pairs must be counted.
2. **Calibrated level.** The hypothesis is accepted if p ≤ α. The level α is fixed at registration
   from the lab: the largest of 0.1%, 0.25%, 0.5%, 1%, 1.5%, 2%, 2.5% and 5% at which, under the
   calendar-day generator with 4,000 synthetic periods, the rule accepts in at most 2.5% of periods
   under each null truth:
   - every pair at 0;
   - the exploration values of the statistic centred on their median, for the same pairs in every
     period;
   - the same values drawn at random for the pairs in every period (pairs as a random sample).

   For the statistic of H-001 and H-003 (volume-weighted realized spread at 60 s minus the maker fee),
   from `results/lab/levels_days.json`:

   | Trades | 30 days | 61 days |
   |---|---|---|
   | all | α = 0.5%: size 1.8%; power 91.3% at a median of +10 bp | α = 1%: size 2.1%; power 97.8% |
   | quoted half-spread ≥ 80 bp | α = 2.5%: size 2.4%; power 47.5% at the exploration values | α = 1.5%: size 2.5%; power 46.3% |

   The parametric generator gives 2–2.5% for the same cells (`results/lab/levels.json`); the stricter
   calendar-day levels apply. Another statistic, subset or window needs its own run of the lab, with
   day pools built for that statistic, before registration.
3. **Lab before registration.** The hypothesis file records, from the calendar-day generator, the
   level, the size under each null truth, and the power at the effect seen in exploration and at half
   of it, with the parametric generator as a cross-check. A rule with power below 50% at the
   exploration effect may be registered only with that stated as its main limitation.
4. **Per-pair results are descriptive.** Per-pair p-values over windows of up to 61 days are not used
   to accept or reject anything and are not corrected across pairs. Each pair is reported with its
   statistic, the statistic without each of its three heaviest days, and the share of those days.
5. **Heavy days stay in.** No day, trade or pair is removed for being extreme. Data may be removed
   only if shown to be wrong (problem hours from the build, or a check against the raw archive),
   decided and recorded before the test period is opened. A rule that stops quoting in some
   conditions decides from as-of information only and is evaluated on the days it would have quoted.
6. **Economics over the longest history.** Estimates of profit or capacity use the longest available
   history with all days included, and report how many days carried the result.

## Known limitations

- The calendar-day generator knows only the 92 days of August–October 2026. Events heavier or more
  frequent than those are not reflected in the levels.
- The levels hold for the statistic, subsets and windows they were computed for, and for the null
  truths checked. Fixed configurations with many pairs exactly at 0 next to pairs far above 0 defeat
  any median rule; they are outside the random-pairs model of rule 1.
- The Monte Carlo error of a size near 2.5% with 4,000 periods is about 0.25 points. Two levels sit at
  the boundary (wide-spread trades: 2.4% over 30 days, 2.5% over 61 days).
- On wide-spread trades the spread of the statistic across pairs is large (−36 to +138 bp in
  exploration), so a shift of the median by 10 bp is almost never detected (power 0.6–3.2%).
- The median over pairs says that a typical pair has an edge, not which one. Choosing pairs needs its
  own hypothesis tested on new data; holdout pairs follow D-001.
- Rule 2 replaces the criterion written before step 2f (the centred truth accepted in at most 5%) by
  a stricter one, after seeing the results. The change goes in the conservative direction and is
  recorded in `docs/stage3-plan.md`.

## Reading H-001 to H-003 under this standard

- **H-001** (rejected) stands. Its median over pairs is −9.63 bp in development and −11.09 bp in
  validation, below 0 at any level. Its 2 of 24 per-pair rejections in development are what its
  per-pair test produces with no effect at all (with every pair at 0, Benjamini–Hochberg on those
  p-values finds at least one pair in 88.2% of 61-day periods), and neither replicated.
- **H-002** (rejected) is a statement about the share of price impact within the first 100 ms, not
  about edge; this record does not change it.
- **H-003** (registered; decided after 2026-12-01) already uses rule 1, at a level of 2.5%, which is
  the calibrated level for its subset and window: size 2.4% with every pair at 0 and 0.9% with random
  pairs, power 47.5% at the exploration values. Its per-pair p-values are reported only, as
  registered.
