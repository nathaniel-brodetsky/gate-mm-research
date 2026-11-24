# Stage 3 — Research program

Status: started 2026-11-24. Owner's brief: run every hypothesis and every method we have, record every
result. No orders, no keys: research on public data only.

## Rules that keep many tests honest

1. A method's results are trusted only after the method has been checked on synthetic data calibrated
   to ours, where the truth is known (the lab).
2. Hypotheses are registered in batches before their test data exists. A batch is tested at once, with
   the false discovery rate controlled across the batch, by a method the lab has shown to hold its size.
3. Exploration and test data never overlap. Exploration: 2026-08-01 – 2026-10-31, and November once
   it has been opened for H-001 – H-003. Next test window: December 2026 (deals published
   2027-01-01). The U-001 holdout pairs stay closed until a hypothesis is decided on the research pairs.
4. Everything is recorded: every variant including failures, the code commit and the data it ran on.

## Steps

1. Synthetic lab for the statistical methods *(done 2026-11-24, `results/lab/`)*. The generator
   (`mmlab.stochastic_models.markout_panel`) is calibrated per pair on the exploration period; the lab
   (`mmlab.evaluation.lab`) measures the coverage and size of the day bootstrap (30 and 61 days), the
   false discovery rate of Benjamini–Hochberg across 24 dependent pairs, the coverage of the bootstrap
   over pairs, and the power of the registered H-003 rule in November.
   Findings (synthetic months calibrated on the exploration period, 400 repetitions per check):
   - The registered day bootstrap (stationary blocks of mean 3 days, percentile interval) is too
     liberal. With a true theta of 0 it rejects at the 5% level in 9.5% of 30-day months for the
     median pair (7.9% over 61 days), and in up to 30.2% for DEEP_USDT. Its 95% interval covers the
     truth in 87.5% of 30-day months for the median pair, 52.5% for DEEP_USDT. On trades at quoted
     half-spreads of at least 80 bp: 11.7% rejections, 82.8% coverage.
   - Benjamini–Hochberg at q = 0.10 on those p-values, 24 pairs, no effect anywhere: at least one false
     discovery in 59.3% of 30-day months and 41.0% of 61-day periods. With half the pairs at +10 bp
     the false discovery rate is 11.6% (30 days) and 9.6% (61 days), power 66.5% and 73.5%.
   - The bootstrap over pairs of the median holds: coverage 93.0–94.8%, lower end above 0 under a
     true median of 0 in 2.4–3.9% of cases (nominal 2.5%).
   - The registered H-003 rule in a synthetic November accepts in 50.7% of months if every pair's
     August–October effect persists, 44.9% if it halves and 4.0% if it is absent.
2. If the lab finds a method miscalibrated, *(done 2026-11-24: `mmlab.evaluation.day_tests` holds five
   candidates; `python -m mmlab.evaluation.lab compare` runs them on the same synthetic months, with
   Benjamini–Hochberg and Benjamini–Yekutieli on top; `results/lab/compare.json`)* compare alternatives on the same synthetic data (for
   example a studentized or wider-block bootstrap, a t-test on daily values with a HAC variance,
   sign-flip tests over days, a correction of the false discovery rate for dependence) and register
   the one that holds its size for the next batch.
   Findings (400 synthetic months per pair and setting; size = rejections at 5% with theta 0):
   - No candidate holds 5% over 30 days on data like ours. The bootstrap-t is best: median size 7.6–7.7%
     (worst pair 21–23%), interval coverage 90.2–90.7%. Over 61 days: 6.5% (worst 16.3–16.5%),
     coverage 92.1–92.6%. On trades at quoted half-spreads of at least 80 bp (30 days): 8.5% with the
     Politis–White block (worst 11.3%), coverage 88.2%. The registered percentile bootstrap gives
     9.1% and 7.9%, the Newey–West t test 10.1% and 7.6%.
   - Studentizing costs some power: rejection at +10 bp, all trades, 82% against 90% over 30 days
     and 96.5% against 98.5% over 61 days.
   - Across 24 pairs with no effect anywhere, a false discovery after Benjamini–Hochberg on bootstrap-t
     p-values happens in 35–37% of 30-day months and 26–29% of 61-day periods; after
     Benjamini–Yekutieli in 19–20% and 11–12% (target 10%). With half the pairs at +10 bp,
     Benjamini–Yekutieli on bootstrap-t keeps the false discovery rate at 5.2–5.3% (30 days) and
     4.0–4.2% (61 days), with power 46–47% and 57–58%.
   - The calibration behind these numbers: trade-level noise has very heavy tails (fitted Student df
     4.0–4.7 on all trades, up to 9.4 on wide-spread trades); daily trade counts are strongly over-dispersed (gamma shape 0.03–3.4, except XT_USDT and
     SAITO_USDT); SUKU_USDT and DEEP_USDT hit the AR limit of 0.9, which produces the worst cases.
   - *Superseded by [D-002](decisions/D-002-testing-standard.md) after steps 2b–2g:* per-pair tests
     are descriptive, decisions use the median over pairs at a calibrated level. Original proposal:
     per-pair tests use the bootstrap-t with the Politis–White block and Benjamini–Yekutieli across
     pairs, over test windows of at least 61 days; claims about all pairs together use the median with
     a bootstrap over pairs, which holds its size.
2b. Realism of the generator *(done 2026-11-24, `results/lab/realism.json`)*: `python -m
    mmlab.evaluation.lab realism` ranks five real statistics of each pair (daily theta spread, lag-1
    correlation and tails, share of days with trades, variation of daily counts) among 1,000 synthetic
    periods. Ranks piling up near 0 or 1 show what the lab gets wrong.
    Findings: the parametric generator gets the lag-1 correlation (median rank 0.60) and the variation
    of daily counts (0.61) right, but its daily theta is far more dispersed than the real one (median
    rank 0.01; 13 of 24 pairs below 2.5%, 11 of 17 on wide-spread trades), it has too many days without
    trades (9 pairs above 97.5%), and it gets the tails wrong in both directions (8 pairs low, 8 high).
    The likely cause is its assumption that the markout noise of a trade does not depend on its size.
    The lab's numbers above are therefore not yet a basis for a standard.
2c. Resampling generator *(done 2026-11-24, `results/lab/*_empirical.json`)*: `python -m
    mmlab.evaluation.lab pools` keeps, per pair, the real trades as pairs (quote value, within-day
    residual) and the real daily trade counts; a synthetic month draws days' counts from the real ones
    and trades with their real sizes and residuals, then adds a known theta and an AR(1) daily effect
    estimated net of the trade noise. Within a day the volume-weighted residuals sum to 0, so the truth
    is exactly theta.
    Findings: the share of days with trades now fits (median rank 0.50, no pair outside 2.5–97.5%), and
    so do the lag-1 correlation and the variation of daily counts. The daily theta is still too
    dispersed (15 of 24 pairs below 2.5%, 15 of 17 on wide-spread trades), and the real daily tails are
    heavier than the synthetic ones for 8 pairs. With it the registered day bootstrap rejects a true 0 in
    9.6% of 30-day months for the median pair (7.6% over 61 days, worst 38.5% for XT_USDT);
    Benjamini–Hochberg finds something with no effect anywhere in 57% and 41% of periods; the H-003
    rule accepts in 46.6% / 21.1% / 2.4% of Novembers (effect persists / halves / absent). The best
    per-pair test is again the bootstrap-t with the Politis–White block: 8.5% (30 days), 7.0% (61 days),
    9.5% on wide-spread trades; Benjamini–Yekutieli on it still makes a false discovery in 27.8% and
    18.8% of periods with no effect anywhere.
    Cause: within a day, makers on the two sides of the book hedge each other (when the price moves,
    the one who sold loses what the one who bought gains), so trade residuals within a day are
    negatively correlated. Drawing trades one by one breaks that and makes days noisier than real.
    The unit of resampling has to be the day.
2d. Calendar-day generator *(done 2026-11-24, `results/lab/*_days.json`)*: `python -m
    mmlab.evaluation.lab pools --kind days` keeps, per pair, every calendar day of the calibration
    period as it was (volume-weighted markout sum centred on the pair's theta, volume, trade count;
    zeros on days without trades). A synthetic period draws real days in blocks of consecutive days
    (stationary bootstrap, mean block by Politis–White on the pair's days) and adds a known theta times
    the day's volume, so the truth is exactly theta and everything within a day, including the hedging
    between sides, sizes, tails and empty days, is real. In checks with several pairs, all pairs take
    the same calendar days, which keeps their real dependence (pairs then share the median block; the
    AR(1) common factor of the other generators is not used).
    Limits: there are only 92 source days, so synthetic periods overlap with each other and repeat
    August–October; tails and regimes not seen then are missing; and there is no dependence beyond a
    block.
    Findings (400 synthetic periods per check; realism 1,000):
    - It is the only generator that passes the realism check: median ranks 0.48–0.68, one miss in 24
      pairs and five statistics (lag-1 correlation of PINGO_USDT on all trades, higher in reality).
      Mean blocks range from 1.00 to 28.08 days (TEVA_USDT 28.08, CKB_USDT 19.30, ORCA_USDT 9.62,
      IMU_USDT 6.47).
    - Per-pair day tests break far more than with the other generators. The registered day bootstrap
      rejects a true 0 in 15.8% of 30-day months for the median pair (worst 51.5%, PINGO_USDT) and in
      13.4% over 61 days (worst 47.5%, SAITO_USDT); its 95% interval covers the truth in 79.4% and
      85.6%. The best candidate, the bootstrap-t with the Politis–White block, rejects in 13.4% (worst
      37.8%) and 10.4% (worst 39.0%), with power at +10 bp of 69.4% and 78.5%. On wide-spread trades the
      median size is 6.0–6.5% (HAC 11.7%) but coverage only 78–85%.
    - Across 24 pairs with no effect anywhere, Benjamini–Hochberg on the registered p-values finds
      something in 93.5% of 30-day months and 88.2% of 61-day periods; Benjamini–Yekutieli on
      bootstrap-t p-values in 53.5% and 53.2%.
    - The registered H-003 rule (median over pairs, bootstrap over pairs) accepts in 45.3% / 23.6% /
      1.9% of synthetic Novembers if the effect persists / halves / is absent (with the other
      generators 50.7% / 44.9% / 4.0% and 46.6% / 21.1% / 2.4%). This is the only check of a pooled
      rule on generated days, and only for wide-spread trades with every pair at 0. *Corrected after
      this record was first committed:* it said the bootstrap over pairs "holds under all three
      generators" with coverage 93.0–94.8%; that check (`median_ci`) draws pair estimates from a
      normal model and does not use the generator, so it says nothing about real days. Step 2f
      measures the pooled rule properly.
    - The expectation recorded with the generator before the run ("Add a calendar-day generator to the
      synthetic lab"), that it would be the optimistic bound, was wrong: it is the most realistic one
      and the hardest on per-pair tests. Its numbers take precedence, unless step 2e finds that the days
      driving them are data errors.
2e. Heaviest days *(done 2026-11-24, `results/lab/tails_days.json`)*: `python -m
    mmlab.evaluation.tails` reads the day pools and the comparison and reports, per pair, how lopsided
    the days are (skewness, excess kurtosis, share of the three heaviest days against normal days),
    the worst days with their dates, how much theta moves without each, their share of volume, the
    day's price range and its hours with data problems, and across pairs whether the size of the
    bootstrap-t follows lopsidedness or block length (Spearman, descriptive) and which dates are among
    the worst of several pairs.
    Findings (exploration period; sizes from the calendar-day generator):
    - The size of the bootstrap-t over 30 days follows the skewness of the pair's days (Spearman
      −0.82, p < 0.001, 24 pairs; excess kurtosis +0.47, p = 0.021), not the block length (−0.06,
      p = 0.78). On wide-spread trades: skewness −0.72 (p = 0.001, 17 pairs).
    - On all trades the days are skewed to the loss side in 18 of 24 pairs (below −1 in 14). The five
      pairs with the largest sizes (29–38%: XT, PINGO, ANON, IMU, SAITO) have skewness −5.1 to −8.4;
      the six with positive skewness have sizes of 3.3–9.0%. The three heaviest days carry 21–52% of
      the absolute daily deviations, against about 10% for normal days.
    - The worst day is a day of extreme prices: in 18 of 24 pairs its price range is at least three
      times the pair's median day (ANON_USDT 478% against 11%, IMU_USDT 313% against 7%, CREO_USDT 175%
      against 10%, PINGO_USDT 164% against 5%). It carries up to 38.5% of a pair's three-month volume
      (MLN_USDT), and dropping it alone raises theta by up to 10.9 bp (ZIX_USDT; SAITO_USDT 10.8,
      ANON_USDT 9.8).
    - Not data errors: 23 of 24 worst days have no hour with data problems. AIDOGE_USDT on 2026-09-23
      has 7 and is to be checked against the raw archive before a test relies on it.
    - Not one market-wide shock: 5 dates are among the three worst days of 3 or more pairs (one of 4);
      with worst days independent across pairs, 3.9 such dates are expected and 5 or more happen in
      33% of cases (simulation). Heavy days are events of single tokens.
    - On wide-spread trades the skewness is positive in 12 of 17 pairs: gains come in a few days (the
      three heaviest days carry 0.54–0.63 of the deviations for PEOPLEFB_USDT, NAVX_USDT and
      PINGO_USDT, among the pairs with the largest theta). There a one-sided test errs on the safe side.
    Consequences: when a pair's losses come from rare events of its token, a window of 30–61 days
    cannot measure its theta; a per-pair test in such a window mostly checks whether an event fell
    into it. The economically relevant question becomes whether such days can be recognised from
    as-of data early enough to stop quoting, and what is lost before that (step 3).
2f. Pooled median rule on realistic days *(done 2026-11-24, `results/lab/pooled*.json`)*: `python -m
    mmlab.evaluation.lab pooled` runs the rule of H-003 (median over pairs of each pair's
    volume-weighted theta, percentile bootstrap over pairs, lower end above 0, at least 8 pairs) for
    all trades and wide-spread trades, over 30 and 61 days, under four truths: every pair at 0; the
    exploration thetas minus their median (true median 0 with the real spread across pairs); the same
    plus 10 bp; the exploration thetas as they are. It reports the true median of the pairs counted.
    Findings (800 periods per truth; acceptance at the registered level, nominal 2.5%):
    - Calendar-day generator, all trades: every pair at 0 is accepted in 7.1% of 30-day periods and
      5.0% of 61-day ones; the centred thetas in 0.0%; +10 bp in 99.5%. Wide-spread trades: 2.5% and
      4.0% with every pair at 0, 0.3% and 0.0% centred, 4.1% and 2.0% at +10 bp, 48.4% and 59.1% at the
      exploration thetas (H-003's own subset and window: 2.5%, at nominal).
    - Every pair at 0, all trades, other generators: 2.9% and 3.3% (parametric), 4.1% and 3.3%
      (trade resampling).
    - Why: with every pair exactly at 0 and days skewed to the loss side, a pair's 30-day estimate is
      usually above its mean (the event day is missing), so the median over pairs drifts up. With the
      real, fixed spread across pairs, the signs of the estimates follow the true thetas and the rule
      almost never errs, which answers a question about these particular pairs, not the size.
    - The criterion written before this run (centred truth, at most 5%) is met, but it checked an easy
      case. D-002 will use a stricter one, a change made after seeing these results and in the
      conservative direction: acceptance at most 2.5% under every null truth.
    - What the rule tests: the median of the pairs' window estimates, with pairs as a random sample
      from the population of illiquid pairs (D-001 draws them at random within strata). The true
      median of the pairs' theta is reached only if the estimates are not skewed; the lab measures the
      gap. Fixed configurations with many pairs exactly at 0 next to pairs far above 0 defeat any
      median rule and are outside this model.
2g. Level of the pooled rule *(done 2026-11-24, `results/lab/levels*.json`)*: `lab pooled` keeps the
    bootstrap p-value of the median of every period and adds the random-pairs null (`centred_random`:
    in every period the pairs draw their thetas with replacement from the centred values). The summary
    gives the acceptance at levels from 0.1% to 5% and the calibrated level: the largest one at which
    every null truth (`zero`, `centred`, `centred_random`) is accepted in at most 2.5% of periods.
    4,000 periods per truth (Monte Carlo standard error about 0.25 points at 2.5%).
    Findings, calendar-day generator (parametric in brackets):
    - All trades: α = 0.5% over 30 days (2%), 1% over 61 days (2%); there, every pair at 0 is accepted
      in 1.8% and 2.1% of periods, random pairs in 0.5% and 1.1%, and a median of +10 bp in 91.3% and
      97.8%.
    - Wide-spread trades: α = 2.5% over 30 days (2.5%), 1.5% over 61 days (2.5%); every pair at 0 in
      2.4% and 2.5%, random pairs in 0.9% and 0.5%; power at the exploration thetas 47.5% and 46.3%, at
      +10 bp 3.2% and 0.6% (the thetas differ by over 170 bp across pairs).
    - H-003's registered level, 2.5% on wide-spread trades over 30 days, is the calibrated one.
    Fixed in [D-002](decisions/D-002-testing-standard.md).
3. After step 2e the question is concrete: can the days of extreme prices of a token be recognised
   from as-of data early enough to stop quoting, and what is lost before that? Rolling volatility,
   price range, volume and trade flow are the baselines.
   3a. Quote pauses *(done 2026-11-24, `results/h004_explore/quote_pause.json`)*: `python -m
       mmlab.microstructure.quote_pause` evaluates, on the exploration period and the research pairs
       only, 24 rules that stop quoting while the mid's range or the number of trades over the last
       60, 300 or 1,800 s is more than 2, 3, 5 or 10 times its mean over the previous 7 days
       (`mmlab.feature_engineering.asof_activity`: one-second grid, a signal is used 1 s after it was
       known; the archive's books count from their availability). A paused maker takes no part in the
       trades during the pause and earns, on the others, what the liquidity that traded earned
       (volume-weighted RS(60 s) minus the fee).
       Findings (median over 24 pairs; theta of all trades −10.45 bp):
       - No rule moves the median pair much. The best, a range over 1,800 s above 5 times its 7-day
         mean, raises theta by 0.69 bp, improves 15 of 24 pairs, keeps 81% of the volume and pauses
         47% of the worst day. The 24 rules give −0.46 to +0.69 bp and improve 8 to 15 pairs, which is
         what correlated rules without an effect give; 9 of the 12 rules on trade counts lower theta.
       - Even the ex-post bound is small: dropping each pair's three worst days raises theta by 3.36 bp.
         The loss is not mainly in the event days. The liquidity that trades on these pairs captures
         close to 0 bp of realized spread at 60 s on ordinary days too (H-001: median +0.37 bp before
         the fee in development, −1.09 bp in validation), and the 10 bp maker fee makes theta
         negative. Heavy days make the result risky and hard to test (step 2e), not negative.
       - Pausing 24–75% of the worst day's volume gains little. This fits H-002 (about 70% of the
         5-minute impact comes within the first 100 ms): most of the loss is probably done before a
         signal 1 s old reacts. Not measured directly.
       - No pause rule goes into H-004 as an edge hypothesis. Whether pauses cut the tails (risk, not
         the mean) is a question for the risk module.
       Fees (Gate's own pages, read 2026-11-24): VIP 0 pays 0.1% as maker. Negative maker fees on
       spot USDT pairs come only with the Global Market Maker Program
       (https://www.gate.com/gmmc): from 0.15% of the platform's monthly spot maker volume, or as a
       30-day trial for users with more than USD 20 million traded in 30 days; its 2022 announcement
       (https://www.gate.com/announcements/article/28527) lists −0.005% for the MM1 trial and at least
       −0.012% at the best tier. Even at −0.012% the median pair's traded liquidity would sit near 0
       (+1.6 bp in development, +0.1 bp in validation). The 24 research pairs had median daily volumes
       of at most USD 0.9 million together in July (upper ends of the D-001 strata), for all their
       trading, so a maker on them alone stays far below either threshold. An edge has to come from
       where and when to quote (steps 4–5: H-003 and conditions beyond the spread) or from quoting
       better than the average liquidity (step 6), not from fees or pauses.
   Synthetic market with known regime changes: do topological features (persistence of sliding-window
   embeddings of mid, spread and flow) detect them better and earlier than rolling volatility? Depth
   snapshots of the order book are built only if they do.
4. Feature library on the archive: quoted spread, imbalance, depth (if built), volatility, trade flow
   and its sign autocorrelation, time since the last trade, price impact per unit volume (Kyle's
   lambda), and topological features if step 3 passes.
   4a. Realized spread by as-of state *(this step)*: `python -m mmlab.microstructure.state_explore`
       cuts the trades of the exploration period into fixed buckets of five features known 1 s
       before the trade (`mmlab.feature_engineering.trade_state`): top-of-book imbalance and taker
       flow over 60 and 300 s, both signed so that positive points the taker's way; time since the
       last trade; the mid's range over 300 s against its 7-day mean. Per bucket: the median over
       pairs (at least 30 trades on 5 days) of theta, for all trades and for half-spreads of at least
       80 bp. The taker side, which `h003.states` does not return, is recovered by repeating its
       filter and checking that every row lines up; registered code is not touched before H-003 is
       decided.
5. Exploration on the exploration data, then one registered batch H-004… under D-002 before
   2027-01-01, tested on December.
6. A queue-aware event-driven simulator (fills from the archive's make/take deltas, never on a price
   touch) for whatever survives step 5, with results as a function of latency. Public data resolve the
   book to 20 ms at best (WebSocket) and 100 ms in the archive, so 100 and 400 ms can be simulated and
   1 and 10 ms cannot.
