# ToughCert

Fracture toughness prediction for steels and high entropy alloys, with
prediction intervals that carry a finite-sample statistical guarantee,
an applicability domain check on every prediction, and provenance back
to the source publication for every number the model produces.

This started as a plain regression pipeline (predict K_IC from
composition, processing and basic mechanical properties). The problem
with a point estimate is that nobody designing hardware can use one. A
valid plane-strain fracture toughness test per ASTM E399 or E1820 costs
real money per specimen, and design organisations certify against lower
bounds (MMPDS A- and B-basis values), not against means. So the project
was rebuilt around the question an engineer actually asks: what is the
lowest toughness I should reasonably expect from this material, how
sure are we, and which physical tests should we pay for next.

## What it does

Given a table of materials (composition string, condition, phase, grain
size, test temperature, whatever mechanical data is available, all
columns optional except some way to identify the material), ToughCert
produces for each row:

- a point estimate of fracture toughness (MPa m^0.5)
- 90% and 95% conformal prediction intervals. These are computed with
  group-aware CV+ (Barber, Candes, Ramdas and Tibshirani, Annals of
  Statistics 2021) and have guaranteed coverage of at least 80% and 90%
  respectively, for any regressor, at any sample size, as long as new
  material systems are exchangeable with the training groups
- a trust score and tier. Tier A means the query sits inside the
  training distribution, tier B is borderline, tier C means the model
  is extrapolating and the bounds should not be relied on
- the nearest training specimens with their literature citations, so
  every prediction can be traced to measured data during review

The interval lower bound is the number meant for decisions. Ranking
candidate alloys by guaranteed floor instead of by point estimate is
what makes the output usable for material down-selection.

### Why group-aware conformal matters here

The training data is mined from publications. Specimens from the same
paper share a lab, a melt and a test method, so rows are clustered, not
independent. If you calibrate conformal residuals with random row
splits, information leaks across the split and the intervals come out
too narrow for genuinely new alloys. ToughCert assigns calibration
folds at the level of publication plus composition, which is the level
at which a new query is actually new. As far as we know no other
materials property tool does this.

### Physics features

Raw composition strings are converted to descriptors with established
physical meaning before modeling: ideal mixing entropy, Miedema mixing
enthalpy (pairwise values after Takeuchi and Inoue 2005, with the
covered pair fraction tracked as its own feature), atomic size mismatch
delta, electronegativity spread, valence electron concentration,
mixture melting point, the Yang-Zhang Omega parameter and a rule of
mixtures density. From the measured columns it derives homologous test
temperature T/Tm, the Hall-Petch term 1/sqrt(d), elastic yield strain
YS/E, strain hardening capacity (UTS-YS)/YS, and the indentation
indices H/E and H^3/E^2. These help most exactly where data is thin,
and they make the applicability domain interpretable in physical
coordinates.

## How the numbers hold up

Measured on the bundled dataset (131 specimens from 90
publication/composition groups, K_IC from 0.2 to 459 MPa m^0.5, test
temperatures from 20 K to 1298 K), with whole groups held out so the
model is always scored on material systems it has never seen:

| Quantity | Value |
| --- | --- |
| 90% interval, empirical coverage on held-out groups | 91.6% (guarantee: >= 80%) |
| 95% interval, empirical coverage on held-out groups | 95.1% (guarantee: >= 90%) |
| Point MAE on held-out groups | 25.2 MPa m^0.5 |
| R2 on held-out groups | 0.52 |
| Bundled unseen set (15 specimens), measured value inside 90% bounds | 13 of 15 |

The intervals are wide. That is the honest output of 131 noisy
literature points spanning three orders of magnitude, and it is exactly
the information a test-planning decision needs. Every qualification run
recomputes this table for the current data and writes it into the model
card, so the calibration claim is always backed by the artifact in
front of you, not by this README.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 1. featurize the raw tables (writes data/processed_*.csv and data/schema.json)
python -m src.prepare_data ^
  --train assets/combined_fracture_training.csv ^
  --unseen assets/combined_fracture_unseen.csv ^
  --out_train data/processed_train.csv ^
  --out_unseen data/processed_unseen.csv

# 2. train the qualification artifact (model selection, conformal
#    calibration, applicability domain, model card)
python -m src.qualify --config configs/default.yaml

# 3. predict with bounds, trust tiers and provenance, plus an HTML report
python -m src.certify --data data/processed_unseen.csv ^
  --out predictions.csv --report reports/qualification_report.html

# 4a. screen candidate alloys around known compositions, ranked by
#     guaranteed lower bound under a density ceiling
python -m src.screen --mode screen --temperature 298 --max-density 8.0 --top 20

# 4b. or ask which physical tests are worth buying next
python -m src.screen --mode advise --data data/processed_unseen.csv --top 10
```

A committed example of the outputs is in `examples/`: the predictions
CSV, the test priority ranking and the HTML report
(`examples/qualification_report.html`, self-contained, open it in a
browser).

## Design allowables

`src/allowables.py` computes one-sided lower tolerance bounds from
measured samples: B-basis (90% coverage at 95% confidence) and A-basis
(99/95), using exact normal-theory factors from the noncentral t
distribution and, where the sample size permits, distribution-free
order statistic bounds. Use it on your own test results and compare
against the model bounds. The model output itself is a screening value
for prioritisation and down-selection. It is not a substitute for
testing or for MMPDS/CMH-17 qualification, and the report says so on
every page.

## Interpreting the output columns

| Column | Meaning |
| --- | --- |
| `predicted_toughness_mpa_m0_5` | point estimate |
| `lower_90` / `upper_90` | conformal bounds, >= 80% guaranteed coverage |
| `lower_95` / `upper_95` | conformal bounds, >= 90% guaranteed coverage |
| `trust_score` | 0 to 100, distance-based, 100 is deep interpolation |
| `trust_tier` | A interpolation, B boundary, C extrapolation |
| `nearest_training_anchors` | closest measured specimens with citations |
| `test_value_score` (advise mode) | relative interval width times novelty |

Treat tier C rows as unanswered questions, not as predictions. They are
the rows the advise mode will usually tell you to test first.

## Limitations, stated plainly

- 131 training points. The model interpolates a sparse literature
  corpus; it does not know mechanism. The conformal machinery is there
  precisely because the point model is weak.
- The data mixes K_IC, K_Q and other toughness measures as reported by
  the source papers, in MPa m^0.5, without re-validating specimen size
  criteria.
- Miedema enthalpies are model estimates, rounded, and pairs involving
  N, O, S and Pb are excluded (the coverage fraction is a feature, so
  the model can discount those rows).
- The coverage guarantee is conditional on group exchangeability. A
  query from a genuinely different population (a ceramic, a weld, an
  irradiated steel) gets a tier C flag, and its bounds mean little.
- Old pipeline entry points (`src.train`, `src.predict`, the segment
  scripts and the FT-Transformer) still work but do not produce
  intervals. Use the qualify/certify path for anything that matters.

## Repository layout

```
assets/            bundled literature dataset (with citations)
configs/           YAML configs
examples/          committed example outputs, incl. the HTML report
src/physics.py     composition and mechanics descriptors
src/conformal.py   group-aware CV+ intervals and calibration checks
src/applicability.py  trust scores, tiers, nearest-neighbour provenance
src/allowables.py  A-/B-basis tolerance bounds
src/qualify.py     trains the qualification artifact + model card
src/certify.py     batch predictions, CSV + HTML report
src/screen.py      candidate screening and test prioritisation
src/train.py       legacy point-estimate training (kept working)
tests/             pytest suite, runs the real pipeline end to end
```

## Tests

```bash
pytest -q
```

42 tests, including an end-to-end run of prepare/qualify/certify/screen
on the bundled dataset and a subprocess test of the CLI artifact round
trip.

## References

- Barber, Candes, Ramdas, Tibshirani. Predictive inference with the
  jackknife+. Annals of Statistics 49(1), 2021.
- Takeuchi, Inoue. Classification of bulk metallic glasses by atomic
  size difference, heat of mixing and period of constituent elements.
  Materials Transactions 46(12), 2005.
- Yang, Zhang. Prediction of high-entropy stabilized solid-solution in
  multi-component alloys. Materials Chemistry and Physics 132, 2012.
- MMPDS-2023 / CMH-17 for the definition of A- and B-basis values.
