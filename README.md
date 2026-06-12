# Fracture Toughness Qualification Suite (FTQS)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20669425-blue)](https://doi.org/10.5281/zenodo.20669425)

Fracture toughness prediction for steels and high entropy alloys, with
prediction intervals that carry a finite-sample statistical guarantee,
an applicability domain check on every prediction, and provenance back
to the source publication for every number the model produces.

## What it does

Given a table of materials (composition string, condition, phase, grain
size, test temperature, whatever mechanical data is available, all
columns optional except some way to identify the material), FTQS
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
too narrow for genuinely new alloys. FTQS assigns calibration
folds at the level of publication plus composition, which is the level
at which a new query is actually new. As far as we know no other
materials property tool does this.

Calibration is also done separately for the brittle and ductile phase
classes (a Mondrian split fixed on ductile-to-brittle-transition
physics, never tuned on results): pooled calibration would let
over-coverage on tough FCC alloys subsidize under-coverage on brittle
intermetallics. And the guarantee language is exact rather than
generous: the model card reports the real Barber et al. Theorem 4
floor, which at this sample size is about 0.71 for the nominal 90%
interval, not the often-quoted 0.80, states that group-level folding
extends the theorem heuristically, and backs the group-level claim
with held-out-publication evidence plus a provably valid subsampled
reference (one row per publication, Dunn et al. 2022).

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

## The data

The bundled dataset has 162 records (146 training, 16 held-out) built
from two committed sources by `python -m src.ingest_fan2023`:

- the full fracture toughness sheet of Fan et al., "Dataset for
  Fracture and Impact Toughness of High-Entropy Alloys", Scientific
  Data 10 (2023), fetched from Materials Cloud
  (doi:10.24435/materialscloud:d6-pf) and stored at
  `assets/fan2023_hea_toughness.xlsx`. All 148 records are used: 131
  K_IC, 17 K_Q, including the refractory NbTaTiZr-Mo and NbTaTiV
  series tested between 77 K and 226 K that the earlier CSV dropped.
  The measure type (K_IC, K_Q, or J-converted) and the specimen
  geometry class are model features; the geometry matters because 77
  of the 148 source records are indentation-derived toughness, a
  different measurand from fracture-mechanics specimens. Reported
  measurement uncertainties are carried as metadata.
- `assets/manual_records.csv`: 14 hand-collected records (steels from
  Ritchie 1976 and supplier datasheets, cryogenic CoCrNi and
  CoCrFeMnNi, WC-Co hardmetals) that are not in the source dataset.
- the same spreadsheet's Charpy sheets are ingested as companion
  assets (78 impact energy and 14 impact toughness records), not yet
  modeled.

The unseen split is pinned at the group level by
`assets/unseen_groups.json` and the ingest enforces zero
publication+composition overlap between the two sides, so the held-out
set really is new material systems. J values convert to K via
K = sqrt(J*E/(1-nu^2)) with nu = 0.3, only when the record reports a
modulus.

## How the numbers hold up

Selection-inclusive evidence: every held-out-group split re-runs the
model selection, so the numbers describe the full pipeline including
its data-driven choices. Measured on the bundled dataset (146 training
specimens from 83 publication/composition groups, toughness from 0.2
to 459 MPa m^0.5, test temperatures from 20 K to 1298 K):

| Quantity | Value |
| --- | --- |
| 90% interval coverage, held-out groups, pooled over 8 splits | 91.4% (95% CI 87 to 95) |
| 95% interval coverage, same protocol | 97.0% (95% CI 94 to 99) |
| Provable row-level floor at the 90% level (Thm 4, n=146, K=8) | 71% |
| Subsampled one-row-per-publication reference, 90% level | 95.3% coverage |
| Point MAE / R2 on held-out groups | 21.9 MPa m^0.5 / 0.58 |
| Pinned unseen set (16 rows, 11 disjoint groups), inside 90% bounds | 13 of 16 |

The conditional audit in the model card breaks coverage out by phase
bin, geometry class, measure type and temperature regime. Its main
finding right now: coverage below room temperature is 67% (CI 41 to
87, 18 rows from 5 publications), so sub-RT queries deserve extra
caution; everything else sits at or above nominal. Of the three unseen
misses, two are AlCoCrFeNi coatings whose measured 1.0 and 3.9
MPa m^0.5 sit in a cross-lab scatter band the literature itself spans,
and one is the cryogenic CoCrNi record (measured 459, the highest
value in the dataset) where the bound was conservative on the low
side.

On that scatter: replicates within one paper agree to 0.18 log-units,
but nominally identical compositions at the same temperature from
different labs differ by 0.47 log-units once processing varies. That
between-lab spread is irreducible at query time, it is recorded in
every model card, and it is the honest context for any complaint that
the intervals are wide. Every qualification run recomputes this whole
table for the current data, so the calibration claim is always backed
by the artifact in front of you, not by this README.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# or, as an installed package with the `ftqs` command (torch optional):
# pip install -e .
# or straight from GitHub:
# pip install git+https://github.com/Krataios14/FTQS

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
| `phase_bin` | which Mondrian calibration bin the row was routed to |
| `lower_90` / `upper_90` | conformal bounds; provable floor and measured coverage in the model card |
| `lower_95` / `upper_95` | same at the 95% level |
| `interval_unbounded` | set when n is too small for a finite bound at that level |
| `trust_score` | 0 to 100, distance-based, 100 is deep interpolation |
| `trust_tier` | A interpolation, B boundary, C extrapolation |
| `nearest_training_anchors` | closest measured specimens with citations |
| `test_value_score` (advise mode) | relative interval width times novelty |
| `min_distance_to_selected` (advise mode) | diversity spacing of the advised batch |

Treat tier C rows as unanswered questions, not as predictions. They are
the rows the advise mode will usually tell you to test first.

## Limitations, stated plainly

- 146 training points. The model interpolates a sparse literature
  corpus; it does not know mechanism. The conformal machinery is there
  precisely because the point model is weak.
- The provable coverage floor is 1 - 2*alpha minus a finite-sample
  excess (about 71% at the nominal 90% level here), under row
  exchangeability with equal folds; group folding extends it
  heuristically. The held-out-publication evidence and the subsampled
  reference are what support the group-level claim.
- The conditional audit currently shows under-coverage below room
  temperature (67%, wide CI). Treat sub-RT queries with extra caution
  until more cryogenic groups exist.
- Half the source records are indentation-derived toughness, not
  specimen K_IC; the geometry class feature and audit stratum carry
  that, but specimen-size validity criteria are taken as reported, not
  re-checked. J to K conversion assumes plane strain and nu = 0.3.
- Nominally identical compositions from different labs differ by 0.47
  log-units (up to roughly 10x in K) once processing varies. The model
  sees composition, condition, phase and processing text, and cannot
  resolve what those columns do not encode.
- Miedema enthalpies are model estimates, rounded, and pairs involving
  N, O, S and Pb are excluded (the coverage fraction is a feature, so
  the model can discount those rows).
- A query from a genuinely different population (a ceramic, a weld, an
  irradiated steel) gets a tier C flag, and its bounds mean little.
- Old pipeline entry points (`src.train`, `src.predict`, the segment
  scripts and the FT-Transformer) still work but do not produce
  intervals. Use the qualify/certify path for anything that matters.

The full methodology, with every formula checked against the
implementation, is in `docs/METHODS.md`.

## Repository layout

```
assets/            source dataset (xlsx), manual records, combined CSVs,
                   Charpy companion assets, pinned unseen groups
configs/           YAML configs
docs/METHODS.md    the methodology handbook, formula by formula
examples/          committed example outputs, incl. the HTML report
src/ingest_fan2023.py  rebuilds the combined CSVs from the sources
src/physics.py     composition and mechanics descriptors, Mondrian bin rule
src/conformal.py   group CV+, Mondrian bins, floors, multi-alpha intervals
src/applicability.py  trust scores, tiers, nearest-neighbour provenance
src/allowables.py  A-/B-basis tolerance bounds
src/qualify.py     qualification artifact, nested evidence, model card
src/certify.py     batch predictions, CSV + HTML report
src/screen.py      candidate screening, diversity-aware test advising
src/cli.py         umbrella CLI (installed as `ftqs`)
src/train.py       legacy point-estimate training (kept working)
tests/             pytest suite, runs the real pipeline end to end
```

## Tests

```bash
pytest -q
```

72 tests, including an end-to-end run of prepare/qualify/certify/screen
on the bundled dataset, the dataset ingest from the source spreadsheet,
and a subprocess test of the CLI artifact round trip. CI runs the suite
on every push.

## Citing

Cite the software via doi:10.5281/zenodo.20669425 (each release is
archived on Zenodo). When using the bundled data, also cite Fan et
al. (2023), first entry below. Code is MIT licensed; the dataset is
CC-BY 4.0, see NOTICE.

## References

- Fan, Chen, Steingrimsson, Xiong, Li, Liaw. Dataset for Fracture and
  Impact Toughness of High-Entropy Alloys. Scientific Data 10, 37
  (2023). Data: Materials Cloud, doi:10.24435/materialscloud:d6-pf.
- Barber, Candes, Ramdas, Tibshirani. Predictive inference with the
  jackknife+. Annals of Statistics 49(1), 2021.
- Dunn, Wasserman, Ramdas. Distribution-free prediction sets for
  two-layer hierarchical models. JASA, 2022.
- Takeuchi, Inoue. Classification of bulk metallic glasses by atomic
  size difference, heat of mixing and period of constituent elements.
  Materials Transactions 46(12), 2005.
- Yang, Zhang. Prediction of high-entropy stabilized solid-solution in
  multi-component alloys. Materials Chemistry and Physics 132, 2012.
- MMPDS-2023 / CMH-17 for the definition of A- and B-basis values.
