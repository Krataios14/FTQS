# PLAN

## Where this project is

The original goal (a tabular model that predicts fracture toughness
from composition and processing data) is done and kept working under
`src.train` / `src.predict`. The project has since been rebuilt around
qualification rather than prediction: conservative bounds, trust
scoring, provenance and test planning. See README for the full picture.

## Current architecture

1. Featurization (`src.prepare_data`, `src.physics`)
   - composition strings -> element fractions + physics descriptors
   - reference and raw composition kept as provenance metadata
2. Qualification (`src.qualify`)
   - base regressor selected by out-of-fold group-CV MAE
   - group-aware CV+ conformal ensemble (folds by publication+composition)
   - kNN applicability domain calibrated on training self-distances
   - held-out-group calibration evidence written to the model card
3. Certification (`src.certify`, `src.report`)
   - predictions with 90/95% bounds, trust tiers, nearest anchors
   - self-contained HTML report for design review
4. Screening (`src.screen`)
   - candidate generation on the at.% simplex, ranked by lower bound
   - advise mode: rank physical tests by expected information value
5. Allowables (`src.allowables`)
   - A-/B-basis tolerance bounds for measured test data

## Known gaps / next steps

- Mondrian (per-tier) conformal calibration, so tier B/C rows get their
  own wider quantiles instead of one pooled residual distribution.
- Separate the toughness measure type (K_IC vs K_Q vs K_JIC) into a
  category once the source data records it reliably.
- Grow the dataset. Every added publication group improves both the
  model and the strength of the calibration evidence.
- Optional: conformalized quantile regression (CQR) for adaptive
  interval widths once there is enough data to fit quantile models.

## Constraints

- Runs on CPU in seconds, well under 16 GB RAM.
- Deterministic for a fixed seed and config.
- No claims beyond what the held-out-group evidence in the model card
  supports.
