# PLAN

## Where this project is

The original goal (a tabular model that predicts fracture toughness
from composition and processing data) is done and kept working under
`src.train` / `src.predict`. The project has since been rebuilt around
qualification rather than prediction: conservative bounds with honestly
stated guarantees, trust scoring, provenance and test planning. See
README.md for the user view and docs/METHODS.md for the methodology.

## Current architecture

1. Featurization (`src.prepare_data`, `src.physics`)
   - composition strings -> element fractions + physics descriptors
   - reference, composition and measurement uncertainty kept as
     provenance metadata; specimen geometry class and toughness
     measure type kept as features
2. Qualification (`src.qualify`)
   - base regressor selected by median OOF group-CV MAE across several
     fold seeds (gbdt, extra_trees, blend, random_forest, ridge)
   - pre-committed Mondrian conformal model: per-bin group-aware CV+
     for the brittle and ductile phase classes, pooled fallback
   - selection-inclusive nested calibration evidence with exact
     Clopper-Pearson bands and Theorem-4 provable floors
   - conditional coverage audit on fixed strata at alpha = 0.10
   - subsampled one-row-per-publication reference (Dunn et al. 2022)
   - replicate-scatter decomposition (within-lab vs between-lab)
   - out-of-fold permutation importance with paired value/mask columns
   - kNN applicability domain calibrated on training self-distances
3. Certification (`src.certify`, `src.report`)
   - predictions with multi-alpha bounds, phase-bin routing, trust
     tiers, nearest anchors, unbounded-interval flags
   - self-contained HTML report: calibration spectrum, audit table,
     importance chart, replicate-scatter statement, model card
4. Screening (`src.screen`)
   - candidate generation on the at.% simplex, ranked by lower bound
   - advise mode: diversity-aware test batch selection
5. Allowables (`src.allowables`)
   - A-/B-basis tolerance bounds for measured test data
6. Packaging: installable via pyproject (ftqs CLI), CI on GitHub
   Actions, torch optional.

## Known gaps / next steps

- The Charpy companion assets (78 impact energy + 14 impact toughness
  records) are ingested but not modeled; a companion impact model or a
  Charpy correlation feature is the obvious next use.
- Per-bin Mondrian uses one fixed physical split (brittle/ductile).
  When the dataset grows past ~300 rows, a geometry-class split
  (indentation vs specimen) inside each phase bin becomes feasible.
- A residual-based continuous difficulty model (tree-leaf proximity,
  winsorized) was reviewed and deliberately deferred: at n=147 the
  simulated variant produced heavy-tailed scores and physically absurd
  upper bounds. Revisit above ~500 rows.
- Grow the dataset: NIMS data sheets and the DTIC plane strain K_IC
  handbook (AD-773673) exist for steels but are scanned PDFs and need
  careful manual extraction, not scraping.

## Constraints

- Runs on CPU in minutes, well under 16 GB RAM.
- Deterministic for a fixed seed and config.
- No claims beyond what the held-out-group evidence in the model card
  supports; guarantee language follows Barber et al. (2021) Theorem 4
  with its finite-sample excess stated, not rounded away.
