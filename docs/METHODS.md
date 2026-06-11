# FTQS Methods

This document describes how the Fracture Toughness Qualification Suite
builds its dataset, its features, its prediction intervals and its trust
scores, at the level of detail a reviewing stress engineer or auditor
needs before relying on the output. Every formula below was checked
against the source file it is attributed to. Where the code and this
document disagree, the code wins; please report the discrepancy.

Relevant source files: `src/ingest_fan2023.py`, `src/prepare_data.py`,
`src/physics.py`, `src/data.py`, `src/qualify.py`, `src/conformal.py`,
`src/applicability.py`, `src/allowables.py`, `src/certify.py`.

## 1. Data lineage

### 1.1 Sources

The bundled dataset is rebuilt deterministically by
`python -m src.ingest_fan2023` from two committed inputs:

- `assets/fan2023_hea_toughness.xlsx`: the "Fracture toughness" sheet of
  Fan, Chen, Steingrimsson, Xiong, Li, Liaw, "Dataset for Fracture and
  Impact Toughness of High-Entropy Alloys", Scientific Data 10 (2023),
  as published on Materials Cloud (doi:10.24435/materialscloud:d6-pf).
  The sheet has 148 rows and all 148 convert: 131 K_IC and 17 K_Q. The
  J-to-K conversion path (section 1.5) exists but contributes zero rows
  with the current spreadsheet, because every J-bearing row also
  carries a K_IC or K_Q value, which takes priority. Source rows with
  no usable toughness value would be skipped with their IDs printed;
  none are at present.
- `assets/manual_records.csv`: 14 hand-collected records outside the
  source dataset (steels from Ritchie 1976 and supplier datasheets,
  cryogenic CoCrNi and CoCrFeMnNi, WC-Co hardmetals). These are merged
  in unchanged; some rows deliberately omit the composition (the model
  then relies on the measured columns alone).

The merge keeps only rows with a finite toughness value, giving 162
records, split 147 training / 15 unseen (section 1.7).

### 1.2 Composition formula parsing

`ingest_fan2023.parse_formula` converts molar-ratio formula strings such
as `Al0.2CrFeNiTi0.2` into element ratios:

- An element token is a capital letter, an optional lowercase letter,
  and an optional number. A missing number means a ratio of 1.
- A parenthesized group with a multiplier, e.g. `(NbTaTiZr)95Mo5`,
  distributes the multiplier over the group members in proportion to
  their in-group ratios: each member gets `multiplier * r / sum(r)`.
  For `(NbTaTiZr)95Mo5` each of Nb, Ta, Ti, Zr receives 23.75 and Mo
  receives 5.
- A trailing parenthesized annotation that contains no digits, e.g.
  `(single crystalline)`, is stripped before parsing.
- Repeated elements accumulate.

The ratios are normalized and serialized as a dash-separated at.% string
(`Co33.33-Cr33.33-Ni33.33` style, elements sorted alphabetically, two
decimals), which is the composition format used everywhere downstream.

One typo in the source spreadsheet is fixed before parsing, recorded in
the `COMPOSITION_FIXES` table: `(NbTaTiZr)90M10` is read as
`(NbTaTiZr)90Mo10`. The row sits between the Mo5 and Mo20 rows of the
same Mo-substitution series in the source, so "M" is taken to mean Mo.
This is the only composition edit applied to the source data.

### 1.3 Value and uncertainty parsing

Numeric cells in the source sheet mix plain numbers, `value ± unc`
strings, parenthetical notes, and ranges. `parse_value_pm` handles all
of them and returns a `(value, uncertainty)` pair:

- a plain number gives `(value, NaN)`
- `5.8±0.2` gives `(5.8, 0.2)` (the `±` is normalized to `+-` first)
- parenthetical notes are dropped, so `295 (200K)` gives `(295, NaN)`
- a range `10-20` gives the midpoint with the half-width as the
  uncertainty: `(15, 5)`
- anything unparseable gives `(NaN, NaN)`

The toughness uncertainty is carried through the pipeline as metadata
(`Toughness_uncertainty_MPa_m0.5`, 43 of the 162 records have one). It
is never used as a model feature.

### 1.4 Hardness unit conversion

If a record reports Vickers hardness (HV) but no hardness in GPa, the
HV value is converted as `H_GPa = HV * 0.009807`. This is the unit
identity 1 HV = 1 kgf/mm^2 = 9.807 MPa. A reported GPa value always
takes precedence.

### 1.5 Toughness measure selection and J-to-K conversion

For each source row the toughness value is taken in this priority
order, and the choice is recorded in the `Toughness_measure` column:

1. K_IC, if finite (`KIC`)
2. else K_Q (`KQ`)
3. else J_IC converted to K, if a Young's modulus is reported (`KJIC`)
4. else J_Q converted to K, under the same condition (`KJQ`)
5. else the row is skipped

The J conversion assumes plane strain:

    K = sqrt(J * E / (1 - nu^2)),  nu = 0.3

with J in kJ/m^2 and E in GPa; the implementation converts to SI
(`E' = E * 1e9 / (1 - nu^2)` in Pa, `K = sqrt(J * 1e3 * E') / 1e6` in
MPa m^0.5) and returns NaN unless both J and E are finite and positive.
Converted values carry no uncertainty. The measure label is kept as a
categorical model feature, so the model can account for the difference
between a valid K_IC, a provisional K_Q and a J-derived value rather
than treating them as interchangeable. Specimen-size validity criteria
are taken as reported by the source papers and are not re-checked.

### 1.6 Manual records

`assets/manual_records.csv` uses the same column schema as the
converted sheet and is concatenated without modification. Its rows have
no `Toughness_measure` value (they count as "manual" in the ingest
summary). Each row carries a `Reference` string sufficient to locate
the source measurement.

### 1.7 The pinned unseen split

The 15-record unseen set is pinned by `assets/unseen_keys.json`, a list
of `{k, temp, ref_prefix}` keys. `split_unseen` matches each key to at
most one combined row: the first not-yet-used row whose toughness
rounds to `k` at two decimals, whose test temperature rounds to `temp`
at zero decimals, and whose reference string starts with the first 20
characters of `ref_prefix`. Matched rows become the unseen set, the
rest the training set. Pinning by value rather than by row index keeps
the split stable when the dataset is rebuilt or rows are reordered, so
evaluation numbers stay comparable across dataset revisions.

## 2. Featurization (src/physics.py)

`prepare_data` normalizes column names to lowercase snake_case, expands
the at.% composition string into per-element `elem_*` columns, and
calls `physics.add_physics_features`, which appends the `phys_*`
descriptors below. The composition parser at this stage
(`physics.parse_composition`) reads the dash-separated at.% format;
tokens without a numeric value are ignored.

### 2.1 Composition descriptors

All composition descriptors operate on the atomic fractions `c_i` of
the elements that appear in the 26-element property table
`ELEMENT_PROPS` (radius, Pauling electronegativity, VEC, melting point,
atomic mass, density), renormalized to sum to 1.

- `phys_smix_r`: ideal mixing entropy in units of the gas constant,
  `dS_mix / R = -sum(c_i * ln(c_i))`. Standard ideal-solution
  configurational entropy.
- `phys_dhmix`: Miedema-model mixing enthalpy in kJ/mol,
  `dH_mix = sum over pairs i<j of 4 * H_ij * c_i * c_j`, with the
  pairwise `H_ij` values tabulated after Takeuchi and Inoue, Materials
  Transactions 46 (2005). The tabulated values are rounded model
  estimates, not measurements. Pairs involving N, O, S and Pb are
  deliberately absent from the table because Miedema estimates are
  unreliable there.
- `phys_dhmix_coverage`: the fraction of total pair weight
  `sum(c_i * c_j)` whose `H_ij` is in the table. Pairs missing from the
  table are simply excluded from `dH_mix`, so this column tells the
  model (and the reader) how much of the enthalpy estimate is actually
  backed by a tabulated value. If no pair is covered, `dH_mix` is NaN.
  A single-element composition gets `dH_mix = 0` with coverage 1.
- `phys_delta_r`: atomic size mismatch in percent,
  `delta = 100 * sqrt(sum(c_i * (1 - r_i / r_bar)^2))` with
  `r_bar = sum(c_i * r_i)`. The standard HEA solid-solution descriptor.
- `phys_dchi`: electronegativity spread,
  `sqrt(sum(c_i * (chi_i - chi_bar)^2))`, Pauling scale.
- `phys_vec`: valence electron concentration, `sum(c_i * VEC_i)`.
- `phys_tm_mix`: composition-weighted melting point in K,
  `sum(c_i * Tm_i)`.
- `phys_omega`: the Yang-Zhang phase-stability parameter
  `Omega = Tm_mix * dS_mix / |dH_mix|` (Yang and Zhang, Materials
  Chemistry and Physics 132, 2012). The implementation computes
  `tm_mix * smix_r * 8.314e-3 / max(|dH_mix|, 1e-3)`, where the factor
  8.314e-3 kJ/(mol K) restores entropy units so the ratio is
  dimensionless, the 1e-3 floor on `|dH_mix|` avoids division by zero
  for near-ideal mixtures, and the result is capped at 1000. Omega is
  NaN whenever `dH_mix` is NaN.
- `phys_density_rom`: rule-of-mixtures density,
  `rho = sum(c_i * M_i) / sum(c_i * M_i / rho_i)`, i.e. total mass over
  total volume per mole of atoms, which is a harmonic mean of the
  elemental densities weighted by mass fraction. NaN if any constituent
  has no tabulated density (N, O).
- `phys_n_elements`, `phys_max_frac`, `phys_mass_mean`: element count,
  largest atomic fraction, and mean atomic mass. Bookkeeping
  descriptors, useful for separating dilute steels from equiatomic
  HEAs.

### 2.2 Mechanics-derived descriptors

Computed per row from the measured columns, where present:

- `phys_t_homologous`: homologous test temperature `T / Tm_mix`, with T
  the test temperature in K and Tm_mix from above.
- `phys_hall_petch`: the Hall-Petch term `1 / sqrt(d)` with d the grain
  size in um, defined only where d > 0.
- `phys_yield_strain`: elastic strain at yield, `YS / E` with YS in MPa
  and E converted to MPa (`E_GPa * 1000`), defined only where E > 0.
- `phys_strain_hardening`: strain-hardening capacity `(UTS - YS) / YS`,
  defined only where YS > 0.
- `phys_h_over_e`: the indentation index `H / E`, both in GPa, E > 0.
- `phys_h3_e2`: the indentation resistance index `H^3 / E^2`, E > 0.

Any infinities produced by these ratios are replaced with NaN.

### 2.3 Missing-value behavior

If a row has no parsable composition, or none of its elements are in
the property table, every composition descriptor is NaN. Mechanics
descriptors are NaN wherever an input column is missing or a guard
(d > 0, E > 0, YS > 0) fails. No imputation happens at this stage; NaNs
flow into the preprocessing step, which records them in missingness
masks and imputes with training medians (section 3). The model
therefore sees both an imputed value and the fact that it was imputed.

## 3. Preprocessing and feature matrix layout

### 3.1 Feature typing

`prepare_data` classifies columns: numerical features are the `elem_*`
and `phys_*` columns plus anything with a unit suffix (`_um`, `_g_cm3`,
`_gpa`, `_mpa`, `_percent`, `_k`); the provenance columns (`reference`,
`composition_at_percent`, `toughness_uncertainty_mpa_m0_5`) are
metadata and never become features; everything else (material
condition, phase, processing history, toughness measure) is
categorical. `qualify.prune_numeric_features` then drops numeric
columns that are all-NaN, observed in less than 5% of training rows
(`min_non_null_fraction: 0.05`), or, for `elem_*` columns, nonzero in
less than 2% of rows (`min_non_zero_fraction: 0.02`); thresholds come
from `configs/default.yaml`.

### 3.2 Numeric, mask and categorical pipelines (src/data.py)

- Numeric: NaNs are filled with the per-column training median, then
  scaled with sklearn's `RobustScaler` (center by median, scale by
  IQR), fit on the imputed training table. Robust statistics are used
  because the columns are heavy-tailed (toughness spans 0.2 to 459
  MPa m^0.5; grain sizes and strengths span orders of magnitude).
- Masks: for every numeric feature a binary indicator is kept, 1 where
  the value was observed and 0 where it was imputed (`transform_df`).
- Categoricals: each column is integer-coded per training vocabulary
  with `UNK = 0`; NaN maps to "UNK", and any category unseen at predict
  time maps to 0 as well (`CategoricalEncoder`).

`qualify.build_matrix` concatenates the three blocks into the single
matrix used by every downstream component (regressor, conformal
machinery, applicability domain):

    X = [ scaled numerics | missingness masks | categorical codes ]

in that order, all cast to float32.

### 3.3 Target transform and its inverses

With the default config the target is transformed as

    y_t = (log1p(max(y, 0)) - mean) / std

with mean and std computed on the transformed training targets
(`target_transform: log1p`, `target_standardize: true`). The log
compresses the three-orders-of-magnitude target range so residuals are
comparable across brittle hardmetals and tough cryogenic alloys.

Two inverse transforms exist, and the distinction matters:

- `data.inverse_transform_target` clips the de-standardized value into
  the training target range before applying `expm1`. It belongs to the
  legacy point-estimate pipeline (`src.train` / `src.predict`).
- `qualify.make_inverse_transform` is the inverse used by the
  qualification path for both point predictions and interval bounds.
  It de-standardizes and applies `expm1` with only a numerical
  overflow clamp at +/-700 in log space. It applies no training-range
  clipping, on purpose: clipping a conformal lower bound upward to the
  training minimum would assert a higher guaranteed floor than the
  conformal calibration supports. The lower bound is the decision
  quantity, so that direction of error is the dangerous one. (Clipping
  an upper bound downward would be equally invalid for coverage.)

Because the unclipped inverse is strictly monotone increasing, mapping
interval endpoints through it preserves coverage: y is inside [lo, hi]
in transformed space exactly when it is inside the mapped interval in
MPa m^0.5. The clipped inverse has no such property, which is the
second reason it stays out of the interval path.

One further adjustment happens in `certify.certify_dataframe`: lower
bounds are floored at 0 after inversion. Toughness is nonnegative, so
raising a negative lower bound to the physical floor of 0 cannot
exclude any feasible true value and does not affect coverage.

## 4. Conformal prediction (src/conformal.py)

### 4.1 Method

FTQS uses CV+ (cross-conformal), the K-fold generalization of the
jackknife+ of Barber, Candes, Ramdas and Tibshirani, "Predictive
inference with the jackknife+", Annals of Statistics 49(1), 2021,
implemented in `GroupCVPlus`. Fitting proceeds as follows: rows are
assigned to K folds (default K = 8, reduced if there are fewer groups,
never below 2); for each fold k a fresh regressor is trained on the
other folds and the held rows receive out-of-fold absolute residuals
`R_i = |y_i - mu_{-k(i)}(x_i)|`; a final model trained on everything
provides the point prediction. All of this happens in transformed
target space.

### 4.2 Group-level folding

Fold assignment (`_fold_assignment`) operates on groups, not rows: the
unique group labels are shuffled with a seeded RNG and dealt round-robin
to folds, so all rows of a group always share a fold. The group key
(`qualify.build_group_key`) is source publication plus composition
string (147 training rows form 90 groups), with a config fallback when
those columns are absent.

The reason is exchangeability. The conformal guarantee assumes the test
point is exchangeable with the calibration points. Literature-mined
rows are not exchangeable at the row level: specimens from one paper
share a lab, a melt, a test method and usually a composition, so a
random row split puts near-duplicates of the test row into the
calibration set, the residuals come out too small, and the intervals
are too narrow precisely for the case the tool exists for, a material
system nobody has tested. Folding at the publication-plus-composition
level restores exchangeability at the level at which a query is
actually new.

### 4.3 Interval construction

For a test point x, `predict_interval` forms, over all n training
points,

    v_lo_i = mu_{-k(i)}(x) - R_i
    v_hi_i = mu_{-k(i)}(x) + R_i

where `mu_{-k(i)}` is the fold model trained without sample i's fold.
The bounds are order statistics of these sets. In 1-based terms:

- lower bound: the `floor(alpha * (n + 1))`-th smallest of the v_lo
  values, never lower-indexed than the 1st;
- upper bound: the `ceil((1 - alpha) * (n + 1))`-th smallest of the
  v_hi values, capped at the n-th.

In the code (0-based): `i_lo = max(floor(alpha*(n+1)), 1) - 1` and
`i_hi = min(ceil((1-alpha)*(n+1)), n) - 1` into the sorted arrays. This
matches the CV+ interval definition in Barber et al.

### 4.4 The guarantee

The CV+ theorem of Barber et al. 2021 (their Theorem 4) gives
distribution-free finite-sample coverage of at least 1 - 2*alpha for
any regressor, minus a small additive term for K-fold CV+ that shrinks
as the per-fold sample count grows (the jackknife+, the K = n case,
carries no such term, their Theorem 1). FTQS quotes 1 - 2*alpha
as the working guarantee: the nominal 90% interval (alpha = 0.10) is
guaranteed at >= 80%, the nominal 95% interval (alpha = 0.05) at
>= 90%. In practice CV+ coverage tends to land near 1 - alpha, and the
empirical evidence below is what the model card actually reports, so
the claim never rests on the theorem alone. The guarantee is
conditional on group exchangeability: a query from a genuinely
different population is exactly what the applicability domain (section
5) is there to flag.

### 4.5 Model selection and held-out-group evaluation

`qualify` fits a `GroupCVPlus` for each candidate regressor
(HistGradientBoostingRegressor, ExtraTrees, RandomForest, Ridge) and
selects by mean out-of-fold absolute residual, i.e. group-CV MAE in
transformed space. The selection criterion is itself out-of-fold, but
note the model family is chosen on the full training set; only the
calibration evidence below involves data the fitted model never saw.

`evaluate_group_coverage` produces that evidence: for each of 8
repetitions it draws 20% of the unique groups (at least one) as a test
set, fits a fresh `GroupCVPlus` on the remaining groups, and measures
empirical interval coverage, mean width, MAE and (pooled over
repetitions) R2 on the held-out groups, all reported in MPa m^0.5 via
the monotone inverse. This measures the deployment condition: whole
material systems absent from training. The resulting table is written
into `model_card.json` on every qualification run, so the calibration
claim is always backed by the artifact at hand rather than by static
documentation.

## 5. Applicability domain (src/applicability.py)

The conformal guarantee says nothing about a query that is not
exchangeable with the training groups. The `TrustModel` makes
extrapolation visible. It operates on the same feature matrix X the
regressor sees.

- Standardization: NaNs are filled with the per-column training
  nanmedian; columns are centered by the median of the filled training
  data and scaled by the IQR, falling back to the standard deviation
  and then to 1.0 for degenerate columns.
- Calibration: a kNN structure (k = 5, or n - 1 if smaller) is fit on
  the standardized training rows. Each training row's self-distance is
  the mean Euclidean distance to its k nearest other training rows
  (the zero distance to itself is dropped). The sorted self-distances
  form the reference distribution.
- Scoring: a query's distance is its mean distance to its k nearest
  training rows. Its percentile within the reference distribution
  gives `trust_score = 100 - percentile` (clipped to [0, 100]) and the
  tier:
  - Tier A (interpolation): at or below the 80th percentile of
    training self-distances.
  - Tier B (boundary): between the 80th and 97.5th percentile.
  - Tier C (extrapolation): beyond the 97.5th percentile. The model is
    guessing, and the exchangeability assumption behind the conformal
    bounds is weak here. Treat tier C rows as unanswered questions.
- Provenance: for every query the k nearest training specimens are
  returned with their metadata (composition, reference, condition,
  phase, processing history, test temperature, measured toughness),
  and `certify` serializes them into the `nearest_training_anchors`
  column. Every prediction can therefore be traced during review to
  the measured datapoints that anchor it.

## 6. Design allowables (src/allowables.py)

MMPDS defines the B-basis allowable as the value exceeded by 90% of the
population with 95% confidence, and A-basis as 99%/95%
(`BASIS_DEFINITIONS`). `src/allowables.py` computes one-sided lower
tolerance bounds from measured samples by two methods.

Normal-theory bound: `value = x_bar - k * s`, with s the sample
standard deviation (ddof = 1) and the exact one-sided tolerance factor
from the noncentral t distribution:

    k = nct.ppf(conf, df = n - 1, nc = z_cov * sqrt(n)) / sqrt(n)

where `z_cov = Phi^-1(coverage)` is the standard normal quantile of the
coverage level. This is the textbook exact factor; it assumes the
sample is normal, which fracture toughness data often is not, so the
nonparametric bound exists as a check.

Distribution-free bound: the r-th smallest sample value is a valid
lower tolerance bound when the probability that at least the required
proportion of the population exceeds it reaches the confidence level,
which reduces to the Beta cdf condition

    Beta_cdf(1 - coverage; r, n - r + 1) >= confidence.

`nonparametric_rank` returns the largest 1-based rank r satisfying it
(the left side decreases in r, so the scan stops at the first failure),
and the bound is that order statistic. The condition with r = 1 gives
the minimum sample sizes: 1 - 0.90^n >= 0.95 requires n >= 29 for
B-basis, and 1 - 0.99^n >= 0.95 requires n >= 299 for A-basis. Below
those sizes no distribution-free bound exists and the function returns
None.

`basis_summary` reports both methods for both bases. Intended use:
compute screening allowables from your own physical test results and
compare them against the model's conformal lower bounds. The model
output is a screening value for test planning and material
down-selection. It is not a substitute for ASTM E399/E1820 testing or
for MMPDS/CMH-17 qualification, and both the model card and every page
of the HTML report say so.

## 7. Intended use and limitations

FTQS is a screening tool. The intended workflow is: rank candidate
materials by the conformal lower bound (not the point estimate), use
the trust tier to decide which bounds to take seriously, use the advise
mode to decide which physical tests buy the most information, then
test. Decisions about structure go through measured data and the
allowables machinery, not through the model.

Limitations, consistent with the README:

- 147 training points. The model interpolates a sparse literature
  corpus and does not know mechanism. The conformal machinery exists
  precisely because the point model is weak (held-out-group R2 is
  about 0.27 on the bundled data).
- The toughness measure type (K_IC vs K_Q vs J-converted) is a feature,
  but specimen-size validity is taken as reported by the source
  papers. The J-to-K conversion assumes plane strain and nu = 0.3.
- Nominally identical compositions from different labs differ by up to
  10x in measured toughness. The model sees composition, condition,
  phase and processing text; it cannot resolve what those columns do
  not encode. The bundled unseen-set misses are exactly of this kind.
- Miedema enthalpies are rounded model estimates; pairs involving N,
  O, S and Pb are excluded, and `phys_dhmix_coverage` lets the model
  discount low-coverage rows.
- The coverage guarantee is conditional on group exchangeability. A
  ceramic, a weld or an irradiated steel gets a tier C flag and its
  bounds mean little.
- The legacy entry points (`src.train`, `src.predict`, the segment
  scripts) produce point estimates without intervals. Anything that
  matters should go through the qualify/certify path.

## References

- Fan, Chen, Steingrimsson, Xiong, Li, Liaw. Dataset for Fracture and
  Impact Toughness of High-Entropy Alloys. Scientific Data 10, 37
  (2023). Data: Materials Cloud, doi:10.24435/materialscloud:d6-pf.
- Barber, Candes, Ramdas, Tibshirani. Predictive inference with the
  jackknife+. Annals of Statistics 49(1), 2021.
- Takeuchi, Inoue. Classification of bulk metallic glasses by atomic
  size difference, heat of mixing and period of constituent elements.
  Materials Transactions 46(12), 2005.
- Yang, Zhang. Prediction of high-entropy stabilized solid-solution in
  multi-component alloys. Materials Chemistry and Physics 132, 2012.
- MMPDS-2023 / CMH-17 for the definition of A- and B-basis values.
- Ritchie, R. O. (1976), source of several manual steel records; full
  citations are carried per record in the `Reference` column.
