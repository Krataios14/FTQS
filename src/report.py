"""Self-contained HTML qualification report.

Single file, no external assets. Sections: batch summary, calibration
evidence from the model card, per-specimen predictions with conformal
bounds and trust tiers, nearest training anchors, model card details
and a usage disclaimer.
"""

from __future__ import annotations

import base64
import html
import io
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIER_COLORS = {"A": "#1a7f37", "B": "#b08800", "C": "#c0392b"}
TIER_LABELS = {
    "A": "interpolation",
    "B": "boundary",
    "C": "extrapolation",
}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _interval_figure(df: pd.DataFrame) -> Optional[str]:
    if "lower_90" not in df.columns:
        return None
    order = np.argsort(df["predicted_toughness_mpa_m0_5"].to_numpy())
    pred = df["predicted_toughness_mpa_m0_5"].to_numpy()[order]
    lo = df["lower_90"].to_numpy()[order]
    hi = df["upper_90"].to_numpy()[order]
    tiers = df["trust_tier"].to_numpy()[order]
    x = np.arange(len(pred))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for tier in ("A", "B", "C"):
        mask = tiers == tier
        if not mask.any():
            continue
        ax.errorbar(
            x[mask], pred[mask],
            yerr=[pred[mask] - lo[mask], hi[mask] - pred[mask]],
            fmt="o", ms=5, lw=1.2, capsize=3,
            color=TIER_COLORS[tier],
            label=f"Tier {tier} ({TIER_LABELS[tier]})",
        )
    if "measured_toughness_mpa_m0_5" in df.columns:
        meas = df["measured_toughness_mpa_m0_5"].to_numpy()[order]
        ax.scatter(x, meas, marker="x", s=40, color="#333", label="measured", zorder=5)
    ax.set_xlabel("specimen (sorted by prediction)")
    ax.set_ylabel("fracture toughness, MPa m$^{0.5}$")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle("Predictions with 90% conformal bounds", fontsize=11)
    return _fig_to_b64(fig)


def _calibration_figure(model_card: Dict) -> Optional[str]:
    evidence = model_card.get("calibration_evidence", {})
    per_alpha = evidence.get("per_alpha", {})
    if not per_alpha:
        return None
    entries = sorted(per_alpha.values(), key=lambda e: e["nominal_coverage"])
    nominal = [e["nominal_coverage"] for e in entries]
    empirical = [e["empirical_coverage"] for e in entries]
    ci_lo = [e["empirical_coverage"] - e["coverage_ci95"][0] for e in entries]
    ci_hi = [e["coverage_ci95"][1] - e["empirical_coverage"] for e in entries]
    floors = [e.get("provable_floor_rowlevel", 0.0) for e in entries]

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot([0.5, 1.0], [0.5, 1.0], color="#aab4be", lw=1, ls="--", label="nominal")
    ax.errorbar(
        nominal, empirical, yerr=[ci_lo, ci_hi], fmt="o", ms=5, capsize=3,
        color="#2c6fbb", label="empirical (held-out groups, 95% CI)",
    )
    ax.plot(nominal, floors, "v", ms=6, color="#c0392b", label="provable floor (row-level)")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("coverage")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.suptitle("Calibration spectrum, selection-inclusive", fontsize=11)
    return _fig_to_b64(fig)


def _importance_figure(model_card: Dict) -> Optional[str]:
    imp = model_card.get("permutation_importance", {})
    top = imp.get("top", [])
    if not top:
        return None
    names = [t[0] for t in top][:12][::-1]
    vals = [t[1] for t in top][:12][::-1]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.barh(names, vals, color="#3a6b8a")
    ax.set_xlabel("OOF MAE increase when permuted")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Out-of-fold permutation importance", fontsize=11)
    return _fig_to_b64(fig)


def _audit_table(model_card: Dict) -> str:
    evidence = model_card.get("calibration_evidence", {})
    audit = evidence.get("audit", {})
    if not audit:
        return ""
    alpha = evidence.get("audit_alpha", 0.1)
    rows = []
    for stratum, levels in audit.items():
        for level, e in levels.items():
            if e.get("insufficient"):
                cov = "insufficient data"
                ci = "&ndash;"
            else:
                cov = f"{e['coverage'] * 100:.0f}%"
                ci = f"[{e['coverage_ci95'][0] * 100:.0f}, {e['coverage_ci95'][1] * 100:.0f}]%"
            rows.append(
                f"<tr><td>{html.escape(stratum)}</td><td>{html.escape(str(level))}</td>"
                f"<td>{e['n']}</td><td>{e['n_groups']}</td><td>{cov}</td><td>{ci}</td></tr>"
            )
    return (
        f"<h2>Conditional coverage audit ({int((1 - alpha) * 100)}% intervals)</h2>"
        '<p style="font-size:12.5px;color:#445;">Coverage of the held-out-group evidence '
        "broken out by fixed specimen attributes. Rows within one publication are "
        "correlated, so the group counts are the better guide to information content. "
        "Strata with too few rows or groups are flagged rather than scored.</p>"
        "<table><thead><tr><th>stratum</th><th>level</th><th>rows</th><th>groups</th>"
        "<th>coverage</th><th>95% CI</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _tier_chip(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#666")
    label = TIER_LABELS.get(tier, "?")
    return (
        f'<span style="background:{color};color:#fff;padding:1px 8px;'
        f'border-radius:9px;font-size:11px;">{tier} &middot; {label}</span>'
    )


def _predictions_table(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "nearest_training_anchors"]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "trust_tier":
                cells.append(f"<td>{_tier_chip(str(v))}</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:,.2f}</td>" if pd.notna(v) else "<td>&ndash;</td>")
            else:
                s = html.escape(str(v)) if pd.notna(v) else "&ndash;"
                cells.append(f"<td>{s}</td>")
        anchor = row.get("nearest_training_anchors", "")
        anchor_html = html.escape(str(anchor))
        rows.append(
            f"<tr>{''.join(cells)}</tr>"
            f'<tr class="anchors"><td colspan="{len(cols)}">anchored by: {anchor_html}</td></tr>'
        )
    return (
        '<table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1c2733; margin: 0;
       background: #f4f6f8; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 24px; }
header { background: #16222e; color: #fff; padding: 26px 0; }
header .wrap { padding-top: 0; padding-bottom: 0; }
h1 { margin: 0; font-size: 22px; }
h1 small { color: #9fb3c8; font-weight: 400; font-size: 13px; display: block; margin-top: 4px; }
h2 { font-size: 16px; border-bottom: 2px solid #d7dee5; padding-bottom: 6px; margin-top: 34px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; background: #fff; }
th, td { border: 1px solid #dde4ea; padding: 5px 8px; text-align: left; }
th { background: #e8edf2; }
tr.anchors td { font-size: 11px; color: #5b6b7a; background: #fafbfc; }
.cards { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 14px; }
.card { background: #fff; border: 1px solid #dde4ea; border-radius: 8px;
        padding: 12px 18px; min-width: 140px; }
.card .num { font-size: 22px; font-weight: 600; }
.card .lbl { font-size: 11px; color: #5b6b7a; text-transform: uppercase; }
.figs { display: flex; gap: 18px; flex-wrap: wrap; }
.figs img { max-width: 100%; background: #fff; border: 1px solid #dde4ea;
            border-radius: 6px; padding: 6px; }
.disclaimer { background: #fff7e0; border: 1px solid #e6d59a; border-radius: 6px;
              padding: 12px 16px; font-size: 12.5px; margin-top: 30px; }
pre { background: #fff; border: 1px solid #dde4ea; border-radius: 6px;
      padding: 12px; font-size: 11.5px; overflow-x: auto; }
footer { color: #7a8a99; font-size: 11px; margin: 30px 0 10px; }
"""


def render_report(
    path: str,
    predictions: pd.DataFrame,
    model_card: Dict,
    neighbor_blocks: Optional[List[pd.DataFrame]] = None,
    title: str = "Fracture Toughness Qualification Report",
) -> str:
    n = len(predictions)
    tier_counts = predictions["trust_tier"].value_counts().to_dict()
    td = model_card.get("training_data", {})
    ev = model_card.get("calibration_evidence", {})
    ev90 = ev.get("per_alpha", {}).get("alpha_0.10", ev.get("alpha_0.10", {}))

    interval_b64 = _interval_figure(predictions)
    calib_b64 = _calibration_figure(model_card)
    imp_b64 = _importance_figure(model_card)

    cards = f"""
    <div class="cards">
      <div class="card"><div class="num">{n}</div><div class="lbl">specimens evaluated</div></div>
      <div class="card"><div class="num">{tier_counts.get('A', 0)}</div><div class="lbl">tier A (interpolation)</div></div>
      <div class="card"><div class="num">{tier_counts.get('B', 0)}</div><div class="lbl">tier B (boundary)</div></div>
      <div class="card"><div class="num">{tier_counts.get('C', 0)}</div><div class="lbl">tier C (extrapolation)</div></div>
      <div class="card"><div class="num">{ev90.get('empirical_coverage', float('nan')) * 100:.0f}%</div>
           <div class="lbl">empirical coverage of 90% bounds on held-out groups</div></div>
    </div>
    """

    figs = "<div class='figs'>"
    if interval_b64:
        figs += f'<img src="data:image/png;base64,{interval_b64}" alt="intervals"/>'
    if calib_b64:
        figs += f'<img src="data:image/png;base64,{calib_b64}" alt="calibration"/>'
    if imp_b64:
        figs += f'<img src="data:image/png;base64,{imp_b64}" alt="importance"/>'
    figs += "</div>"

    scatter = model_card.get("replicate_scatter", {})
    scatter_note = ""
    between = scatter.get("between_lab_any_condition", {})
    within = scatter.get("within_lab", {})
    if between and np.isfinite(between.get("std_log", float("nan"))):
        scatter_note = (
            '<p style="font-size:12.5px;color:#445;">Replicate scatter on this dataset: '
            f"within-lab repeatability {within.get('std_log', float('nan')):.2f} log-units "
            f"({within.get('n_replicate_sets', 0)} replicate sets) vs "
            f"{between.get('std_log', float('nan')):.2f} log-units between labs at matched "
            f"composition and temperature ({between.get('n_clusters', 0)} clusters, processing "
            "free to differ). The between-lab part is irreducible at query time: nominally "
            "identical alloys from different labs genuinely differ by this much.</p>"
        )

    selected = model_card.get("model_selection", {}).get("selected", "?")
    method = model_card.get("conformal", {}).get("method", "?")

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<header><div class="wrap">
<h1>{html.escape(title)}
<small>Fracture Toughness Qualification Suite (FTQS) &middot; base model: {html.escape(str(selected))} &middot; intervals: {html.escape(str(method))}
&middot; training set: {td.get('n_specimens', '?')} specimens / {td.get('n_groups', '?')} source groups
&middot; data fingerprint {html.escape(str(td.get('sha256_16', '')))}</small></h1>
</div></header>
<div class="wrap">

<h2>Batch summary</h2>
{cards}

<h2>Predictions and conformal bounds</h2>
{figs}
<p style="font-size:12.5px;color:#445;">Bounds are Mondrian group-aware CV+ conformal
intervals, calibrated separately for the brittle and ductile phase classes. The provable
row-level floor at each level is shown in the calibration figure (it is below the nominal
level by a finite-sample excess); group-level validity is supported by the held-out-group
evidence, which re-runs model selection inside every split. Tier C rows fall outside the
training distribution; their bounds should not be relied on.</p>
{scatter_note}
{_predictions_table(predictions)}

{_audit_table(model_card)}

<h2>Model card</h2>
<pre>{html.escape(__import__('json').dumps(model_card, indent=2))}</pre>

<div class="disclaimer"><b>Intended use.</b> {html.escape(str(model_card.get('intended_use', '')))}
</div>

<footer>Generated by the Fracture Toughness Qualification Suite (FTQS). Report is self-contained and suitable for archival.</footer>
</div></body></html>"""

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return path
