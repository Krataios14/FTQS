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
    if not evidence:
        return None
    labels, nominal, empirical = [], [], []
    for key, ev in sorted(evidence.items()):
        labels.append(f"{int(round(ev['nominal_coverage'] * 100))}% interval")
        nominal.append(ev["nominal_coverage"])
        empirical.append(ev["empirical_coverage"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    w = 0.35
    ax.bar(x - w / 2, nominal, w, label="nominal", color="#7f8c9b")
    ax.bar(x + w / 2, empirical, w, label="empirical (held-out groups)", color="#2c6fbb")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Calibration on unseen material systems", fontsize=11)
    return _fig_to_b64(fig)


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
    ev90 = ev.get("alpha_0.10", {})

    interval_b64 = _interval_figure(predictions)
    calib_b64 = _calibration_figure(model_card)

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
    figs += "</div>"

    selected = model_card.get("model_selection", {}).get("selected", "?")
    method = model_card.get("conformal", {}).get("method", "?")

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<header><div class="wrap">
<h1>{html.escape(title)}
<small>ToughCert &middot; base model: {html.escape(str(selected))} &middot; intervals: {html.escape(str(method))}
&middot; training set: {td.get('n_specimens', '?')} specimens / {td.get('n_groups', '?')} source groups
&middot; data fingerprint {html.escape(str(td.get('sha256_16', '')))}</small></h1>
</div></header>
<div class="wrap">

<h2>Batch summary</h2>
{cards}

<h2>Predictions and conformal bounds</h2>
{figs}
<p style="font-size:12.5px;color:#445;">Bounds are group-aware CV+ conformal intervals.
At the 90% level they carry a finite-sample coverage guarantee of at least 80%
under group exchangeability, with empirical held-out-group coverage shown above.
Tier C rows fall outside the training distribution; their bounds should not be relied on.</p>
{_predictions_table(predictions)}

<h2>Model card</h2>
<pre>{html.escape(__import__('json').dumps(model_card, indent=2))}</pre>

<div class="disclaimer"><b>Intended use.</b> {html.escape(str(model_card.get('intended_use', '')))}
</div>

<footer>Generated by ToughCert. Report is self-contained and suitable for archival.</footer>
</div></body></html>"""

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return path
