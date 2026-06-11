from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest_fan2023 import (
    convert_fracture_sheet,
    convert_impact_energy_sheet,
    convert_impact_toughness_sheet,
    geometry_class,
    j_to_k,
    parse_formula,
    parse_value_pm,
    split_unseen_groups,
    to_at_percent_string,
)
from src.physics import parse_composition

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "assets" / "fan2023_hea_toughness.xlsx"


def test_parse_formula_equimolar_default():
    r = parse_formula("CoCrFeMnNi")
    assert r == {"Co": 1.0, "Cr": 1.0, "Fe": 1.0, "Mn": 1.0, "Ni": 1.0}


def test_parse_formula_decimal_ratios():
    r = parse_formula("Al0.2CrFeNiTi0.2")
    assert r["Al"] == pytest.approx(0.2)
    assert r["Cr"] == 1.0
    assert r["Ti"] == pytest.approx(0.2)


def test_parse_formula_group_multiplier():
    r = parse_formula("(NbTaTiZr)95Mo5")
    assert r["Nb"] == pytest.approx(23.75)
    assert r["Mo"] == pytest.approx(5.0)
    s = to_at_percent_string(r)
    total = sum(parse_composition(s).values())
    assert total == pytest.approx(100.0, abs=0.05)


def test_parse_formula_source_typo_fixed():
    r = parse_formula("(NbTaTiZr)90M10")
    assert "Mo" in r and r["Mo"] == pytest.approx(10.0)
    assert "M" not in r


def test_parse_formula_strips_annotation():
    r = parse_formula("MoNbTaW (single crystalline)")
    assert set(r) == {"Mo", "Nb", "Ta", "W"}


def test_parse_value_pm():
    assert parse_value_pm("5.8±0.2") == (pytest.approx(5.8), pytest.approx(0.2))
    assert parse_value_pm(7.0)[0] == 7.0
    assert np.isnan(parse_value_pm(7.0)[1])
    assert parse_value_pm("295 (200K)")[0] == pytest.approx(295.0)
    v, u = parse_value_pm("10-20")
    assert v == pytest.approx(15.0) and u == pytest.approx(5.0)
    assert np.isnan(parse_value_pm("n/a")[0])


def test_j_to_k_plane_strain():
    # J = 100 kJ/m2, E = 200 GPa, nu = 0.3 -> ~148 MPa m^0.5
    assert j_to_k(100.0, 200.0) == pytest.approx(148.25, abs=0.5)
    assert np.isnan(j_to_k(np.nan, 200.0))
    assert np.isnan(j_to_k(100.0, np.nan))


def test_convert_real_sheet():
    df = convert_fracture_sheet(str(XLSX))
    # 148 source records, a handful have no usable toughness value
    assert len(df) >= 140
    measures = df["Toughness_measure"].value_counts()
    assert measures["KIC"] >= 125
    assert measures["KQ"] >= 12
    # every composition parses back to ~100 at.%
    for comp in df["Composition (at. %)"]:
        total = sum(parse_composition(comp).values())
        assert total == pytest.approx(100.0, abs=0.2)
    # all toughness values positive and finite
    k = df["Fracture_toughness_MPa_m0.5"]
    assert np.isfinite(k).all() and (k > 0).all()
    # the refractory KQ series is present (this was missing before)
    kq = df[df["Toughness_measure"] == "KQ"]
    assert (kq["Testing_temperature_K"] < 250).any()


def test_split_unseen_groups_moves_whole_groups():
    df = pd.DataFrame(
        {
            "Composition (at. %)": ["Fe50.00-Ni50.00"] * 3 + ["Al100.00"],
            "Fracture_toughness_MPa_m0.5": [10.0, 20.0, 30.0, 40.0],
            "Reference": ["Paper B", "Paper B", "Paper C", "Paper C"],
        }
    )
    from src.ingest_fan2023 import group_key

    pinned = [{"key": group_key("Paper B", "Fe50.00-Ni50.00")}]
    train, unseen = split_unseen_groups(df, pinned)
    # Both Paper B rows of that composition move together
    assert len(unseen) == 2
    assert (unseen["Reference"] == "Paper B").all()
    assert len(train) == 2


def test_geometry_class():
    assert geometry_class("indentation") == "indentation"
    assert geometry_class("SENB") == "bend"
    assert geometry_class("SEVNB") == "bend"
    assert geometry_class("C(T)") == "compact_tension"
    assert geometry_class("CT") == "compact_tension"
    assert geometry_class("cantilever beams") == "cantilever"
    assert geometry_class(float("nan")) == "unknown"


def test_bundled_split_is_group_disjoint():
    from src.ingest_fan2023 import group_key

    tr = pd.read_csv(ROOT / "assets" / "combined_fracture_training.csv")
    un = pd.read_csv(ROOT / "assets" / "combined_fracture_unseen.csv")
    kt = {group_key(r, c) for r, c in zip(tr["Reference"], tr["Composition (at. %)"])}
    ku = {group_key(r, c) for r, c in zip(un["Reference"], un["Composition (at. %)"])}
    assert not (kt & ku)
    assert len(un) >= 14
    # Geometry class is recorded for source-derived rows
    assert "Test_geometry" in tr.columns
    assert (tr["Test_geometry"] == "indentation").sum() >= 60


def test_convert_impact_energy_real_sheet():
    df = convert_impact_energy_sheet(str(XLSX))
    # 78 source records, all carry a Charpy energy value
    assert len(df) >= 70
    for comp in df["Composition (at. %)"]:
        total = sum(parse_composition(comp).values())
        assert total == pytest.approx(100.0, abs=0.2)
    e = df["Impact_energy_J"]
    assert np.isfinite(e).all() and (e > 0).all()
    t = df["Testing_temperature_K"]
    assert ((t >= 4.0) & (t <= 1700.0)).all()


def test_convert_impact_toughness_real_sheet():
    df = convert_impact_toughness_sheet(str(XLSX))
    # 14 source records, reported in kJ/m2
    assert len(df) >= 12
    for comp in df["Composition (at. %)"]:
        total = sum(parse_composition(comp).values())
        assert total == pytest.approx(100.0, abs=0.2)
    it = df["Impact_toughness_kJ_m2"]
    assert np.isfinite(it).all() and (it > 0).all()
    t = df["Testing_temperature_K"]
    assert ((t >= 4.0) & (t <= 1700.0)).all()
