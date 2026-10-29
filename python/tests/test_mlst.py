import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drydenlyap import (gramians, hankel_sv, h2_norm, hinf_norm, modal_table,
                        stability_shift, sigma_max)


def second_order(wn=2.0, z=0.2):
    A = np.array([[0, 1], [-wn ** 2, -2 * z * wn]])
    B = np.array([[0], [wn ** 2]])
    C = np.array([[1.0, 0]])
    return A, B, C


def test_hinf_resonant_peak():
    wn, z = 2.0, 0.2
    A, B, C = second_order(wn, z)
    peak, wpk, _ = hinf_norm(A, B, C, band=(1e-2, 1e2))
    assert peak == pytest.approx(1 / (2 * z * np.sqrt(1 - z ** 2)), rel=1e-5)
    assert wpk == pytest.approx(wn * np.sqrt(1 - 2 * z ** 2), rel=1e-3)


def test_hinf_band_restricts_search():
    A, B, C = second_order(2.0, 0.2)
    peak, wpk, _ = hinf_norm(A, B, C, band=(5.0, 50.0))
    assert wpk == pytest.approx(5.0, rel=1e-3)
    assert peak == pytest.approx(sigma_max(A, B, C, [5.0])[0], rel=1e-6)


def test_h2_matches_frequency_integral():
    A, B, C = second_order(1.5, 0.3)
    h2, _ = h2_norm(A, B, C)
    w = np.linspace(0, 200, 40001)
    s = sigma_max(A, B, C, w)
    assert h2 == pytest.approx(np.sqrt(np.trapezoid(s ** 2, w) / np.pi), rel=1e-3)


def test_first_order_gramians_and_hsv():
    a, b, c = 2.0, 3.0, 0.5
    Wc, Wo, eps = gramians([[-a]], [[b]], [[c]])
    assert eps == 0
    assert Wc[0, 0] == pytest.approx(b * b / (2 * a))
    assert Wo[0, 0] == pytest.approx(c * c / (2 * a))
    assert hankel_sv(Wc, Wo)[0] == pytest.approx(abs(b * c) / (2 * a))


def test_modal_table_diagonal_system():
    A = np.diag([-1.0, -4.0]); B = np.array([[1.0], [2.0]]); C = np.array([[1.0, 1.0]])
    rows, _ = modal_table(A, B, C, state_names=["x1", "x2"])
    assert [r["dominant_state"] for r in rows] == ["x1", "x2"]
    assert rows[0]["wn"] == pytest.approx(1.0) and rows[1]["wn"] == pytest.approx(4.0)
    assert max(r["wc_norm"] for r in rows) == pytest.approx(1.0)


def test_modal_table_merges_complex_pair():
    A, B, C = second_order(2.0, 0.2)
    rows, _ = modal_table(A, B, C)
    assert len(rows) == 1 and rows[0]["oscillatory"]
    assert rows[0]["zeta"] == pytest.approx(0.2) and rows[0]["wn"] == pytest.approx(2.0)


def test_marginal_mode_gets_shift():
    A = np.array([[0.0, 1.0], [0.0, -1.0]])        # integrator + stable pole
    eps = stability_shift(A)
    assert eps > 0
    Wc, Wo, e2 = gramians(A, [[0.0], [1.0]], [[1.0, 0.0]])
    assert e2 == pytest.approx(eps) and np.all(np.isfinite(Wc))
