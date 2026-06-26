from __future__ import annotations

import numpy as np

from grading_calibration.statistics.fdr import benjamini_hochberg, benjamini_yekutieli


def legacy_bh_adjust(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = pvals[idx] * m / rank
        if val > prev:
            val = prev
        prev = val
        adj[idx] = min(val, 1.0)
    return adj


def legacy_by_adjust(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    m = len(pvals)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = pvals[idx] * m * c_m / rank
        if val > prev:
            val = prev
        prev = val
        adj[idx] = min(val, 1.0)
    return adj


def assert_matches_legacy(pvals: list[float]) -> None:
    np.testing.assert_allclose(
        benjamini_hochberg(pvals),
        legacy_bh_adjust(pvals),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        benjamini_yekutieli(pvals),
        legacy_by_adjust(pvals),
        rtol=0,
        atol=1e-12,
    )


def test_fdr_empty_input_matches_legacy() -> None:
    assert benjamini_hochberg([]) == []
    assert benjamini_yekutieli([]) == []


def test_fdr_single_value_matches_legacy() -> None:
    assert_matches_legacy([0.2])


def test_fdr_tied_values_keep_legacy_stable_order() -> None:
    assert_matches_legacy([0.04, 0.01, 0.01, 0.2, 0.04])


def test_fdr_production_like_values_match_legacy() -> None:
    assert_matches_legacy([0.0, 1.0, 0.5, 1e-9, 0.03, 0.03, 0.8])


def test_utils_benjamini_hochberg_delegates_to_centralized_behavior() -> None:
    from utils.statistics import benjamini_hochberg as utils_bh

    pvals = [0.04, 0.01, 0.01, 0.2, 0.04]
    adjusted, significant = utils_bh(np.asarray(pvals, dtype=float), alpha=0.05)
    np.testing.assert_allclose(
        adjusted,
        benjamini_hochberg(pvals),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        significant,
        np.asarray(benjamini_hochberg(pvals), dtype=float) < 0.05,
    )


def test_significant_at_boundary_and_nonfinite_values() -> None:
    from grading_calibration.statistics.fdr import significant_at

    assert significant_at(0.05, 0.05)
    assert not significant_at(0.05, 0.05, inclusive=False)
    assert not significant_at(float("nan"), 0.05)
    assert not significant_at(float("inf"), 0.05)
    assert not significant_at("not-a-pvalue", 0.05)


def test_parse_pvalue_and_sentinel_detection() -> None:
    from grading_calibration.statistics.fdr import is_na_sentinel, parse_pvalue

    assert is_na_sentinel(-2147483648.0)
    assert is_na_sentinel(-53687091200.0)
    assert not is_na_sentinel(-1e-20)
    assert np.isnan(parse_pvalue("-2147483648"))
    assert np.isnan(parse_pvalue(""))
    assert np.isnan(parse_pvalue("not-a-pvalue"))
    assert parse_pvalue("-1e-300") == 0.0
    assert parse_pvalue("1.25") == 1.0
    assert parse_pvalue("0.031") == 0.031


def test_nan_aware_fdr_excludes_sentinel_from_family_size() -> None:
    from grading_calibration.statistics.fdr import (
        benjamini_hochberg_excluding_nan,
        parse_pvalue,
        significant_at,
    )

    parsed = [parse_pvalue(v) for v in ["0.01", "-2147483648", "0.02"]]
    adjusted = benjamini_hochberg_excluding_nan(parsed)

    np.testing.assert_allclose([adjusted[0], adjusted[2]], [0.02, 0.02], rtol=0, atol=1e-12)
    assert np.isnan(adjusted[1])
    assert not significant_at(adjusted[1], 0.05)


def test_utils_benjamini_hochberg_inclusive_flag_is_opt_in() -> None:
    from utils.statistics import benjamini_hochberg as utils_bh

    pvals = np.asarray([0.05], dtype=float)
    _, strict = utils_bh(pvals, alpha=0.05)
    _, inclusive = utils_bh(pvals, alpha=0.05, inclusive=True)

    assert strict.tolist() == [False]
    assert inclusive.tolist() == [True]
