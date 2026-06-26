"""False-discovery-rate adjustment helpers.

The default BH/BY functions preserve the legacy post-processing scripts' ordering:
ties keep the input order because Python's ``sorted`` is stable. Phase-6 canonical
correction helpers are additive and must be opted into by callers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

NA_SENTINEL_THRESHOLD = -2_000_000_000.0


def is_na_sentinel(value: object) -> bool:
    """Return True for R INT_MIN-style NA sentinels and propagated multiples."""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric <= NA_SENTINEL_THRESHOLD


def parse_pvalue(value: object) -> float:
    """Parse a p-value using canonical Phase-6 semantics.

    R NA sentinels become NaN and are excluded by NaN-aware FDR helpers. Legacy
    legitimate-underflow behavior is retained for non-sentinel negatives, which
    are clamped to 0.0. Values above 1.0 are clamped to 1.0.
    """
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        pval = float(text)
    except ValueError:
        return float("nan")
    if not math.isfinite(pval):
        return float("nan")
    if is_na_sentinel(pval):
        return float("nan")
    if pval < 0:
        return 0.0
    if pval > 1:
        return 1.0
    return pval


def significant_at(adjusted: object, alpha: float, *, inclusive: bool = True) -> bool:
    """Return whether an adjusted p-value is significant at alpha.

    Non-finite adjusted p-values are never significant. Inclusive mode uses the
    standard BH ``<=`` operator; strict mode preserves legacy ``<`` call sites.
    """
    try:
        adjusted_p = float(adjusted)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if not math.isfinite(adjusted_p):
        return False
    return adjusted_p <= alpha if inclusive else adjusted_p < alpha


def benjamini_hochberg(pvals: Sequence[float]) -> list[float]:
    """Return BH-adjusted p-values using the legacy stable tie ordering."""
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


def benjamini_yekutieli(pvals: Sequence[float]) -> list[float]:
    """Return BY-adjusted p-values using the conservative script behavior."""
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


def adjust_excluding_nan(
    pvals: Sequence[float],
    adjuster: Callable[[Sequence[float]], list[float]] = benjamini_hochberg,
) -> list[float]:
    """Adjust finite p-values only, reinserting NaN at excluded positions."""
    adjusted = [float("nan")] * len(pvals)
    finite_items = [(idx, float(pval)) for idx, pval in enumerate(pvals) if math.isfinite(pval)]
    if not finite_items:
        return adjusted
    finite_adjusted = adjuster([pval for _, pval in finite_items])
    for (idx, _), adj in zip(finite_items, finite_adjusted):
        adjusted[idx] = adj
    return adjusted


def benjamini_hochberg_excluding_nan(pvals: Sequence[float]) -> list[float]:
    """BH-adjust p-values after excluding NaNs from the FDR family size."""
    return adjust_excluding_nan(pvals, benjamini_hochberg)


def benjamini_yekutieli_excluding_nan(pvals: Sequence[float]) -> list[float]:
    """BY-adjust p-values after excluding NaNs from the FDR family size."""
    return adjust_excluding_nan(pvals, benjamini_yekutieli)
