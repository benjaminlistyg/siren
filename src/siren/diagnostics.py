"""
Diagnostics for short forms.

Model-free audits that operate on a similarity matrix and an item selection:

- :func:`coverage_report` -- for every dropped item, its closest retained
  proxy and the similarity to it; flags content gaps.
- :func:`coverage_percentile` -- where a short form falls among random
  dimension-stratified subsets of the same per-dimension size.
- :func:`redundancy` -- near-duplicate item pairs and clusters.
- :func:`fidelity_report` -- one-table summary of coherence, coverage,
  between-dimension overlap, and null percentiles.

All functions accept ``selected`` either as a mapping of dimension name ->
list of item indices (the format returned by selectors) or as the full
results dictionary returned by :meth:`siren.SIREN.reduce_scale`.
"""

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .selection import select_random_stratified

Selection = Union[Dict[str, List[int]], Dict[str, Dict]]


def _as_selection(selected: Selection) -> Dict[str, List[int]]:
    """Accept either {dim: [indices]} or reduce_scale() output."""
    out = {}
    for dim, val in selected.items():
        if isinstance(val, dict):
            out[dim] = [int(i) for i in val["indices"]]
        else:
            out[dim] = [int(i) for i in val]
    return out


def _flat(selected: Dict[str, List[int]]) -> List[int]:
    return [i for idx in selected.values() for i in idx]


def dimension_indices(dimension_labels: Sequence[str]) -> Dict[str, List[int]]:
    """Map dimension name -> item indices, preserving first-appearance order."""
    out: Dict[str, List[int]] = {}
    for i, d in enumerate(dimension_labels):
        out.setdefault(d, []).append(i)
    return out


# ---------------------------------------------------------------------------
# Elementary statistics
# ---------------------------------------------------------------------------

def coverage_score(similarity_matrix: np.ndarray, dim_indices: Dict[str, List[int]],
                   selected: Selection) -> Dict[str, float]:
    """
    Per-dimension content coverage.

    For each dimension: mean over *dropped* items of the maximum absolute
    similarity to any retained item of that dimension. Equals 1.0 if nothing
    was dropped. This is the within-dimension term of the ``'coverage'``
    objective in :meth:`siren.SIREN.reduce_scale`.
    """
    S = np.abs(np.asarray(similarity_matrix))
    sel = _as_selection(selected)
    out = {}
    for dim, indices in dim_indices.items():
        kept = np.array(sel.get(dim, []), dtype=int)
        dropped = np.setdiff1d(np.array(indices), kept)
        if len(kept) == 0:
            out[dim] = float("nan")
        elif len(dropped) == 0:
            out[dim] = 1.0
        else:
            out[dim] = float(np.mean(S[np.ix_(dropped, kept)].max(axis=1)))
    return out


def coherence_score(similarity_matrix: np.ndarray, dim_indices: Dict[str, List[int]],
                    selected: Selection) -> Dict[str, float]:
    """
    Per-dimension coherence: mean pairwise absolute similarity among retained
    items. ``nan`` when fewer than two items are retained.
    """
    S = np.abs(np.asarray(similarity_matrix))
    sel = _as_selection(selected)
    out = {}
    for dim in dim_indices:
        kept = sel.get(dim, [])
        pairs = [S[i, j] for i, j in itertools.combinations(kept, 2)]
        out[dim] = float(np.mean(pairs)) if pairs else float("nan")
    return out


def between_score(similarity_matrix: np.ndarray, dim_indices: Dict[str, List[int]],
                  selected: Selection) -> Dict[str, float]:
    """
    Per-dimension between-dimension overlap: mean absolute similarity between
    the dimension's retained items and all retained items of other dimensions.
    """
    S = np.abs(np.asarray(similarity_matrix))
    sel = _as_selection(selected)
    out = {}
    for dim in dim_indices:
        kept = sel.get(dim, [])
        others = [i for d, idx in sel.items() if d != dim for i in idx]
        if kept and others:
            out[dim] = float(S[np.ix_(kept, others)].mean())
        else:
            out[dim] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def coverage_report(similarity_matrix: np.ndarray,
                    dim_indices: Dict[str, List[int]],
                    selected: Selection,
                    items: Optional[Sequence[str]] = None,
                    gap_threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
    """
    Audit what a short form gives up.

    For every dropped item, finds the retained item of the same dimension
    that is most similar to it (its *proxy*) and reports that similarity.
    Dropped items whose best proxy falls below ``gap_threshold`` are flagged
    as content gaps.

    Args:
        similarity_matrix: Pairwise item similarity matrix.
        dim_indices: Mapping of dimension name -> item indices.
        selected: Selected items ({dim: [indices]} or reduce_scale() output).
        items: Optional item texts, included in the tables when given.
        gap_threshold: Proxy similarity below which a dropped item is a gap.

    Returns:
        Dict with three DataFrames:
            - ``'dropped'``: one row per dropped item (dimension, index, text,
              proxy_index, proxy_text, proxy_similarity, is_gap)
            - ``'gaps'``: the subset of ``'dropped'`` with ``is_gap == True``
            - ``'summary'``: per-dimension n_kept, n_dropped, mean/min proxy
              similarity, n_gaps
    """
    S = np.abs(np.asarray(similarity_matrix))
    sel = _as_selection(selected)
    rows = []
    for dim, indices in dim_indices.items():
        kept = np.array(sel.get(dim, []), dtype=int)
        dropped = np.setdiff1d(np.array(indices), kept)
        for d in dropped:
            if len(kept) == 0:
                proxy, sim = -1, float("nan")
            else:
                j = int(np.argmax(S[d, kept]))
                proxy, sim = int(kept[j]), float(S[d, kept[j]])
            rows.append({
                "dimension": dim,
                "index": int(d),
                "item": items[d] if items is not None else None,
                "proxy_index": proxy,
                "proxy_item": items[proxy] if (items is not None and proxy >= 0) else None,
                "proxy_similarity": sim,
                "is_gap": bool(sim < gap_threshold) if not np.isnan(sim) else True,
            })
    cols = ["dimension", "index", "item", "proxy_index", "proxy_item",
            "proxy_similarity", "is_gap"]
    dropped_df = pd.DataFrame(rows, columns=cols)
    if items is None:
        dropped_df = dropped_df.drop(columns=["item", "proxy_item"])
    summary_rows = []
    for dim, indices in dim_indices.items():
        sub = dropped_df[dropped_df["dimension"] == dim]
        summary_rows.append({
            "dimension": dim,
            "n_pool": len(indices),
            "n_kept": len(sel.get(dim, [])),
            "n_dropped": len(sub),
            "mean_proxy_similarity": float(sub["proxy_similarity"].mean()) if len(sub) else 1.0,
            "min_proxy_similarity": float(sub["proxy_similarity"].min()) if len(sub) else 1.0,
            "n_gaps": int(sub["is_gap"].sum()) if len(sub) else 0,
        })
    return {
        "dropped": dropped_df,
        "gaps": dropped_df[dropped_df["is_gap"]].reset_index(drop=True),
        "summary": pd.DataFrame(summary_rows),
    }


def coverage_percentile(similarity_matrix: np.ndarray,
                        dim_indices: Dict[str, List[int]],
                        selected: Selection,
                        statistic: str = "coverage",
                        n_random: int = 1000,
                        seed: Optional[int] = None) -> Dict[str, float]:
    """
    Benchmark a short form against random dimension-stratified subsets.

    Draws ``n_random`` random subsets with the same number of items per
    dimension as ``selected`` and computes the chosen statistic (averaged
    over dimensions) for each. Reports where the observed form falls in that
    null distribution. A coverage percentile near 50 means the form is no more
    content-representative than a random stratified draw; the same audit can
    be applied to any proposed short form, including data-driven ones.

    Args:
        similarity_matrix: Pairwise item similarity matrix.
        dim_indices: Mapping of dimension name -> item indices.
        selected: Selected items ({dim: [indices]} or reduce_scale() output).
        statistic: ``'coverage'``, ``'coherence'``, or ``'between'``.
        n_random: Number of random subsets in the null distribution.
        seed: Seed for the random draws.

    Returns:
        Dict with ``observed``, ``percentile`` (0-100), ``null_mean``,
        ``null_sd``, ``z``, ``n_random``, ``statistic``.
    """
    funcs = {"coverage": coverage_score, "coherence": coherence_score,
             "between": between_score}
    if statistic not in funcs:
        raise ValueError(f"statistic must be one of {sorted(funcs)}, got {statistic!r}")
    fn = funcs[statistic]
    sel = _as_selection(selected)
    counts = {dim: len(sel.get(dim, [])) for dim in dim_indices}
    rng = np.random.default_rng(seed)

    def summary(selection):
        vals = np.array(list(fn(similarity_matrix, dim_indices, selection).values()), dtype=float)
        return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")

    observed = summary(sel)
    null = np.array([summary(select_random_stratified(dim_indices, counts, rng))
                     for _ in range(n_random)])
    # For 'between', lower is better: percentile counts null values >= observed.
    if statistic == "between":
        pct = 100.0 * np.mean(null >= observed)
    else:
        pct = 100.0 * np.mean(null <= observed)
    sd = float(np.nanstd(null, ddof=1)) if n_random > 1 else float("nan")
    return {
        "statistic": statistic,
        "observed": observed,
        "percentile": float(pct),
        "null_mean": float(np.nanmean(null)),
        "null_sd": sd,
        "z": float((observed - np.nanmean(null)) / sd) if sd and sd > 0 else float("nan"),
        "n_random": int(n_random),
    }


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------

def redundancy(similarity_matrix: np.ndarray,
               threshold: float = 0.8,
               items: Optional[Sequence[str]] = None,
               dimension_labels: Optional[Sequence[str]] = None,
               within_dimension_only: bool = False) -> Dict[str, object]:
    """
    Flag near-duplicate items.

    Item pairs whose absolute similarity is at or above ``threshold`` are
    flagged. Flagged pairs are grouped into clusters (connected components),
    and one item per cluster is suggested for retention: the item with the
    highest mean absolute similarity to all other items (the most central).

    Args:
        similarity_matrix: Pairwise item similarity matrix.
        threshold: Absolute similarity at or above which a pair is redundant.
        items: Optional item texts for the output tables.
        dimension_labels: Optional labels; when given, each pair is tagged
            with whether both items share a dimension.
        within_dimension_only: If True, only pairs within the same dimension
            are flagged (requires ``dimension_labels``).

    Returns:
        Dict with ``'pairs'`` (DataFrame: i, j, similarity, [same_dimension]),
        ``'clusters'`` (list of lists of item indices), and
        ``'suggest_remove'`` (list of item indices: all but the most central
        item in each cluster).
    """
    S = np.abs(np.asarray(similarity_matrix))
    n = S.shape[0]
    if within_dimension_only and dimension_labels is None:
        raise ValueError("within_dimension_only=True requires dimension_labels")
    rows = []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in itertools.combinations(range(n), 2):
        if S[i, j] < threshold:
            continue
        same = None
        if dimension_labels is not None:
            same = dimension_labels[i] == dimension_labels[j]
        if within_dimension_only and not same:
            continue
        row = {"i": i, "j": j, "similarity": float(S[i, j])}
        if items is not None:
            row["item_i"], row["item_j"] = items[i], items[j]
        if dimension_labels is not None:
            row["same_dimension"] = bool(same)
        rows.append(row)
        parent[find(i)] = find(j)

    if rows:
        pairs = pd.DataFrame(rows).sort_values("similarity", ascending=False)
        pairs = pairs.reset_index(drop=True)
    else:
        pairs = pd.DataFrame(columns=["i", "j", "similarity"])
    flagged = sorted({i for r in rows for i in (r["i"], r["j"])})
    groups: Dict[int, List[int]] = {}
    for i in flagged:
        groups.setdefault(find(i), []).append(i)
    clusters = sorted(groups.values(), key=lambda c: (-len(c), c))
    mean_sim = (S.sum(axis=1) - 1.0) / max(n - 1, 1)
    suggest_remove = []
    for c in clusters:
        keep = max(c, key=lambda i: mean_sim[i])
        suggest_remove.extend(i for i in c if i != keep)
    return {"pairs": pairs, "clusters": clusters, "suggest_remove": sorted(suggest_remove)}


# ---------------------------------------------------------------------------
# Fidelity report
# ---------------------------------------------------------------------------

def fidelity_report(similarity_matrix: np.ndarray,
                    dim_indices: Dict[str, List[int]],
                    selected: Selection,
                    n_random: int = 1000,
                    seed: Optional[int] = None,
                    statistics: Iterable[str] = ("coverage", "coherence")) -> Dict[str, object]:
    """
    One-table fidelity summary for a short form.

    Reports, per dimension and overall, the retained-item coherence, content
    coverage, and between-dimension overlap of a selection, plus the
    percentile of the overall coverage and coherence among random
    dimension-stratified subsets of the same size.

    Args:
        similarity_matrix: Pairwise item similarity matrix.
        dim_indices: Mapping of dimension name -> item indices.
        selected: Selected items ({dim: [indices]} or reduce_scale() output).
        n_random: Size of the null distribution (0 to skip percentiles).
        seed: Seed for the null draws.
        statistics: Which statistics to percentile-benchmark.

    Returns:
        Dict with ``'table'`` (DataFrame, one row per dimension plus
        ``'overall'``), ``'percentiles'`` (dict statistic -> result of
        :func:`coverage_percentile`), ``'n_items'`` and ``'n_pool'``.
    """
    sel = _as_selection(selected)
    cov = coverage_score(similarity_matrix, dim_indices, sel)
    coh = coherence_score(similarity_matrix, dim_indices, sel)
    bet = between_score(similarity_matrix, dim_indices, sel)
    rows = []
    for dim, indices in dim_indices.items():
        rows.append({"dimension": dim, "n_pool": len(indices), "n_kept": len(sel.get(dim, [])),
                     "coherence": coh[dim], "coverage": cov[dim], "between": bet[dim]})
    overall = {"dimension": "overall",
               "n_pool": sum(len(v) for v in dim_indices.values()),
               "n_kept": len(_flat(sel)),
               "coherence": float(np.nanmean(list(coh.values()))),
               "coverage": float(np.nanmean(list(cov.values()))),
               "between": float(np.nanmean(list(bet.values())))}
    rows.append(overall)
    table = pd.DataFrame(rows)
    percentiles = {}
    if n_random and n_random > 0:
        for stat in statistics:
            percentiles[stat] = coverage_percentile(similarity_matrix, dim_indices, sel,
                                                    statistic=stat, n_random=n_random, seed=seed)
    return {"table": table, "percentiles": percentiles,
            "n_items": overall["n_kept"], "n_pool": overall["n_pool"]}
