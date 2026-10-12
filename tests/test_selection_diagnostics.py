"""
Model-free tests for selectors, diagnostics, plots, and the precomputed-input
path of SIREN.reduce_scale(). None of these load a sentence-transformer.
"""

import numpy as np
import pandas as pd
import pytest

from siren import (
    SIREN,
    AntColonyOptimizer,
    select_centroid,
    select_greedy_coverage,
    select_random_stratified,
    coverage_score,
    coherence_score,
    between_score,
    coverage_report,
    coverage_percentile,
    redundancy,
    fidelity_report,
)


def _toy_matrix():
    """8 items, two dimensions. In A, items 0-2 are tight and item 3 is an
    oddball; in B, items 4-7 are moderately similar. Items 0 and 1 are
    near-duplicates."""
    S = np.eye(8)
    for i in range(4):
        for j in range(4):
            if i != j:
                S[i, j] = 0.6
    S[0, 1] = S[1, 0] = 0.95
    S[3, :3] = S[:3, 3] = 0.1
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                S[i, j] = 0.5
    S[4, 5] = S[5, 4] = 0.7
    return S


DIM = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
LABELS = ["A"] * 4 + ["B"] * 4
ITEMS = [f"item {i}" for i in range(8)]


def test_select_centroid_keeps_central_items():
    sel = select_centroid(_toy_matrix(), DIM, 2)
    assert set(sel["A"]) == {0, 1}          # the near-duplicates are most central
    assert 3 not in sel["A"]
    assert len(sel["B"]) == 2


def test_select_greedy_coverage_avoids_redundancy():
    sel = select_greedy_coverage(_toy_matrix(), DIM, 2)
    assert len(sel["A"]) == 2
    assert not ({0, 1} <= set(sel["A"]))    # does not keep both near-duplicates
    # coverage of greedy selection >= coverage of centroid selection in A
    cov_g = coverage_score(_toy_matrix(), DIM, sel)["A"]
    cov_c = coverage_score(_toy_matrix(), DIM, select_centroid(_toy_matrix(), DIM, 2))["A"]
    assert cov_g >= cov_c


def test_select_random_stratified_respects_counts_and_seed():
    rng = np.random.default_rng(1)
    a = select_random_stratified(DIM, {"A": 2, "B": 1}, rng)
    assert len(a["A"]) == 2 and len(a["B"]) == 1
    b = select_random_stratified(DIM, {"A": 2, "B": 1}, np.random.default_rng(1))
    assert a == b


def test_selectors_validate_targets():
    with pytest.raises(ValueError):
        select_centroid(_toy_matrix(), DIM, 5)
    with pytest.raises(ValueError):
        select_greedy_coverage(_toy_matrix(), DIM, {"A": 1})


def test_elementary_scores():
    S = _toy_matrix()
    sel = {"A": [0, 2], "B": [4, 6]}
    cov = coverage_score(S, DIM, sel)
    assert cov["A"] == pytest.approx(np.mean([max(S[1, 0], S[1, 2]), max(S[3, 0], S[3, 2])]))
    coh = coherence_score(S, DIM, sel)
    assert coh["A"] == pytest.approx(S[0, 2])
    assert np.isnan(coherence_score(S, DIM, {"A": [0], "B": [4]})["A"])
    bet = between_score(S, DIM, sel)
    assert bet["A"] == pytest.approx(np.mean([S[0, 4], S[0, 6], S[2, 4], S[2, 6]]))
    # full selection => coverage 1
    assert coverage_score(S, DIM, DIM)["A"] == 1.0


def test_coverage_report_flags_gaps():
    S = _toy_matrix()
    rep = coverage_report(S, DIM, {"A": [0, 1], "B": [4, 5]}, items=ITEMS, gap_threshold=0.5)
    assert set(rep) == {"dropped", "gaps", "summary"}
    dropped = rep["dropped"]
    assert len(dropped) == 4
    odd = dropped[dropped["index"] == 3].iloc[0]
    assert odd["is_gap"] and odd["proxy_similarity"] == pytest.approx(0.1)
    assert odd["proxy_item"] in ("item 0", "item 1")
    assert list(rep["gaps"]["index"]) == [3]
    summary = rep["summary"].set_index("dimension")
    assert summary.loc["A", "n_gaps"] == 1 and summary.loc["B", "n_gaps"] == 0
    assert summary.loc["A", "n_kept"] == 2 and summary.loc["A", "n_dropped"] == 2


def test_coverage_report_accepts_reduce_scale_output():
    S = _toy_matrix()
    result = {"A": {"indices": [0, 2], "items": [], "metrics": {}},
              "B": {"indices": [4, 6], "items": [], "metrics": {}}}
    rep = coverage_report(S, DIM, result)
    assert len(rep["dropped"]) == 4 and "item" not in rep["dropped"].columns


def test_coverage_percentile_ranks_good_selection_high():
    S = _toy_matrix()
    good = select_greedy_coverage(S, DIM, 2)
    bad = {"A": [0, 1], "B": [4, 5]}  # redundant pairs
    pg = coverage_percentile(S, DIM, good, n_random=300, seed=0)
    pb = coverage_percentile(S, DIM, bad, n_random=300, seed=0)
    assert pg["percentile"] > pb["percentile"]
    assert 0 <= pg["percentile"] <= 100
    assert pg["n_random"] == 300 and pg["statistic"] == "coverage"
    # reproducible
    assert coverage_percentile(S, DIM, good, n_random=300, seed=0) == pg
    # coherence percentile: the redundant pair should rank high
    pc = coverage_percentile(S, DIM, bad, statistic="coherence", n_random=300, seed=0)
    assert pc["percentile"] >= 90
    with pytest.raises(ValueError):
        coverage_percentile(S, DIM, good, statistic="nonsense")


def test_redundancy_clusters_and_suggestion():
    S = _toy_matrix()
    red = redundancy(S, threshold=0.9, items=ITEMS, dimension_labels=LABELS)
    assert len(red["pairs"]) == 1
    assert set(red["pairs"].loc[0, ["i", "j"]]) == {0, 1}
    assert bool(red["pairs"].loc[0, "same_dimension"]) is True
    assert red["clusters"] == [[0, 1]]
    assert len(red["suggest_remove"]) == 1 and red["suggest_remove"][0] in (0, 1)
    # lower threshold: more pairs, connected component grows
    red2 = redundancy(S, threshold=0.6)
    assert len(red2["pairs"]) > 1 and len(red2["clusters"][0]) >= 3
    # within-dimension-only filter
    red3 = redundancy(S, threshold=0.0, dimension_labels=LABELS, within_dimension_only=True)
    assert red3["pairs"]["same_dimension"].all()
    with pytest.raises(ValueError):
        redundancy(S, within_dimension_only=True)
    empty = redundancy(S, threshold=0.99)
    assert len(empty["pairs"]) == 0 and empty["clusters"] == [] and empty["suggest_remove"] == []


def test_fidelity_report_table():
    S = _toy_matrix()
    rep = fidelity_report(S, DIM, {"A": [0, 2], "B": [4, 6]}, n_random=100, seed=0)
    table = rep["table"].set_index("dimension")
    assert list(table.index) == ["A", "B", "overall"]
    assert table.loc["overall", "n_kept"] == 4 and table.loc["overall", "n_pool"] == 8
    assert set(rep["percentiles"]) == {"coverage", "coherence"}
    assert rep["n_items"] == 4
    rep0 = fidelity_report(S, DIM, {"A": [0, 2], "B": [4, 6]}, n_random=0)
    assert rep0["percentiles"] == {}


# ---------------------------------------------------------------------------
# SIREN without a model: precomputed inputs, methods, seed, options
# ---------------------------------------------------------------------------

def test_reduce_scale_with_similarity_matrix_and_methods():
    S = _toy_matrix()
    siren = SIREN(model_name=None, seed=0)
    for method in ("centroid", "greedy", "optimize"):
        result, df = siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, similarity_matrix=S,
                                        objective="coverage", method=method,
                                        suppress_details=True, n_tries=2)
        assert set(result) == {"A", "B"}
        assert all(len(v["indices"]) == 2 for v in result.values())
        assert "coverage" in result["A"]["metrics"]
        assert df.shape == (8, 8)
        assert siren.last_selection_ == {d: v["indices"] for d, v in result.items()}
    assert siren.last_dimension_labels_ == LABELS


def test_reduce_scale_optimize_matches_greedy_on_toy():
    """ACO with the coverage objective should find the greedy-optimal A pair
    (or better) on this small problem."""
    S = _toy_matrix()
    siren = SIREN(model_name=None, seed=123)
    result, _ = siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, similarity_matrix=S,
                                   objective="coverage", suppress_details=True, n_tries=3)
    cov_opt = coverage_score(S, DIM, result)["A"]
    cov_greedy = coverage_score(S, DIM, select_greedy_coverage(S, DIM, 2))["A"]
    assert cov_opt >= cov_greedy - 1e-9


def test_reduce_scale_seed_reproducible():
    S = _toy_matrix() + np.random.default_rng(5).normal(0, 0.02, (8, 8))
    S = np.clip((S + S.T) / 2, -1, 1)
    np.fill_diagonal(S, 1.0)
    siren = SIREN(model_name=None)
    r1, _ = siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, similarity_matrix=S,
                               suppress_details=True, n_tries=2, seed=7)
    r2, _ = siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, similarity_matrix=S,
                               suppress_details=True, n_tries=2, seed=7)
    assert {d: v["indices"] for d, v in r1.items()} == {d: v["indices"] for d, v in r2.items()}


def test_reduce_scale_with_embeddings_and_reverse():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(8, 16))
    E[1] = -E[0]  # item 1 is a reversed paraphrase of item 0
    siren = SIREN(model_name=None, seed=0)
    result, df = siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, embeddings=E,
                                    reverse_scored=[1], method="greedy", suppress_details=True)
    # after flipping, items 0 and 1 are identical (similarity 1) so greedy
    # coverage should not keep both
    assert df.values[0, 1] == pytest.approx(1.0)
    assert not ({0, 1} <= set(result["A"]["indices"]))
    assert siren.last_embeddings_.shape == (8, 16)


def test_optimizer_options_forwarded(monkeypatch):
    seen = {}

    class SpyACO(AntColonyOptimizer):
        def optimize(self, objective_function, x0, args, constraints, bounds, **kwargs):
            seen.update(kwargs.get("options", {}))
            return super().optimize(objective_function, x0, args, constraints, bounds, **kwargs)

    siren = SIREN(model_name=None, optimizer=SpyACO(), seed=1,
                  optimizer_options={"n_ants": 5, "maxiter": 3})
    siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, similarity_matrix=_toy_matrix(),
                       suppress_details=True, n_tries=1, optimizer_options={"maxiter": 2})
    assert seen["n_ants"] == 5 and seen["maxiter"] == 2 and seen["seed"] == 1


def test_reduce_scale_validates_inputs():
    siren = SIREN(model_name=None)
    with pytest.raises(ValueError):
        siren.reduce_scale(ITEMS, LABELS, similarity_matrix=np.eye(7), suppress_details=True)
    with pytest.raises(ValueError):
        siren.reduce_scale(ITEMS, LABELS, similarity_matrix=_toy_matrix(), method="magic",
                           suppress_details=True)
    with pytest.raises(ValueError):
        siren.reduce_scale(ITEMS, LABELS[:-1], similarity_matrix=_toy_matrix(),
                           suppress_details=True)
    with pytest.raises(ValueError):
        siren.reduce_scale(ITEMS, LABELS, items_per_dim={"A": 1}, similarity_matrix=_toy_matrix(),
                           suppress_details=True)
    with pytest.raises(ValueError):
        # no model configured and no precomputed input
        siren.reduce_scale(ITEMS, LABELS, suppress_details=True)


def test_infer_dimensions_from_embeddings():
    rng = np.random.default_rng(0)
    a = rng.normal(size=16); b = rng.normal(size=16)
    E = np.vstack([a + rng.normal(0, 0.1, 16) for _ in range(4)] +
                  [b + rng.normal(0, 0.1, 16) for _ in range(4)])
    siren = SIREN(model_name=None, seed=0)
    labels = siren.infer_dimensions(embeddings=E)
    assert len(set(labels)) == 2
    assert labels[:4] == [labels[0]] * 4 and labels[4:] == [labels[4]] * 4
    assert labels[0] == "D1"  # renumbered by first appearance
    labels3 = siren.infer_dimensions(embeddings=E, n_dimensions=3)
    assert len(set(labels3)) == 3
    # reduce_scale with no labels infers them
    result, _ = siren.reduce_scale(ITEMS, None, items_per_dim=1, embeddings=E,
                                   method="greedy", suppress_details=True)
    assert set(result) == {"D1", "D2"}
    assert siren.last_dimension_labels_ == labels
    with pytest.raises(ValueError):
        siren.infer_dimensions(embeddings=E, n_dimensions=0)


def test_item_fit_with_precomputed_embeddings():
    rng = np.random.default_rng(0)
    a = rng.normal(size=16); b = rng.normal(size=16)
    E = np.vstack([a + rng.normal(0, 0.1, 16) for _ in range(4)] +
                  [b + rng.normal(0, 0.1, 16) for _ in range(4)])
    C = np.vstack([E[0], b + rng.normal(0, 0.3, 16)])
    siren = SIREN(model_name=None)
    df = siren.item_fit(["dup of item 0", "new B item"], ITEMS, LABELS,
                        embeddings=E, candidate_embeddings=C, redundancy_threshold=0.95)
    assert isinstance(df, pd.DataFrame) and len(df) == 2
    assert df.loc[0, "best_dimension"] == "A" and bool(df.loc[0, "is_redundant"]) is True
    assert df.loc[0, "nearest_index"] == 0
    assert df.loc[1, "best_dimension"] == "B" and bool(df.loc[1, "is_redundant"]) is False
    assert {"sim_A", "sim_B"} <= set(df.columns)


def test_siren_report_wrappers():
    S = _toy_matrix()
    siren = SIREN(model_name=None, seed=0)
    with pytest.raises(ValueError):
        siren.fidelity_report()
    siren.reduce_scale(ITEMS, LABELS, items_per_dim=2, similarity_matrix=S, method="greedy",
                       suppress_details=True)
    fid = siren.fidelity_report(n_random=50)
    assert "table" in fid and fid["n_items"] == 4
    cov = siren.coverage_report(items=ITEMS)
    assert len(cov["dropped"]) == 4


def test_plots_run_without_display():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from siren import plot_item_map, plot_similarity_heatmap
    rng = np.random.default_rng(0)
    E = rng.normal(size=(8, 16))
    sel = {"A": [0, 2], "B": [4, 6]}
    ax = plot_item_map(embeddings=E, dimension_labels=LABELS, selected=sel, items=ITEMS,
                       annotate=True)
    assert ax is not None
    ax2 = plot_item_map(similarity_matrix=_toy_matrix(), dimension_labels=LABELS, method="mds")
    assert ax2 is not None
    ax3 = plot_similarity_heatmap(_toy_matrix(), dimension_labels=LABELS, selected=sel,
                                  items=ITEMS)
    assert ax3 is not None
    with pytest.raises(ValueError):
        plot_item_map(similarity_matrix=_toy_matrix(), method="pca")
