"""
Diagnostics, alternative selectors, structure discovery, and plots in SIREN.

Run with:  python examples/diagnostics_example.py
Requires the default sentence-transformer model (downloaded on first use)
and, for the plots, matplotlib (pip install siren-scale[plots]).
"""

import numpy as np

from siren import SIREN, coverage_percentile, redundancy, select_greedy_coverage


def main():
    items = [
        # Dimension A: Self-confidence
        "I feel confident in my abilities.",
        "I believe I can achieve my goals.",
        "I trust my decision-making skills.",
        "I am capable of handling challenges.",
        "I feel positive about my potential.",
        "I often doubt myself.",                    # reverse scored (index 5)
        "I hesitate to take on new challenges.",    # reverse scored (index 6)
        # Dimension B: Social support
        "I have people I can rely on.",
        "Others are there when I need help.",
        "I feel supported by my friends.",
        "My family is there for me.",
        "I have a strong support network.",
        "I feel alone when things go wrong.",       # reverse scored (index 12)
    ]
    dims = ["A"] * 7 + ["B"] * 6
    reverse = [5, 6, 12]

    # Embed once, reuse everywhere
    siren = SIREN(seed=0)
    E = siren.embed(items)
    S = siren.similarity_from_embeddings(E, reverse_scored=reverse)

    # 1. Optimized short form (coverage objective) from the precomputed matrix
    result, _ = siren.reduce_scale(items, dims, items_per_dim=3, similarity_matrix=S,
                                   objective="coverage", suppress_details=True)
    siren.print_comparison(items, dims, result, items_per_dim=3, show_summary=True,
                           reverse_scored=reverse)

    # 2. What did we give up? Dropped items, their proxies, and content gaps.
    audit = siren.coverage_report(items=items, gap_threshold=0.5)
    print("\nDropped items and their closest retained proxy:")
    print(audit["dropped"][["dimension", "item", "proxy_item", "proxy_similarity", "is_gap"]]
          .to_string(index=False))

    # 3. Is this better than a random stratified draw of the same size?
    fid = siren.fidelity_report(n_random=1000)
    print("\nFidelity table:")
    print(fid["table"].round(3).to_string(index=False))
    for stat, res in fid["percentiles"].items():
        print(f"  {stat:9s}: observed {res['observed']:.3f}, "
              f"percentile {res['percentile']:.0f} vs random subsets")

    # 4. The same audit applied to any other short form, e.g. a greedy baseline
    dim_idx = siren.get_dimension_indices(dims)
    greedy = select_greedy_coverage(S, dim_idx, 3)
    pct = coverage_percentile(S, dim_idx, greedy, n_random=1000, seed=0)
    print(f"\nGreedy baseline coverage percentile: {pct['percentile']:.0f}")

    # 5. Near-duplicate items in the full pool
    red = redundancy(S, threshold=0.75, items=items, dimension_labels=dims)
    print(f"\nRedundant pairs at |cos| >= .75: {len(red['pairs'])}")
    if len(red["pairs"]):
        print(red["pairs"][["item_i", "item_j", "similarity"]].to_string(index=False))

    # 6. No dimension key? Infer one from text.
    inferred = siren.infer_dimensions(embeddings=E)
    print(f"\nInferred structure: {inferred}")

    # 7. Vet new candidate items
    fit = siren.item_fit(["I can count on my colleagues.", "I feel confident in my abilities!"],
                         items, dims, embeddings=E)
    print("\nCandidate item fit:")
    print(fit[["candidate", "best_dimension", "best_dimension_similarity",
               "nearest_similarity", "is_redundant"]].round(3).to_string(index=False))

    # 8. Plots (optional)
    try:
        import matplotlib.pyplot as plt
        from siren import plot_item_map, plot_similarity_heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_item_map(embeddings=E, dimension_labels=dims, selected=result, items=items,
                      annotate=True, ax=axes[0])
        plot_similarity_heatmap(S, dimension_labels=dims, selected=result, items=items,
                                ax=axes[1])
        fig.tight_layout()
        fig.savefig("siren_diagnostics.png", dpi=150)
        print("\nSaved siren_diagnostics.png")
    except ImportError:
        print("\nmatplotlib not installed; skipping plots")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
