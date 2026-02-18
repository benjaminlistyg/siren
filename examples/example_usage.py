"""
Example usage of SIREN for psychometric scale reduction.

This script demonstrates how to use SIREN to reduce a multi-dimensional
psychological scale using Ant Colony Optimization.
"""

from siren import SIREN

def main():
    # Example scale items
    items = [
        # Dimension A: Self-confidence
        "I feel confident in my abilities.",
        "I believe I can achieve my goals.",
        "I trust my decision-making skills.",
        "I am capable of handling challenges.",
        "I feel positive about my potential.",
        "I often doubt myself.",  # reverse scored
        "I hesitate to take on new challenges.",  # reverse scored
        "My confidence grows with each success.",
        "I can overcome most obstacles.",
        "I second-guess my decisions frequently.",  # reverse scored

        # Dimension B: Social Support
        "I have people I can rely on.",
        "Others are there when I need help.",
        "I feel supported by my friends.",
        "My family is there for me.",
        "I have a strong support network.",
        "I often feel isolated.",  # reverse scored
        "People rarely understand my needs.",  # reverse scored
        "I can count on others in difficult times.",
        "My friends give me emotional support.",
        "I feel disconnected from others.",  # reverse scored

        # Dimension C: Stress Management
        "I handle stress well.",
        "I remain calm under pressure.",
        "I have effective coping strategies.",
        "Stressful situations overwhelm me.",  # reverse scored
        "I can maintain my composure in difficult times.",
        "I get anxious easily.",  # reverse scored
        "I know how to calm myself down.",
        "I panic when things go wrong.",  # reverse scored
        "I have techniques to manage my stress.",
        "Pressure makes me freeze up.",  # reverse scored

        # Dimension D: Goal Orientation
        "I set clear goals for myself.",
        "I work systematically towards my objectives.",
        "I have a clear vision for my future.",
        "I lack direction in life.",  # reverse scored
        "I break down big goals into smaller steps.",
        "I rarely plan ahead.",  # reverse scored
        "I stay focused on my long-term goals.",
        "I get distracted from my goals easily.",  # reverse scored
        "I track my progress towards goals.",
        "I struggle to maintain focus on my objectives."  # reverse scored
    ]

    # Dimension labels
    dimensions = ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10

    # Example 1: Using default ACO optimizer
    print("="*120)
    print("EXAMPLE 1: ANT COLONY OPTIMIZATION (DEFAULT)")
    print("="*120)

    siren = SIREN()  # ACO is the default

    print("\nRunning scale reduction with ACO...")
    result, _ = siren.reduce_scale(
        items,
        dimension_labels=dimensions,
        items_per_dim=3,
        suppress_details=True,
        n_tries=3  # Fewer tries for faster demo
    )

    siren.print_comparison(items, dimensions, result, items_per_dim=3, show_summary=False)

    # Example 2: Variable items per dimension
    print("\n\n" + "="*120)
    print("EXAMPLE 2: ACO WITH VARIABLE ITEMS PER DIMENSION")
    print("="*120)

    varying_items = {
        'A': 2,
        'B': 2,
        'C': 3,
        'D': 3
    }

    print("\nRunning ACO with variable items per dimension...")
    result_variable, _ = siren.reduce_scale(
        items,
        dimensions,
        items_per_dim=varying_items,
        suppress_details=True,
        n_tries=3
    )

    siren.print_comparison(items, dimensions, result_variable,
                          items_per_dim=varying_items, show_summary=True)

    # Example 3: Using different optimizers
    print("\n\n" + "="*120)
    print("EXAMPLE 3: COMPARING DIFFERENT OPTIMIZERS")
    print("="*120)

    print("\n--- SLSQP Optimizer ---")
    siren_slsqp = SIREN.with_slsqp()
    result_slsqp, _ = siren_slsqp.reduce_scale(
        items[:20],  # Use fewer items for faster demo
        dimensions[:20],
        items_per_dim=2,
        suppress_details=True,
        n_tries=2
    )
    print(f"Selected items for dimension A: {result_slsqp['A']['items']}")

    print("\n--- Genetic Algorithm Optimizer ---")
    siren_ga = SIREN.with_genetic_algorithm()
    result_ga, _ = siren_ga.reduce_scale(
        items[:20],
        dimensions[:20],
        items_per_dim=2,
        suppress_details=True,
        n_tries=2
    )
    print(f"Selected items for dimension A: {result_ga['A']['items']}")


if __name__ == "__main__":
    main()
