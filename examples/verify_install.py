#!/usr/bin/env python3
"""
Simple script to verify SIREN installation.
Run this after installing the package to ensure everything works.
"""

def verify_import():
    """Test that SIREN can be imported."""
    print("Testing imports...")
    try:
        import siren
        print(f"✓ Successfully imported siren (version {siren.__version__})")

        from siren import SIREN
        print("✓ Successfully imported SIREN class")

        from siren import (
            AntColonyOptimizer,
            SLSQPOptimizer,
            GeneticAlgorithmOptimizer,
            SimulatedAnnealingOptimizer
        )
        print("✓ Successfully imported all optimizer classes")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_initialization():
    """Test that SIREN can be initialized."""
    print("\nTesting SIREN initialization...")
    try:
        from siren import SIREN

        siren = SIREN()
        print("✓ Default SIREN initialization successful")

        siren_aco = SIREN.with_aco()
        print("✓ ACO optimizer initialization successful")

        siren_slsqp = SIREN.with_slsqp()
        print("✓ SLSQP optimizer initialization successful")

        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def verify_basic_functionality():
    """Test basic SIREN functionality."""
    print("\nTesting basic functionality...")
    try:
        from siren import SIREN

        # Simple test items
        items = [
            "I am confident.",
            "I trust myself.",
            "I believe in my abilities.",
            "I have support.",
            "I can rely on others.",
            "People help me."
        ]
        dimensions = ['A', 'A', 'A', 'B', 'B', 'B']

        siren = SIREN()
        result, sim_matrix = siren.reduce_scale(
            items,
            dimensions,
            items_per_dim=2,
            suppress_details=True,
            n_tries=1
        )

        assert 'A' in result, "Dimension A not in results"
        assert 'B' in result, "Dimension B not in results"
        assert len(result['A']['items']) == 2, f"Expected 2 items for A, got {len(result['A']['items'])}"
        assert len(result['B']['items']) == 2, f"Expected 2 items for B, got {len(result['B']['items'])}"

        print("✓ Basic scale reduction successful")
        print(f"  - Selected {len(result['A']['items'])} items for dimension A")
        print(f"  - Selected {len(result['B']['items'])} items for dimension B")

        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("SIREN Installation Verification")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", verify_import()))

    # Test initialization
    results.append(("Initialization", verify_initialization()))

    # Test basic functionality
    results.append(("Basic Functionality", verify_basic_functionality()))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! SIREN is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
