"""
Example: Using the Comprehensive Statistical Analysis Framework

This script demonstrates how to use the new statistical analysis framework
for LLM calibration research.

Usage:
    python example_comprehensive_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def example_basic_metrics_with_ci():
    """Example: Compute metrics with confidence intervals."""
    print("=" * 80)
    print("Example 1: Metrics with Confidence Intervals")
    print("=" * 80)

    from grading_calibration.metrics import expected_calibration_error, bootstrap_ci, MetricResult
    from grading_calibration.domain.models import PredictionResult

    # Create some example predictions (normally you'd load these from CSV)
    predictions = [
        PredictionResult(
            instance_id=f"inst_{i}",
            prediction=1 if i % 2 == 0 else 0,
            raw_confidence=0.6 + (i % 10) * 0.04,
            ground_truth=1 if i % 2 == 0 else 0
        )
        for i in range(100)
    ]

    # Compute ECE
    ece_value = expected_calibration_error(predictions)

    # Compute bootstrap CI
    ece_ci = bootstrap_ci(
        predictions,
        lambda p: expected_calibration_error(p),
        n_bootstrap=1000
    )

    # Create MetricResult container
    ece_result = MetricResult(
        name="ece",
        value=ece_value,
        n_samples=len(predictions),
        confidence_interval=ece_ci
    )

    print(f"\n{ece_result.summary_string()}")
    print("\nThis shows ECE with 95% confidence interval!")


def example_hypothesis_testing():
    """Example: Statistical comparison with hypothesis testing."""
    print("\n" + "=" * 80)
    print("Example 2: Hypothesis Testing")
    print("=" * 80)

    from grading_calibration.metrics import HypothesisTester
    from grading_calibration.domain.models import PredictionResult
    import numpy as np

    # Simulate raw vs calibrated predictions
    np.random.seed(42)

    predictions_raw = [
        PredictionResult(
            instance_id=f"inst_{i}",
            prediction=np.random.choice([0, 1]),
            raw_confidence=np.random.uniform(0.3, 0.9),
            ground_truth=np.random.choice([0, 1])
        )
        for i in range(100)
    ]

    predictions_cal = [
        PredictionResult(
            instance_id=f"inst_{i}",
            prediction=p.prediction,
            raw_confidence=p.raw_confidence * 0.9,  # Simulated calibrated
            ground_truth=p.ground_truth
        )
        for i, p in enumerate(predictions_raw)
    ]

    # Run hypothesis test
    tester = HypothesisTester(alpha=0.05)

    result = tester.paired_t_test(
        predictions_raw,
        predictions_cal,
        metric_func=lambda p: p.raw_confidence
    )

    print(f"\nComparing raw vs calibrated confidence:")
    print(f"  {result.interpretation()}")
    print(f"  Test statistic: {result.statistic:.3f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Effect size (Cohen's d): {result.effect_size:.3f}")


def example_feature_analysis():
    """Example: Feature extraction and analysis."""
    print("\n" + "=" * 80)
    print("Example 3: Feature Analysis")
    print("=" * 80)

    from analysis import FeatureExtractor, FeatureAnalyzer
    from grading_calibration.domain.models import GradingInstance, PredictionResult
    import numpy as np

    # Create example instances
    np.random.seed(42)

    instances = [
        GradingInstance(
            id=f"inst_{i}",
            question="What is the orbital hybridization of the central atom?",
            student_answer=" ".join(["word"] * np.random.randint(10, 50)),
            rubric_item="Correct identification of sp3 hybridization",
            ground_truth_label=np.random.choice([0, 1]),
            metadata={'question_id': 1, 'rubric_id': 'R1', 'student_id': f'S{i}'}
        )
        for i in range(100)
    ]

    predictions = [
        PredictionResult(
            instance_id=inst.id,
            prediction=np.random.choice([0, 1]),
            raw_confidence=np.random.uniform(0.4, 0.95),
            ground_truth=inst.ground_truth_label
        )
        for inst in instances
    ]

    # Extract features
    extractor = FeatureExtractor()
    features_df = extractor.extract_batch(instances, predictions)

    print(f"\nExtracted {len(features_df.columns)} features for {len(features_df)} instances")
    print(f"Features include: {', '.join(list(features_df.columns)[:5])}...")

    # Analyze correlations
    analyzer = FeatureAnalyzer(predictions, instances)
    corrs = analyzer.feature_correlations(target='raw_confidence', method='spearman')

    print(f"\nTop 5 features correlated with confidence:")
    for feature, corr in corrs.head(5).items():
        print(f"  {feature}: r={corr:.3f}")


def example_commonality_analysis():
    """Example: Find patterns in high/low confidence instances."""
    print("\n" + "=" * 80)
    print("Example 4: Commonality Analysis")
    print("=" * 80)

    from analysis import CommonalityAnalyzer
    from grading_calibration.domain.models import GradingInstance, PredictionResult
    import numpy as np

    # Create example data
    np.random.seed(42)

    instances = [
        GradingInstance(
            id=f"inst_{i}",
            question="Chemistry question",
            student_answer=" ".join(["word"] * np.random.randint(10, 50)),
            rubric_item="Rubric item",
            ground_truth_label=1,
            metadata={'question_id': 1}
        )
        for i in range(100)
    ]

    predictions = [
        PredictionResult(
            instance_id=inst.id,
            prediction=1,
            raw_confidence=0.9 if i < 50 else 0.4,  # Half high, half low
            ground_truth=1
        )
        for i, inst in enumerate(instances)
    ]

    # Analyze commonalities
    analyzer = CommonalityAnalyzer(predictions, instances)

    high_conf = analyzer.identify_high_confidence(threshold=0.8)
    low_conf = analyzer.identify_low_confidence(threshold=0.5)

    print(f"\nIdentified {len(high_conf)} high confidence and {len(low_conf)} low confidence instances")

    # Compare groups
    comparison = analyzer.compare_high_vs_low_confidence()

    if not comparison.empty:
        print(f"\nTop 3 distinguishing features:")
        for _, row in comparison.head(3).iterrows():
            print(f"  {row['feature']}: d={row['effect_size']:.2f}, p={row['p_value']:.4f}")


def example_mechanistic_analysis():
    """Example: Mechanistic analysis and recommendations."""
    print("\n" + "=" * 80)
    print("Example 5: Mechanistic Analysis")
    print("=" * 80)

    from analysis import MechanisticAnalyzer
    from grading_calibration.domain.models import GradingInstance, PredictionResult
    import numpy as np

    # Create example data
    np.random.seed(42)

    instances = [
        GradingInstance(
            id=f"inst_{i}",
            question="Question",
            student_answer="Answer",
            rubric_item="Rubric",
            ground_truth_label=1,
            metadata={'question_id': i % 4 + 1}
        )
        for i in range(100)
    ]

    predictions = [
        PredictionResult(
            instance_id=inst.id,
            prediction=1,
            raw_confidence=np.random.uniform(0.5, 0.9),
            ground_truth=1
        )
        for inst in instances
    ]

    # Mechanistic analysis
    analyzer = MechanisticAnalyzer(predictions, instances)

    # Get recommendations
    recommendations = analyzer.recommend_improvements()

    print(f"\nGenerated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS FRAMEWORK - EXAMPLES")
    print("=" * 80)

    try:
        example_basic_metrics_with_ci()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")

    try:
        example_hypothesis_testing()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")

    try:
        example_feature_analysis()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")

    try:
        example_commonality_analysis()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")

    try:
        example_mechanistic_analysis()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Load your actual predictions from CSV")
    print("2. Use these tools to analyze calibration quality")
    print("3. Generate comprehensive reports with statistical rigor")
    print("4. See the implementation plan at: ~/.claude/plans/silly-seeking-toast.md")


if __name__ == "__main__":
    main()
