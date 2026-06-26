"""
Commonality analysis for finding patterns in confidence groups.

This module provides the CommonalityAnalyzer class for identifying patterns
and commonalities within high/low confidence instances, correct/incorrect
predictions, and well/poorly calibrated instances.

Example:
    >>> from analysis import CommonalityAnalyzer
    >>> analyzer = CommonalityAnalyzer(predictions, instances)
    >>> # Compare high vs low confidence
    >>> comparison = analyzer.compare_high_vs_low_confidence()
    >>> summary = analyzer.generate_summary('high_confidence', high_conf_features)
    >>> print(summary)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from .feature_extraction import FeatureExtractor


class CommonalityAnalyzer:
    """
    Identify patterns in high/low confidence instances with statistical validation.

    This class helps understand what makes instances high or low confidence,
    correctly or incorrectly predicted, and well or poorly calibrated.

    Attributes:
        predictions: List of PredictionResult objects
        instances: List of GradingInstance objects
        tester: Optional HypothesisTester for statistical comparisons
        extractor: FeatureExtractor instance

    Example:
        >>> analyzer = CommonalityAnalyzer(predictions, instances)
        >>> high_conf = analyzer.identify_high_confidence(threshold=0.8)
        >>> low_conf = analyzer.identify_low_confidence(threshold=0.5)
        >>> comparison = analyzer.compare_high_vs_low_confidence()
        >>> print(comparison.head(20))
    """

    def __init__(
        self,
        predictions: List[Any],
        instances: List[Any],
        tester: Optional[Any] = None
    ):
        """
        Initialize commonality analyzer.

        Args:
            predictions: List of PredictionResult objects
            instances: List of GradingInstance objects
            tester: Optional HypothesisTester instance for statistical tests
        """
        self.predictions = predictions
        self.instances = instances
        self.tester = tester
        self.extractor = FeatureExtractor()

        # Create lookup for instances
        self.instance_dict = {inst.id: inst for inst in instances}

    # ===========================================================================
    # Group Identification
    # ===========================================================================

    def identify_high_confidence(self, threshold: float = 0.8) -> List[Any]:
        """
        Identify high confidence predictions.

        Args:
            threshold: Confidence threshold (default: 0.8)

        Returns:
            List of PredictionResult objects with raw_confidence > threshold

        Example:
            >>> high_conf_preds = analyzer.identify_high_confidence(0.8)
            >>> print(f"Found {len(high_conf_preds)} high confidence instances")
        """
        return [p for p in self.predictions if p.raw_confidence > threshold]

    def identify_low_confidence(self, threshold: float = 0.5) -> List[Any]:
        """
        Identify low confidence predictions.

        Args:
            threshold: Confidence threshold (default: 0.5)

        Returns:
            List of PredictionResult objects with raw_confidence < threshold
        """
        return [p for p in self.predictions if p.raw_confidence < threshold]

    def identify_correct_predictions(self) -> List[Any]:
        """Identify correct predictions."""
        return [p for p in self.predictions if p.is_correct()]

    def identify_incorrect_predictions(self) -> List[Any]:
        """Identify incorrect predictions."""
        return [p for p in self.predictions if not p.is_correct()]

    def identify_well_calibrated(self, tolerance: float = 0.1) -> List[Any]:
        """
        Identify well-calibrated instances.

        Well-calibrated means |confidence - correctness| < tolerance.

        Args:
            tolerance: Maximum deviation from perfect calibration

        Returns:
            List of well-calibrated predictions

        Example:
            >>> well_cal = analyzer.identify_well_calibrated(tolerance=0.1)
            >>> # These are instances where confidence matches correctness
        """
        well_calibrated = []

        for p in self.predictions:
            correctness = float(p.is_correct())
            deviation = abs(p.raw_confidence - correctness)

            if deviation < tolerance:
                well_calibrated.append(p)

        return well_calibrated

    def identify_poorly_calibrated(self, threshold: float = 0.3) -> List[Any]:
        """
        Identify poorly calibrated instances.

        Poorly calibrated means |confidence - correctness| > threshold.

        Args:
            threshold: Minimum deviation for poor calibration

        Returns:
            List of poorly calibrated predictions
        """
        poorly_calibrated = []

        for p in self.predictions:
            correctness = float(p.is_correct())
            deviation = abs(p.raw_confidence - correctness)

            if deviation > threshold:
                poorly_calibrated.append(p)

        return poorly_calibrated

    # ===========================================================================
    # Feature Comparison
    # ===========================================================================

    def compare_high_vs_low_confidence(
        self,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare features between high and low confidence groups.

        Args:
            high_threshold: Threshold for high confidence
            low_threshold: Threshold for low confidence
            features: Features to compare (None = all numerical features)

        Returns:
            DataFrame with comparison statistics:
                - feature: Feature name
                - high_mean: Mean in high confidence group
                - low_mean: Mean in low confidence group
                - difference: high_mean - low_mean
                - ratio: high_mean / low_mean
                - p_value: Statistical significance
                - effect_size: Cohen's d or Cliff's delta

        Example:
            >>> comparison = analyzer.compare_high_vs_low_confidence()
            >>> # Sort by effect size to find most distinguishing features
            >>> print(comparison.sort_values('effect_size', key=abs, ascending=False))
        """
        from grading_calibration.metrics.statistical_tests import HypothesisTester

        # Get groups
        high_conf = self.identify_high_confidence(high_threshold)
        low_conf = self.identify_low_confidence(low_threshold)

        # Get instances for each group
        high_instances = [self.instance_dict[p.instance_id] for p in high_conf if p.instance_id in self.instance_dict]
        low_instances = [self.instance_dict[p.instance_id] for p in low_conf if p.instance_id in self.instance_dict]

        # Extract features
        high_features_df = self.extractor.extract_batch(high_instances, high_conf)
        low_features_df = self.extractor.extract_batch(low_instances, low_conf)

        # Select features to compare
        if features is None:
            features = high_features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Compare each feature
        tester = self.tester if self.tester is not None else HypothesisTester()

        rows = []
        for feature in features:
            if feature in high_features_df.columns and feature in low_features_df.columns:
                high_values = high_features_df[feature].dropna()
                low_values = low_features_df[feature].dropna()

                if len(high_values) > 0 and len(low_values) > 0:
                    high_mean = high_values.mean()
                    low_mean = low_values.mean()
                    difference = high_mean - low_mean
                    ratio = high_mean / low_mean if low_mean != 0 else np.inf

                    # Statistical test
                    try:
                        from scipy import stats
                        statistic, p_value = stats.ttest_ind(high_values, low_values)
                        effect_size = tester.cohens_d(high_values.values, low_values.values, paired=False)
                    except Exception:
                        p_value = np.nan
                        effect_size = np.nan

                    rows.append({
                        'feature': feature,
                        'high_mean': high_mean,
                        'low_mean': low_mean,
                        'difference': difference,
                        'ratio': ratio,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'n_high': len(high_values),
                        'n_low': len(low_values)
                    })

        df = pd.DataFrame(rows)

        # Sort by absolute effect size
        if not df.empty and 'effect_size' in df.columns:
            df = df.sort_values('effect_size', key=abs, ascending=False)

        return df

    def compare_correct_vs_incorrect(
        self,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare features between correct and incorrect predictions.

        Similar to compare_high_vs_low_confidence but for correctness.

        Args:
            features: Features to compare (None = all numerical)

        Returns:
            DataFrame with comparison statistics
        """
        from grading_calibration.metrics.statistical_tests import HypothesisTester

        # Get groups
        correct = self.identify_correct_predictions()
        incorrect = self.identify_incorrect_predictions()

        # Get instances
        correct_instances = [self.instance_dict[p.instance_id] for p in correct if p.instance_id in self.instance_dict]
        incorrect_instances = [self.instance_dict[p.instance_id] for p in incorrect if p.instance_id in self.instance_dict]

        # Extract features
        correct_features_df = self.extractor.extract_batch(correct_instances, correct)
        incorrect_features_df = self.extractor.extract_batch(incorrect_instances, incorrect)

        # Select features
        if features is None:
            features = correct_features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Compare
        tester = self.tester if self.tester is not None else HypothesisTester()

        rows = []
        for feature in features:
            if feature in correct_features_df.columns and feature in incorrect_features_df.columns:
                correct_values = correct_features_df[feature].dropna()
                incorrect_values = incorrect_features_df[feature].dropna()

                if len(correct_values) > 0 and len(incorrect_values) > 0:
                    correct_mean = correct_values.mean()
                    incorrect_mean = incorrect_values.mean()
                    difference = correct_mean - incorrect_mean
                    ratio = correct_mean / incorrect_mean if incorrect_mean != 0 else np.inf

                    # Statistical test
                    try:
                        from scipy import stats
                        statistic, p_value = stats.ttest_ind(correct_values, incorrect_values)
                        effect_size = tester.cohens_d(correct_values.values, incorrect_values.values, paired=False)
                    except Exception:
                        p_value = np.nan
                        effect_size = np.nan

                    rows.append({
                        'feature': feature,
                        'correct_mean': correct_mean,
                        'incorrect_mean': incorrect_mean,
                        'difference': difference,
                        'ratio': ratio,
                        'p_value': p_value,
                        'effect_size': effect_size
                    })

        df = pd.DataFrame(rows)

        if not df.empty and 'effect_size' in df.columns:
            df = df.sort_values('effect_size', key=abs, ascending=False)

        return df

    def compare_well_vs_poorly_calibrated(
        self,
        well_tolerance: float = 0.1,
        poor_threshold: float = 0.3,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare features between well and poorly calibrated instances.

        Args:
            well_tolerance: Tolerance for well-calibrated
            poor_threshold: Threshold for poorly calibrated
            features: Features to compare

        Returns:
            DataFrame with comparison statistics
        """
        from grading_calibration.metrics.statistical_tests import HypothesisTester

        # Get groups
        well_cal = self.identify_well_calibrated(well_tolerance)
        poor_cal = self.identify_poorly_calibrated(poor_threshold)

        # Get instances
        well_instances = [self.instance_dict[p.instance_id] for p in well_cal if p.instance_id in self.instance_dict]
        poor_instances = [self.instance_dict[p.instance_id] for p in poor_cal if p.instance_id in self.instance_dict]

        # Extract features
        well_features_df = self.extractor.extract_batch(well_instances, well_cal)
        poor_features_df = self.extractor.extract_batch(poor_instances, poor_cal)

        # Select features
        if features is None:
            features = well_features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Compare
        tester = self.tester if self.tester is not None else HypothesisTester()

        rows = []
        for feature in features:
            if feature in well_features_df.columns and feature in poor_features_df.columns:
                well_values = well_features_df[feature].dropna()
                poor_values = poor_features_df[feature].dropna()

                if len(well_values) > 0 and len(poor_values) > 0:
                    well_mean = well_values.mean()
                    poor_mean = poor_values.mean()
                    difference = well_mean - poor_mean
                    ratio = well_mean / poor_mean if poor_mean != 0 else np.inf

                    try:
                        from scipy import stats
                        statistic, p_value = stats.ttest_ind(well_values, poor_values)
                        effect_size = tester.cohens_d(well_values.values, poor_values.values, paired=False)
                    except Exception:
                        p_value = np.nan
                        effect_size = np.nan

                    rows.append({
                        'feature': feature,
                        'well_mean': well_mean,
                        'poor_mean': poor_mean,
                        'difference': difference,
                        'ratio': ratio,
                        'p_value': p_value,
                        'effect_size': effect_size
                    })

        df = pd.DataFrame(rows)

        if not df.empty and 'effect_size' in df.columns:
            df = df.sort_values('effect_size', key=abs, ascending=False)

        return df

    # ===========================================================================
    # Pattern Detection
    # ===========================================================================

    def find_common_keywords(
        self,
        group: List[Any],
        top_k: int = 20,
        min_length: int = 3
    ) -> Dict[str, int]:
        """
        Extract most frequent keywords in student answers for a group.

        Args:
            group: List of PredictionResult objects
            top_k: Number of top keywords to return
            min_length: Minimum word length to consider

        Returns:
            Dict mapping keyword -> frequency

        Example:
            >>> high_conf = analyzer.identify_high_confidence()
            >>> keywords = analyzer.find_common_keywords(high_conf, top_k=10)
            >>> print("Top keywords in high confidence answers:")
            >>> for word, count in keywords.items():
            ...     print(f"  {word}: {count}")
        """
        # Get instances for this group
        instances = [self.instance_dict[p.instance_id] for p in group if p.instance_id in self.instance_dict]

        # Collect all words
        all_words = []
        for inst in instances:
            words = inst.student_answer.lower().split()
            # Filter by length and remove common words
            words = [w for w in words if len(w) >= min_length and w.isalpha()]
            all_words.extend(words)

        # Count frequencies
        word_counts = Counter(all_words)

        # Return top-k
        return dict(word_counts.most_common(top_k))

    def find_distinguishing_features(
        self,
        group_a: List[Any],
        group_b: List[Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find features with largest effect sizes between two groups.

        Args:
            group_a: First group of predictions
            group_b: Second group of predictions
            top_k: Number of top features to return

        Returns:
            List of (feature_name, effect_size) tuples sorted by |effect_size|

        Example:
            >>> high_conf = analyzer.identify_high_confidence()
            >>> low_conf = analyzer.identify_low_confidence()
            >>> distinguishing = analyzer.find_distinguishing_features(
            ...     high_conf, low_conf, top_k=5
            ... )
            >>> print("Top 5 distinguishing features:")
            >>> for feature, effect_size in distinguishing:
            ...     print(f"  {feature}: d={effect_size:.2f}")
        """
        from grading_calibration.metrics.statistical_tests import HypothesisTester

        # Get instances
        instances_a = [self.instance_dict[p.instance_id] for p in group_a if p.instance_id in self.instance_dict]
        instances_b = [self.instance_dict[p.instance_id] for p in group_b if p.instance_id in self.instance_dict]

        # Extract features
        features_a = self.extractor.extract_batch(instances_a, group_a)
        features_b = self.extractor.extract_batch(instances_b, group_b)

        # Compute effect sizes for all numerical features
        numerical_features = features_a.select_dtypes(include=[np.number]).columns
        tester = self.tester if self.tester is not None else HypothesisTester()

        effect_sizes = []

        for feature in numerical_features:
            if feature in features_a.columns and feature in features_b.columns:
                values_a = features_a[feature].dropna()
                values_b = features_b[feature].dropna()

                if len(values_a) > 0 and len(values_b) > 0:
                    try:
                        effect_size = tester.cohens_d(values_a.values, values_b.values, paired=False)
                        effect_sizes.append((feature, effect_size))
                    except Exception:
                        continue

        # Sort by absolute effect size
        effect_sizes.sort(key=lambda x: abs(x[1]), reverse=True)

        return effect_sizes[:top_k]

    # ===========================================================================
    # Summary Generation
    # ===========================================================================

    def generate_summary(
        self,
        group_name: str,
        group: List[Any],
        comparison_group: Optional[List[Any]] = None
    ) -> str:
        """
        Generate human-readable summary of group characteristics.

        Args:
            group_name: Name of the group (e.g., "High Confidence")
            group: List of predictions in this group
            comparison_group: Optional comparison group for relative statistics

        Returns:
            Human-readable summary string

        Example:
            >>> high_conf = analyzer.identify_high_confidence()
            >>> low_conf = analyzer.identify_low_confidence()
            >>> summary = analyzer.generate_summary(
            ...     "High Confidence Group",
            ...     high_conf,
            ...     comparison_group=low_conf
            ... )
            >>> print(summary)
        """
        # Get instances
        instances = [self.instance_dict[p.instance_id] for p in group if p.instance_id in self.instance_dict]

        # Extract features
        features_df = self.extractor.extract_batch(instances, group)

        # Basic statistics
        n = len(group)
        accuracy = np.mean([p.is_correct() for p in group])
        avg_confidence = np.mean([p.raw_confidence for p in group])

        summary_parts = [
            f"# {group_name} (n={n})",
            f"",
            f"**Accuracy**: {accuracy:.1%}",
            f"**Average Confidence**: {avg_confidence:.3f}",
            f"",
            f"## Key Features:"
        ]

        # Add feature statistics
        numerical_features = features_df.select_dtypes(include=[np.number]).columns

        for feature in ['answer_length', 'lexical_diversity', 'readability_score']:
            if feature in numerical_features:
                mean_val = features_df[feature].mean()
                summary_parts.append(f"- **{feature}**: {mean_val:.2f}")

        # If comparison group provided, add relative statistics
        if comparison_group is not None:
            comp_instances = [self.instance_dict[p.instance_id] for p in comparison_group if p.instance_id in self.instance_dict]
            comp_features_df = self.extractor.extract_batch(comp_instances, comparison_group)

            summary_parts.append("")
            summary_parts.append("## Comparison to Other Group:")

            for feature in ['answer_length', 'lexical_diversity']:
                if feature in numerical_features and feature in comp_features_df.columns:
                    our_mean = features_df[feature].mean()
                    their_mean = comp_features_df[feature].mean()
                    if their_mean != 0:
                        ratio = our_mean / their_mean
                        summary_parts.append(f"- **{feature}**: {ratio:.2f}x")

        # Add top keywords
        keywords = self.find_common_keywords(group, top_k=5)
        if keywords:
            summary_parts.append("")
            summary_parts.append("## Common Keywords:")
            for word, count in keywords.items():
                summary_parts.append(f"- {word} ({count} occurrences)")

        return "\n".join(summary_parts)

    # ===========================================================================
    # Visualization
    # ===========================================================================

    def plot_group_comparison(
        self,
        feature: str,
        groups: Dict[str, List[Any]],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create box plot comparing feature distributions across groups.

        Args:
            feature: Feature to compare
            groups: Dict mapping group_name -> list of predictions
            figsize: Figure size

        Returns:
            matplotlib Figure

        Example:
            >>> fig = analyzer.plot_group_comparison(
            ...     'answer_length',
            ...     {
            ...         'High Confidence': high_conf,
            ...         'Low Confidence': low_conf
            ...     }
            ... )
            >>> fig.savefig('answer_length_comparison.png')
        """
        data = []
        labels = []

        for group_name, group_preds in groups.items():
            group_instances = [self.instance_dict[p.instance_id] for p in group_preds if p.instance_id in self.instance_dict]
            features_df = self.extractor.extract_batch(group_instances, group_preds)

            if feature in features_df.columns:
                values = features_df[feature].dropna().values
                data.append(values)
                labels.append(group_name)

        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(data, labels=labels)
        ax.set_ylabel(feature)
        ax.set_title(f"Distribution of {feature} across groups")
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig
