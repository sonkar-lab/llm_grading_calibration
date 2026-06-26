"""
Mechanistic analysis for causal investigation of calibration.

This module provides tools for understanding what causes good or poor calibration,
including stratified evaluation, partial correlations, and automated recommendations.

Example:
    >>> from analysis import MechanisticAnalyzer
    >>> analyzer = MechanisticAnalyzer(predictions, instances)
    >>> # Evaluate calibration by difficulty
    >>> by_difficulty = analyzer.evaluate_by_difficulty(difficulty_proxy='answer_length')
    >>> # Get recommendations
    >>> recommendations = analyzer.recommend_improvements()
    >>> for rec in recommendations:
    ...     print(rec)
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from .feature_extraction import FeatureExtractor
from .feature_analysis import FeatureAnalyzer


class MechanisticAnalyzer:
    """
    Causal analysis and mechanistic investigation of calibration.

    This class provides methods for understanding what drives calibration quality
    and generating actionable recommendations for improvement.

    Attributes:
        predictions: List of PredictionResult objects
        instances: List of GradingInstance objects
        tester: Optional HypothesisTester for statistical tests
        feature_analyzer: FeatureAnalyzer instance

    Example:
        >>> analyzer = MechanisticAnalyzer(predictions, instances)
        >>> # Stratified evaluation
        >>> by_question = analyzer.evaluate_by_question_type()
        >>> # Recommendations
        >>> recs = analyzer.recommend_improvements()
    """

    def __init__(
        self,
        predictions: List[Any],
        instances: List[Any],
        tester: Optional[Any] = None
    ):
        """
        Initialize mechanistic analyzer.

        Args:
            predictions: List of PredictionResult objects
            instances: List of GradingInstance objects
            tester: Optional HypothesisTester instance
        """
        self.predictions = predictions
        self.instances = instances
        self.tester = tester
        self.feature_analyzer = FeatureAnalyzer(predictions, instances)

        # Create lookup for instances
        self.instance_dict = {inst.id: inst for inst in instances}

    # ===========================================================================
    # Stratified Evaluation
    # ===========================================================================

    def evaluate_by_difficulty(
        self,
        metric_func: Optional[Callable] = None,
        difficulty_proxy: str = 'answer_length',
        bins: int = 4
    ):
        """
        Evaluate calibration quality stratified by instance difficulty.

        Uses a proxy feature (e.g., answer_length) to estimate difficulty,
        then computes metrics within each difficulty stratum.

        Args:
            metric_func: Metric to compute (default: ECE)
            difficulty_proxy: Feature to use as difficulty proxy
            bins: Number of difficulty bins

        Returns:
            StratifiedMetrics object

        Example:
            >>> from grading_calibration.metrics import expected_calibration_error
            >>> by_diff = analyzer.evaluate_by_difficulty(
            ...     metric_func=expected_calibration_error,
            ...     difficulty_proxy='answer_length',
            ...     bins=4
            ... )
            >>> print(by_diff.summary_table())
        """
        return self.feature_analyzer.stratify_by_feature(
            feature=difficulty_proxy,
            bins=bins,
            labels=['easy', 'medium', 'hard', 'very_hard'] if bins == 4 else None,
            metric_func=metric_func
        )

    def evaluate_by_question_type(
        self,
        metric_func: Optional[Callable] = None
    ):
        """
        Evaluate calibration quality stratified by question type.

        Args:
            metric_func: Metric to compute (default: ECE)

        Returns:
            StratifiedMetrics object

        Example:
            >>> by_q = analyzer.evaluate_by_question_type()
            >>> # Identify which questions have poor calibration
            >>> sorted_groups = by_q.get_sorted_groups(ascending=False)
            >>> print(f"Worst calibrated question: {sorted_groups[0][0]}")
        """
        return self.feature_analyzer.stratify_by_feature(
            feature='question_id',
            bins=None,
            metric_func=metric_func
        )

    def evaluate_by_rubric_type(
        self,
        metric_func: Optional[Callable] = None
    ):
        """
        Evaluate calibration quality stratified by rubric item.

        Args:
            metric_func: Metric to compute (default: ECE)

        Returns:
            StratifiedMetrics object

        Example:
            >>> by_rubric = analyzer.evaluate_by_rubric_type()
            >>> print(by_rubric.summary_table())
        """
        return self.feature_analyzer.stratify_by_feature(
            feature='rubric_id',
            bins=None,
            metric_func=metric_func
        )

    # ===========================================================================
    # Causal Analysis
    # ===========================================================================

    def partial_correlation(
        self,
        feature: str,
        target: str,
        control_features: List[str]
    ) -> float:
        """
        Compute partial correlation controlling for confounders.

        Helps identify direct vs indirect relationships between features
        and calibration quality.

        Args:
            feature: Feature of interest
            target: Target variable (e.g., 'raw_confidence')
            control_features: Features to control for

        Returns:
            Partial correlation coefficient

        Example:
            >>> # Does answer_length affect confidence after controlling for
            >>> # lexical_diversity and readability?
            >>> partial_corr = analyzer.partial_correlation(
            ...     'answer_length',
            ...     'raw_confidence',
            ...     ['lexical_diversity', 'readability_score']
            ... )
            >>> print(f"Partial correlation: {partial_corr:.3f}")
        """
        df = self.feature_analyzer.compute_features()

        # Select relevant columns
        all_features = [feature, target] + control_features
        data = df[all_features].dropna()

        if len(data) < 30:
            raise ValueError("Not enough valid data for partial correlation")

        # Simple partial correlation using residuals method
        # 1. Regress feature on control features
        # 2. Regress target on control features
        # 3. Correlate residuals
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        # Residualize feature
        X_control = data[control_features]
        model_feature = LinearRegression()
        model_feature.fit(X_control, data[feature])
        residuals_feature = data[feature] - model_feature.predict(X_control)

        # Residualize target
        model_target = LinearRegression()
        model_target.fit(X_control, data[target])
        residuals_target = data[target] - model_target.predict(X_control)

        # Correlation of residuals
        from scipy.stats import pearsonr
        partial_corr, _ = pearsonr(residuals_feature, residuals_target)

        return float(partial_corr)

    # ===========================================================================
    # Recommendation Engine
    # ===========================================================================

    def recommend_improvements(self) -> List[str]:
        """
        Analyze results and suggest concrete improvements.

        Examines calibration quality across different stratifications and
        generates actionable recommendations.

        Returns:
            List of recommendation strings

        Example:
            >>> recommendations = analyzer.recommend_improvements()
            >>> print("Recommendations for improving calibration:")
            >>> for i, rec in enumerate(recommendations, 1):
            ...     print(f"{i}. {rec}")
        """
        from grading_calibration.metrics import expected_calibration_error

        recommendations = []

        # 1. Check overall calibration quality
        overall_ece = expected_calibration_error(self.predictions)

        if overall_ece > 0.15:
            recommendations.append(
                f"Overall ECE is high ({overall_ece:.3f}). Consider applying calibration "
                "(isotonic regression, temperature scaling, or Platt scaling)."
            )

        # 2. Check question-specific calibration
        try:
            by_question = self.evaluate_by_question_type(expected_calibration_error)
            question_values = by_question.get_group_values()

            if len(question_values) > 0:
                worst_q = max(question_values.items(), key=lambda x: x[1])
                best_q = min(question_values.items(), key=lambda x: x[1])

                if worst_q[1] > 1.5 * best_q[1]:
                    recommendations.append(
                        f"Question {worst_q[0]} has significantly worse calibration "
                        f"(ECE={worst_q[1]:.3f} vs {best_q[1]:.3f}). Consider question-specific "
                        "calibration or investigating what makes this question difficult."
                    )
        except Exception:
            pass

        # 3. Check rubric-specific calibration
        try:
            by_rubric = self.evaluate_by_rubric_type(expected_calibration_error)
            rubric_values = by_rubric.get_group_values()

            if len(rubric_values) > 0:
                poorly_calibrated_rubrics = [
                    rid for rid, ece in rubric_values.items() if ece > 0.15
                ]

                if len(poorly_calibrated_rubrics) > 0:
                    recommendations.append(
                        f"Rubric items {', '.join(poorly_calibrated_rubrics[:3])} have poor "
                        "calibration. Consider reformulating these rubric items or using "
                        "rubric-specific calibration."
                    )
        except Exception:
            pass

        # 4. Check confidence-accuracy correlation
        try:
            corrs = self.feature_analyzer.feature_correlations(target='is_correct')

            if 'raw_confidence' in corrs:
                conf_corr = corrs['raw_confidence']

                if conf_corr < 0.4:
                    recommendations.append(
                        f"Confidence poorly correlates with correctness (r={conf_corr:.2f}). "
                        "Consider using a different confidence estimation method or combining "
                        "multiple confidence signals."
                    )
        except Exception:
            pass

        # 5. Check for vote agreement (if self-consistency)
        try:
            df = self.feature_analyzer.compute_features()

            if 'vote_agreement' in df.columns:
                high_agreement = df[df['vote_agreement'] > 0.9]
                if len(high_agreement) > 0:
                    high_agreement_accuracy = high_agreement['is_correct'].mean()

                    if high_agreement_accuracy > 0.9:
                        recommendations.append(
                            f"High vote agreement (>0.9) strongly predicts correctness "
                            f"(acc={high_agreement_accuracy:.1%}). Consider using vote agreement "
                            "as a direct confidence proxy instead of vote proportion."
                        )
        except Exception:
            pass

        # 6. Check answer length effects
        try:
            corrs = self.feature_analyzer.feature_correlations(target='raw_confidence')

            if 'answer_length' in corrs and abs(corrs['answer_length']) > 0.3:
                recommendations.append(
                    f"Answer length strongly correlates with confidence (r={corrs['answer_length']:.2f}). "
                    "Consider normalizing confidence by answer length or using length-stratified calibration."
                )
        except Exception:
            pass

        # Default recommendation if no issues found
        if len(recommendations) == 0:
            recommendations.append(
                "Calibration appears reasonable overall. Continue monitoring performance "
                "and consider A/B testing different calibration methods."
            )

        return recommendations

    # ===========================================================================
    # Simulation and Intervention
    # ===========================================================================

    def simulate_improved_confidence(
        self,
        improvement_factor: float = 1.1
    ) -> List[Any]:
        """
        Simulate what would happen if raw confidence improved by a factor.

        Useful for "what if" scenarios and sensitivity analysis.

        Args:
            improvement_factor: Multiplicative factor for confidence scores

        Returns:
            List of PredictionResult objects with simulated confidences

        Example:
            >>> # What if confidence scores were 10% better?
            >>> simulated = analyzer.simulate_improved_confidence(1.1)
            >>> from grading_calibration.metrics import expected_calibration_error
            >>> simulated_ece = expected_calibration_error(simulated)
            >>> print(f"Simulated ECE: {simulated_ece:.3f}")

        Note:
            This is a simplistic simulation and should be used for
            rough estimation only.
        """
        # Create copies with modified confidence
        simulated = []

        for pred in self.predictions:
            # Create a copy-like object (simplified)
            import copy
            sim_pred = copy.copy(pred)

            # Adjust confidence (clip to [0, 1])
            new_conf = min(1.0, pred.raw_confidence * improvement_factor)
            sim_pred.raw_confidence = new_conf

            simulated.append(sim_pred)

        return simulated

    def simulate_perfect_calibration(self) -> List[Any]:
        """
        Simulate perfectly calibrated predictions.

        Sets confidence = empirical accuracy for each confidence bin.
        Shows theoretical upper bound on calibration quality.

        Returns:
            List of PredictionResult objects with perfect calibration

        Example:
            >>> perfect = analyzer.simulate_perfect_calibration()
            >>> from grading_calibration.metrics import expected_calibration_error
            >>> ece = expected_calibration_error(perfect)
            >>> print(f"ECE under perfect calibration: {ece:.4f}")  # Should be ~0
        """
        import copy

        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        binned_preds = {i: [] for i in range(10)}

        for pred in self.predictions:
            bin_idx = min(9, int(pred.raw_confidence * 10))
            binned_preds[bin_idx].append(pred)

        # Compute accuracy per bin
        bin_accuracies = {}
        for bin_idx, preds in binned_preds.items():
            if len(preds) > 0:
                acc = np.mean([p.is_correct() for p in preds])
                bin_accuracies[bin_idx] = acc
            else:
                bin_accuracies[bin_idx] = 0.5  # Default for empty bins

        # Create perfectly calibrated versions
        perfect = []
        for pred in self.predictions:
            sim_pred = copy.copy(pred)
            bin_idx = min(9, int(pred.raw_confidence * 10))
            sim_pred.raw_confidence = bin_accuracies[bin_idx]
            perfect.append(sim_pred)

        return perfect
