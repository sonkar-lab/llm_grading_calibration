"""
Feature analysis for understanding calibration drivers.

This module provides the FeatureAnalyzer class for analyzing relationships between
features and calibration quality. Supports:
- Correlation analysis (feature vs confidence, correctness)
- Stratified analysis (group by feature values)
- Feature importance (random forest, permutation)
- Interpretable models (linear regression, logistic regression)
- Visualization (heatmaps, scatter plots, etc.)

Example:
    >>> from analysis import FeatureAnalyzer
    >>> analyzer = FeatureAnalyzer(predictions, instances)
    >>> # What correlates with confidence?
    >>> corrs = analyzer.feature_correlations(target='raw_confidence')
    >>> print(corrs.head(10))
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt

from .feature_extraction import FeatureExtractor


class FeatureAnalyzer:
    """
    Analyze relationships between features and calibration quality.

    This class provides comprehensive tools for understanding what drives
    confidence, correctness, and calibration quality.

    Attributes:
        predictions: List of PredictionResult objects
        instances: List of GradingInstance objects
        extractor: FeatureExtractor instance
        features_df: Cached DataFrame of all features

    Example:
        >>> analyzer = FeatureAnalyzer(predictions, instances)
        >>> # What features predict high confidence?
        >>> corrs = analyzer.feature_correlations(target='raw_confidence')
        >>> print(corrs.head(10))
    """

    def __init__(
        self,
        predictions: List[Any],
        instances: List[Any]
    ):
        """
        Initialize feature analyzer.

        Args:
            predictions: List of PredictionResult objects
            instances: List of GradingInstance objects
        """
        self.predictions = predictions
        self.instances = instances
        self.extractor = FeatureExtractor()
        self.features_df = None

    def compute_features(self) -> pd.DataFrame:
        """
        Compute and cache feature DataFrame.

        Returns:
            DataFrame with all extracted features

        Example:
            >>> df = analyzer.compute_features()
            >>> print(df.columns)
        """
        if self.features_df is None:
            self.features_df = self.extractor.extract_batch(
                self.instances,
                self.predictions
            )
        return self.features_df

    # ===========================================================================
    # Correlation Analysis
    # ===========================================================================

    def feature_correlations(
        self,
        target: str = "raw_confidence",
        method: str = "spearman",
        min_valid: int = 30
    ) -> pd.Series:
        """
        Compute correlations between features and target variable.

        Args:
            target: Target variable name (e.g., 'raw_confidence', 'is_correct')
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            min_valid: Minimum non-null values required for correlation

        Returns:
            Series of correlations sorted by absolute value (descending)

        Example:
            >>> corrs = analyzer.feature_correlations(target='raw_confidence')
            >>> print("Top 5 features correlated with confidence:")
            >>> print(corrs.head())
        """
        df = self.compute_features()

        # Select numerical features only (exclude target)
        numerical_features = df.select_dtypes(include=[np.number]).columns
        numerical_features = [f for f in numerical_features if f != target]

        correlations = {}

        for feature in numerical_features:
            # Drop NaN values
            valid_data = df[[feature, target]].dropna()

            if len(valid_data) >= min_valid:
                try:
                    if method == 'pearson':
                        corr, _ = pearsonr(valid_data[feature], valid_data[target])
                    elif method == 'spearman':
                        corr, _ = spearmanr(valid_data[feature], valid_data[target])
                    elif method == 'kendall':
                        corr, _ = kendalltau(valid_data[feature], valid_data[target])
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    correlations[feature] = corr
                except Exception:
                    # Skip features that cause errors
                    continue

        # Sort by absolute value (descending)
        corr_series = pd.Series(correlations)
        corr_series = corr_series.sort_values(key=abs, ascending=False)

        return corr_series

    def correlation_matrix(
        self,
        features: Optional[List[str]] = None,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for selected features.

        Args:
            features: List of feature names (None = all numerical features)
            method: Correlation method

        Returns:
            DataFrame with pairwise correlations

        Example:
            >>> corr_matrix = analyzer.correlation_matrix(
            ...     features=['answer_length', 'lexical_diversity', 'raw_confidence']
            ... )
            >>> print(corr_matrix)
        """
        df = self.compute_features()

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Compute correlation matrix
        corr_matrix = df[features].corr(method=method)

        return corr_matrix

    def plot_correlation_heatmap(
        self,
        features: Optional[List[str]] = None,
        method: str = "spearman",
        figsize: Tuple[int, int] = (12, 10),
        **kwargs
    ) -> plt.Figure:
        """
        Visualize correlation matrix as heatmap.

        Args:
            features: Features to include (None = all numerical)
            method: Correlation method
            figsize: Figure size
            **kwargs: Additional arguments for seaborn heatmap

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = analyzer.plot_correlation_heatmap()
            >>> fig.savefig('correlation_heatmap.png')
        """
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError("seaborn required for heatmap plotting. Install with: pip install seaborn")

        corr_matrix = self.correlation_matrix(features, method)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            ax=ax,
            **kwargs
        )
        ax.set_title(f"Feature Correlation Matrix ({method})")

        plt.tight_layout()
        return fig

    # ===========================================================================
    # Stratified Analysis
    # ===========================================================================

    def stratify_by_feature(
        self,
        feature: str,
        bins: Optional[Union[int, List[float]]] = None,
        labels: Optional[List[str]] = None,
        metric_func: Optional[Callable] = None
    ):
        """
        Stratify predictions by feature value and compute metrics.

        Args:
            feature: Feature to stratify by
            bins: Number of bins or bin edges (for numerical features)
            labels: Labels for bins
            metric_func: Metric to compute (default: ECE)

        Returns:
            StratifiedMetrics object

        Example:
            >>> # Stratify by answer length (quartiles)
            >>> stratified = analyzer.stratify_by_feature(
            ...     'answer_length',
            ...     bins=4,
            ...     labels=['short', 'medium', 'long', 'very_long']
            ... )
            >>> print(stratified.summary_table())
        """
        # Import here to avoid circular dependency
        from grading_calibration.metrics import expected_calibration_error
        from grading_calibration.metrics.containers import MetricResult, StratifiedMetrics

        df = self.compute_features()

        if metric_func is None:
            metric_func = expected_calibration_error

        # Bin numerical features if bins specified
        if bins is not None and pd.api.types.is_numeric_dtype(df[feature]):
            if isinstance(bins, int):
                # Equal-frequency bins (quantiles)
                df[f'{feature}_binned'] = pd.qcut(
                    df[feature],
                    bins,
                    labels=labels,
                    duplicates='drop'
                )
            else:
                # Specified bin edges
                df[f'{feature}_binned'] = pd.cut(df[feature], bins, labels=labels)

            stratify_column = f'{feature}_binned'
        else:
            stratify_column = feature

        # Group predictions by feature value
        groups = {}

        for group_value in df[stratify_column].dropna().unique():
            # Get predictions in this group
            group_indices = df[df[stratify_column] == group_value].index
            group_predictions = [p for p in self.predictions if p.instance_id in group_indices]

            if len(group_predictions) > 0:
                # Compute metric
                try:
                    metric_value = metric_func(group_predictions)

                    groups[str(group_value)] = MetricResult(
                        name=metric_func.__name__,
                        value=metric_value,
                        n_samples=len(group_predictions)
                    )
                except Exception:
                    # Skip groups that fail metric computation
                    continue

        return StratifiedMetrics(
            metric_name=metric_func.__name__,
            groups=groups,
            stratification_key=feature
        )

    def compare_groups(
        self,
        feature: str,
        group_a_value: Any,
        group_b_value: Any,
        metrics: List[str],
        tester: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Statistical comparison between two feature groups.

        Args:
            feature: Feature to group by
            group_a_value: Value defining group A
            group_b_value: Value defining group B
            metrics: List of metric names to compare
            tester: HypothesisTester instance

        Returns:
            Dict mapping metric_name -> HypothesisTestResult

        Example:
            >>> from grading_calibration.metrics.statistical_tests import HypothesisTester
            >>> tester = HypothesisTester()
            >>> comparison = analyzer.compare_groups(
            ...     feature='is_high_confidence',
            ...     group_a_value=1.0,
            ...     group_b_value=0.0,
            ...     metrics=['accuracy', 'ece'],
            ...     tester=tester
            ... )
            >>> for metric, test in comparison.items():
            ...     print(f"{metric}: {test.interpretation()}")
        """
        from grading_calibration.metrics import expected_calibration_error, brier_score
        from grading_calibration.metrics.statistical_tests import HypothesisTester

        if tester is None:
            tester = HypothesisTester()

        df = self.compute_features()

        # Get predictions for each group
        group_a_indices = df[df[feature] == group_a_value].index
        group_b_indices = df[df[feature] == group_b_value].index

        group_a_preds = [p for p in self.predictions if p.instance_id in group_a_indices]
        group_b_preds = [p for p in self.predictions if p.instance_id in group_b_indices]

        # Define metric functions
        metric_funcs = {
            'ece': expected_calibration_error,
            'brier': brier_score,
            'accuracy': lambda preds: np.mean([p.is_correct() for p in preds]),
        }

        # Run tests for each metric
        results = {}
        for metric_name in metrics:
            if metric_name in metric_funcs:
                try:
                    test_result = tester.independent_t_test(
                        group_a_preds,
                        group_b_preds,
                        metric_func=metric_funcs[metric_name]
                    )
                    results[metric_name] = test_result
                except Exception:
                    # Skip if test fails
                    continue

        return results

    # ===========================================================================
    # Feature Importance
    # ===========================================================================

    def feature_importance_for_confidence(
        self,
        method: str = "random_forest",
        top_k: Optional[int] = None
    ) -> pd.Series:
        """
        Compute feature importance for predicting confidence.

        Uses machine learning models to identify which features are most
        important for predicting raw confidence scores.

        Args:
            method: 'random_forest', 'gradient_boosting', or 'permutation'
            top_k: Return only top-k features

        Returns:
            Series of feature importances (sorted descending)

        Example:
            >>> importance = analyzer.feature_importance_for_confidence(
            ...     method='random_forest',
            ...     top_k=10
            ... )
            >>> print("Top 10 features for predicting confidence:")
            >>> print(importance)
        """
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.inspection import permutation_importance
        except ImportError:
            raise ImportError("scikit-learn required for feature importance. Install with: pip install scikit-learn")

        df = self.compute_features()

        # Prepare data
        X = df.select_dtypes(include=[np.number]).drop(columns=['raw_confidence'], errors='ignore')
        y = df['raw_confidence']

        # Remove rows with NaN
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid]

        if len(X) == 0:
            raise ValueError("No valid data for feature importance computation")

        # Fit model
        if method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            importances = model.feature_importances_
        elif method == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
        elif method == 'permutation':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            perm_imp = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importances = perm_imp.importances_mean
        else:
            raise ValueError(f"Unknown method: {method}")

        importance_series = pd.Series(importances, index=X.columns)
        importance_series = importance_series.sort_values(ascending=False)

        if top_k is not None:
            importance_series = importance_series.head(top_k)

        return importance_series

    def feature_importance_for_correctness(
        self,
        method: str = "random_forest",
        top_k: Optional[int] = None
    ) -> pd.Series:
        """
        Compute feature importance for predicting correctness.

        Identifies features that make instances easy or hard to grade correctly.

        Args:
            method: 'random_forest' or 'gradient_boosting'
            top_k: Return only top-k features

        Returns:
            Series of feature importances (sorted descending)

        Example:
            >>> importance = analyzer.feature_importance_for_correctness(top_k=10)
            >>> print("Features that predict correctness:")
            >>> print(importance)
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        df = self.compute_features()

        # Prepare data
        X = df.select_dtypes(include=[np.number]).drop(columns=['is_correct'], errors='ignore')
        y = df['is_correct']

        # Remove rows with NaN
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid]

        if len(X) == 0:
            raise ValueError("No valid data for feature importance computation")

        # Fit model
        if method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            importances = model.feature_importances_
        elif method == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
        else:
            raise ValueError(f"Unknown method: {method}")

        importance_series = pd.Series(importances, index=X.columns)
        importance_series = importance_series.sort_values(ascending=False)

        if top_k is not None:
            importance_series = importance_series.head(top_k)

        return importance_series

    # ===========================================================================
    # Interpretable Models
    # ===========================================================================

    def fit_confidence_model(
        self,
        features: List[str],
        model_type: str = "linear"
    ) -> Any:
        """
        Fit interpretable model: features -> confidence.

        Args:
            features: List of feature names to use as predictors
            model_type: 'linear' or 'ridge'

        Returns:
            Fitted sklearn model with .coef_ attribute

        Example:
            >>> model = analyzer.fit_confidence_model(
            ...     features=['answer_length', 'lexical_diversity', 'vote_agreement']
            ... )
            >>> print("Coefficients:", dict(zip(features, model.coef_)))
        """
        try:
            from sklearn.linear_model import LinearRegression, Ridge
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        df = self.compute_features()

        X = df[features]
        y = df['raw_confidence']

        # Remove NaN
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid]

        if len(X) == 0:
            raise ValueError("No valid data for model fitting")

        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X, y)

        return model

    def fit_correctness_model(
        self,
        features: List[str],
        model_type: str = "logistic"
    ) -> Any:
        """
        Fit interpretable model: features -> correctness.

        Args:
            features: List of feature names to use as predictors
            model_type: 'logistic' (only supported type)

        Returns:
            Fitted sklearn LogisticRegression model

        Example:
            >>> model = analyzer.fit_correctness_model(
            ...     features=['answer_length', 'raw_confidence']
            ... )
            >>> print("Coefficients:", dict(zip(features, model.coef_[0])))
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        df = self.compute_features()

        X = df[features]
        y = df['is_correct']

        # Remove NaN
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid]

        if len(X) == 0:
            raise ValueError("No valid data for model fitting")

        if model_type == 'logistic':
            model = LogisticRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X, y)

        return model

    # ===========================================================================
    # Visualization
    # ===========================================================================

    def plot_feature_vs_metric(
        self,
        feature: str,
        metric_func: Callable,
        bins: int = 10,
        figsize: Tuple[int, int] = (10, 6),
        **kwargs
    ) -> plt.Figure:
        """
        Plot metric as function of binned feature value.

        Args:
            feature: Feature to plot on x-axis
            metric_func: Metric function to compute for each bin
            bins: Number of bins
            figsize: Figure size
            **kwargs: Additional arguments for plt.plot

        Returns:
            matplotlib Figure object

        Example:
            >>> from grading_calibration.metrics import expected_calibration_error
            >>> fig = analyzer.plot_feature_vs_metric(
            ...     'answer_length',
            ...     expected_calibration_error,
            ...     bins=5
            ... )
            >>> fig.savefig('ece_vs_answer_length.png')
        """
        df = self.compute_features()

        # Bin feature
        try:
            df['feature_binned'] = pd.qcut(df[feature], bins, duplicates='drop')
        except ValueError:
            # If qcut fails (e.g., too many duplicates), use cut instead
            df['feature_binned'] = pd.cut(df[feature], bins)

        # Compute metric for each bin
        bin_metrics = []
        bin_labels = []

        for bin_value in df['feature_binned'].dropna().unique():
            bin_indices = df[df['feature_binned'] == bin_value].index
            bin_predictions = [p for p in self.predictions if p.instance_id in bin_indices]

            if len(bin_predictions) > 0:
                try:
                    metric_value = metric_func(bin_predictions)
                    bin_metrics.append(metric_value)
                    bin_labels.append(str(bin_value))
                except Exception:
                    # Skip bins that fail
                    continue

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(bin_metrics)), bin_metrics, marker='o', **kwargs)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_xlabel(feature)
        ax.set_ylabel(metric_func.__name__)
        ax.set_title(f"{metric_func.__name__} vs {feature}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
