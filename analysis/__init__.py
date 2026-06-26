"""
Analysis framework for understanding confidence and calibration drivers.

This package provides tools for feature extraction, feature analysis, commonality
detection, and mechanistic investigation of calibration quality.

Modules:
    feature_extraction: Extract text, metadata, and confidence features
    feature_analysis: Analyze feature-metric relationships
    commonality_analysis: Find patterns in high/low confidence instances
    mechanistic_analysis: Causal analysis and recommendations

Example:
    >>> from analysis import FeatureExtractor, FeatureAnalyzer
    >>> extractor = FeatureExtractor()
    >>> features_df = extractor.extract_batch(instances, predictions)
    >>> analyzer = FeatureAnalyzer(predictions, instances)
    >>> corrs = analyzer.feature_correlations(target='raw_confidence')
"""

from .feature_extraction import FeatureExtractor
from .feature_analysis import FeatureAnalyzer
from .commonality_analysis import CommonalityAnalyzer
from .mechanistic_analysis import MechanisticAnalyzer

__all__ = [
    'FeatureExtractor',
    'FeatureAnalyzer',
    'CommonalityAnalyzer',
    'MechanisticAnalyzer'
]
