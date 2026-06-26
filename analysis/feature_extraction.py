"""
Feature extraction for calibration analysis.

This module provides the FeatureExtractor class for extracting various features
from GradingInstance and PredictionResult objects, including:
- Text features (length, complexity, readability) for all text fields
- Metadata features (question_id, rubric_id, student_id)
- Confidence features (raw, calibrated, bins)
- Prediction features (correctness, error types)
- Vote features (for self-consistency method)
- LLM response features (linguistic markers, uncertainty/confidence language)

Example:
    >>> from analysis.feature_extraction import FeatureExtractor
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract_all_features(instance, prediction)
    >>> # Or extract for batch
    >>> features_df = extractor.extract_batch(instances, predictions)
"""

from tqdm import tqdm

import re
import sys
import pprint
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from grading_calibration.domain.models import GradingInstance, PredictionResult


# Linguistic marker word lists
UNCERTAINTY_MARKERS = [
    "maybe", "possibly", "might", "could be", "i think", "perhaps",
    "probably", "likely", "uncertain", "unsure", "not sure", "unclear",
    "seems", "appears", "may be", "potentially"
]

CONFIDENCE_MARKERS = [
    "definitely", "clearly", "certainly", "must", "obviously",
    "absolutely", "surely", "undoubtedly", "without doubt", "no doubt",
    "confident", "certain", "positive", "sure"
]


def count_syllables(word: str) -> int:
    """
    Count syllables in a word using vowel-based heuristic.

    This is a simple heuristic that works reasonably well for English:
    - Count vowel groups (consecutive vowels = 1 syllable)
    - Subtract 1 for silent 'e' at end
    - Minimum 1 syllable per word

    Args:
        word: Word to count syllables in

    Returns:
        Estimated syllable count (minimum 1)

    Example:
        >>> count_syllables("hello")  # hel-lo
        2
        >>> count_syllables("through")  # through (silent gh, vowel group 'ou')
        1
        >>> count_syllables("area")  # a-re-a
        3
    """
    word = word.lower().strip()
    if not word:
        return 1

    vowels = 'aeiou'
    syllable_count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel

    # Handle silent 'e' at end (e.g., "make" = 1 syllable, not 2)
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1

    # Every word has at least 1 syllable
    return max(1, syllable_count)


class FeatureExtractor:
    """
    Extract features from GradingInstance and PredictionResult for analysis.

    This class provides methods to extract various types of features that can
    be used for correlation analysis, stratification, and mechanistic investigation.

    Example:
        >>> extractor = FeatureExtractor()
        >>> text_features = extractor.extract_text_features(instance)
        >>> confidence_features = extractor.extract_confidence_features(prediction)
    """

    def __init__(
        self,
        *,
        enable_perplexity: bool = False,
        text_feature_mode: str = "full",
        perplexity_model: str = "gpt2",
        perplexity_device: str = "cuda:0",
        perplexity_batch_size: int = 32,
        perplexity_max_length: int = 512,
        perplexity_min_tokens: int = 3,
        perplexity_progress: bool = True,
        perplexity_debug: bool = False,
        perplexity_stream_responses: bool = False,
        perplexity_response_buffer_max: int = 2000,
    ) -> None:
        self.enable_perplexity = enable_perplexity
        self.text_feature_mode = text_feature_mode
        self.perplexity_model = perplexity_model
        self.perplexity_device = perplexity_device
        self.perplexity_batch_size = perplexity_batch_size
        self.perplexity_max_length = perplexity_max_length
        self.perplexity_min_tokens = perplexity_min_tokens
        self.perplexity_progress = perplexity_progress
        self.perplexity_debug = perplexity_debug
        self.perplexity_stream_responses = perplexity_stream_responses
        self.perplexity_response_buffer_max = perplexity_response_buffer_max
        self._perplexity_scorer: Optional[_PerplexityScorer] = None

        if text_feature_mode == "core":
            self._text_feature_names = [
                "length",
                "avg_word_length",
                "lexical_diversity",
                "readability",
            ]
        elif text_feature_mode == "full":
            self._text_feature_names = [
                "length",
                "char_count",
                "sentence_count",
                "avg_word_length",
                "lexical_diversity",
                "readability",
            ]
        else:
            raise ValueError(
                f"Invalid text_feature_mode: {text_feature_mode}. Use 'core' or 'full'."
            )
        self._text_feature_names_set = set(self._text_feature_names)

    def _get_perplexity_scorer(self) -> Optional["_PerplexityScorer"]:
        if not self.enable_perplexity:
            return None
        if self._perplexity_scorer is None:
            self._perplexity_scorer = _PerplexityScorer(
                model_name=self.perplexity_model,
                device=self.perplexity_device,
                batch_size=self.perplexity_batch_size,
                max_length=self.perplexity_max_length,
                min_tokens=self.perplexity_min_tokens,
                show_progress=self.perplexity_progress,
                debug=self.perplexity_debug,
            )
        return self._perplexity_scorer

    def _log_empty_text(
        self,
        label: str,
        instance: GradingInstance,
        prediction: PredictionResult,
        *,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not (self.enable_perplexity and self.perplexity_debug):
            return
        payload = {
            "label": label,
            "instance": instance.to_dict(),
            "prediction": prediction.to_dict(),
        }
        if extra:
            payload["extra"] = extra
        print("[perplexity debug empty] Full observation:", file=sys.stderr)
        print(pprint.pformat(payload, width=140), file=sys.stderr)

    def _extract_text_field_features(
        self,
        text: str,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract text features for any text field with a given prefix.

        This is a generic helper method that computes standard text metrics
        and prefixes all feature names with the specified prefix.

        Args:
            text: The text content to analyze
            prefix: Prefix for feature names (e.g., 'answer_', 'question_', 'rubric_', 'response_')

        Returns:
            Dict with prefixed features:
                - {prefix}length: Word count
                - {prefix}char_count: Character count
                - {prefix}sentence_count: Number of sentences
                - {prefix}avg_word_length: Average word length
                - {prefix}lexical_diversity: Unique words / total words
                - {prefix}readability: Flesch reading ease approximation

        Example:
            >>> features = extractor._extract_text_field_features("Hello world.", "test_")
            >>> print(features['test_length'])  # 2
        """
        features = {}
        feature_set = self._text_feature_names_set
        
        # Handle empty or None text
        if not text or not isinstance(text, str):
            if "length" in feature_set:
                features[f'{prefix}length'] = 0
            if "char_count" in feature_set:
                features[f'{prefix}char_count'] = 0
            if "sentence_count" in feature_set:
                features[f'{prefix}sentence_count'] = 0
            if "avg_word_length" in feature_set:
                features[f'{prefix}avg_word_length'] = 0.0
            if "lexical_diversity" in feature_set:
                features[f'{prefix}lexical_diversity'] = 0.0
            if "readability" in feature_set:
                features[f'{prefix}readability'] = 0.0
            return features

        if "char_count" in feature_set:
            features[f'{prefix}char_count'] = len(text)

        # Sentence count (simple approximation using period, question mark, exclamation)
        sentence_count = None
        if "sentence_count" in feature_set or "readability" in feature_set:
            sentences = re.split(r'[.!?]+', text)
            sentence_count = max(1, len([s for s in sentences if s.strip()]))
            if "sentence_count" in feature_set:
                features[f'{prefix}sentence_count'] = sentence_count

        need_words = any(
            f in feature_set for f in ("length", "avg_word_length", "lexical_diversity", "readability")
        )
        words = text.split() if need_words else []
        if "length" in feature_set:
            features[f'{prefix}length'] = len(words)
        if len(words) > 0:
            if "avg_word_length" in feature_set:
                features[f'{prefix}avg_word_length'] = np.mean([len(w) for w in words])
            if "lexical_diversity" in feature_set:
                unique_words = set(w.lower() for w in words)
                features[f'{prefix}lexical_diversity'] = len(unique_words) / len(words)
        else:
            if "avg_word_length" in feature_set:
                features[f'{prefix}avg_word_length'] = 0.0
            if "lexical_diversity" in feature_set:
                features[f'{prefix}lexical_diversity'] = 0.0

        # Readability score (Flesch reading ease)
        if "readability" in feature_set:
            if sentence_count is None:
                sentences = re.split(r'[.!?]+', text)
                sentence_count = max(1, len([s for s in sentences if s.strip()]))
            length_val = len(words)
            if sentence_count > 0 and length_val > 0:
                avg_sentence_length = length_val / sentence_count

                # Compute actual syllable count using vowel-based heuristic
                total_syllables = sum(count_syllables(word) for word in words)
                avg_syllables_per_word = total_syllables / length_val if length_val > 0 else 1.0

                # Flesch Reading Ease formula
                readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
                features[f'{prefix}readability'] = max(0.0, min(100.0, readability))
            else:
                features[f'{prefix}readability'] = 0.0

        return features

    def _count_linguistic_markers(
        self,
        text: str,
        markers: List[str]
    ) -> int:
        """
        Count occurrences of linguistic markers in text.

        Args:
            text: Text to search in
            markers: List of marker phrases to count

        Returns:
            Total count of marker occurrences
        """
        if not text:
            return 0
        text_lower = text.lower()
        count = 0
        for marker in markers:
            count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        return count

    def extract_text_features(self, instance: GradingInstance) -> Dict[str, float]:
        """
        Extract text-based features from grading instance for all text fields.

        Analyzes the student answer, question, and rubric text to extract
        quantitative features related to length, complexity, and readability.
        Each text field gets its own set of prefixed features.

        Args:
            instance: GradingInstance object

        Returns:
            Dict with features for each text field (answer_, question_, rubric_):
                - {prefix}length: Word count
                - {prefix}char_count: Character count
                - {prefix}sentence_count: Number of sentences
                - {prefix}avg_word_length: Average word length
                - {prefix}lexical_diversity: Unique words / total words
                - {prefix}readability: Flesch reading ease approximation

            Legacy features for backward compatibility:
                - answer_sentence_count (same as above)
                - avg_word_length (answer's avg word length)
                - lexical_diversity (answer's lexical diversity)
                - readability_score (answer's readability)

        Example:
            >>> features = extractor.extract_text_features(instance)
            >>> print(f"Answer length: {features['answer_length']} words")
            >>> print(f"Question length: {features['question_length']} words")
        """
        features = {}

        # Extract features for each text field using the generic helper
        answer_features = self._extract_text_field_features(
            instance.student_answer, 'answer_'
        )
        question_features = self._extract_text_field_features(
            instance.question, 'question_'
        )
        rubric_features = self._extract_text_field_features(
            instance.rubric_item, 'rubric_'
        )

        features.update(answer_features)
        features.update(question_features)
        features.update(rubric_features)

        # Add legacy feature names for backward compatibility
        if 'answer_avg_word_length' in features:
            features['avg_word_length'] = features['answer_avg_word_length']
        if 'answer_lexical_diversity' in features:
            features['lexical_diversity'] = features['answer_lexical_diversity']
        if 'answer_readability' in features:
            features['readability_score'] = features['answer_readability']

        return features

    def extract_metadata_features(
        self,
        instance: GradingInstance,
        prediction: PredictionResult
    ) -> Dict[str, str]:
        """
        Extract categorical metadata from instance and prediction.

        Args:
            instance: GradingInstance object
            prediction: PredictionResult object

        Returns:
            Dict with features:
                - question_id: Question identifier
                - rubric_id: Rubric item identifier
                - student_id: Student identifier
                - split: Dataset split (train/valid/test)
                - has_answer_key: Boolean indicating if answer key exists
                - has_rubric: Boolean indicating if rubric exists

        Example:
            >>> meta = extractor.extract_metadata_features(instance, prediction)
            >>> print(f"Question: {meta['question_id']}, Rubric: {meta['rubric_id']}")
        """
        features = {}

        # From instance metadata
        features['question_id'] = instance.metadata.get('question_id', 'unknown')
        features['rubric_id'] = instance.metadata.get('rubric_id', 'unknown')
        features['student_id'] = instance.metadata.get('student_id', 'unknown')

        # From prediction metadata
        features['split'] = prediction.metadata.get('split', 'unknown')

        # Instance properties
        features['has_answer_key'] = instance.has_answer_key() if hasattr(instance, 'has_answer_key') else False
        features['has_rubric'] = instance.has_rubric() if hasattr(instance, 'has_rubric') else True

        return features

    def extract_confidence_features(self, prediction: PredictionResult) -> Dict[str, float]:
        """
        Extract confidence-related features from prediction.

        Args:
            prediction: PredictionResult object

        Returns:
            Dict with features:
                - raw_confidence: Raw confidence score [0, 1]
                - calibrated_confidence: Calibrated score (if available)
                - confidence_delta: calibrated - raw
                - confidence_bin: Binned confidence (0-9)
                - is_high_confidence: raw_confidence > 0.8
                - is_medium_confidence: 0.5 <= raw_confidence <= 0.8
                - is_low_confidence: raw_confidence < 0.5

        Example:
            >>> conf_features = extractor.extract_confidence_features(prediction)
            >>> if conf_features['is_high_confidence']:
            ...     print("High confidence prediction")
        """
        features = {}

        features['raw_confidence'] = float(prediction.raw_confidence)

        if prediction.calibrated_confidence is not None:
            features['calibrated_confidence'] = float(prediction.calibrated_confidence)
            features['confidence_delta'] = prediction.calibrated_confidence - prediction.raw_confidence
        else:
            features['calibrated_confidence'] = np.nan
            features['confidence_delta'] = np.nan

        # REMOVED: confidence_bin feature (use CalibrationBinData for binning)
        # Binning should only happen via evaluation.calibration_bins.compute_calibration_bins()
        # to ensure consistency across all metrics and analysis.
        # If bin assignment needed for stratification, compute externally using CalibrationBinData.

        # Confidence level categories (keep these - they're simple thresholds, not binning)
        features['is_high_confidence'] = float(prediction.raw_confidence > 0.8)
        features['is_medium_confidence'] = float(0.5 <= prediction.raw_confidence <= 0.8)
        features['is_low_confidence'] = float(prediction.raw_confidence < 0.5)

        return features

    def extract_prediction_features(self, prediction: PredictionResult) -> Dict[str, float]:
        """
        Extract prediction and correctness features.

        Args:
            prediction: PredictionResult object

        Returns:
            Dict with features:
                - prediction: Predicted label (0 or 1)
                - ground_truth: True label (0 or 1)
                - is_correct: Boolean correctness
                - is_false_positive: Predicted 1, truth 0
                - is_false_negative: Predicted 0, truth 1

        Example:
            >>> pred_features = extractor.extract_prediction_features(prediction)
            >>> if pred_features['is_false_positive']:
            ...     print("False positive error")
        """
        features = {}

        features['prediction'] = int(prediction.prediction)
        features['ground_truth'] = int(prediction.ground_truth)
        features['is_correct'] = float(prediction.is_correct())

        # Error type analysis
        features['is_false_positive'] = float(
            prediction.prediction == 1 and prediction.ground_truth == 0
        )
        features['is_false_negative'] = float(
            prediction.prediction == 0 and prediction.ground_truth == 1
        )

        return features

    def extract_vote_features(self, prediction: PredictionResult) -> Dict[str, float]:
        """
        Extract features from voting (self-consistency method).

        Looks for vote_counts, yes_prob, no_prob in prediction metadata.
        Returns NaN for features if voting data not available.

        Args:
            prediction: PredictionResult object

        Returns:
            Dict with features:
                - vote_agreement: max(yes, no) / total
                - vote_entropy: Entropy of vote distribution
                - yes_prob: Proportion of yes votes
                - no_prob: Proportion of no votes
                - vote_margin: |yes_prob - no_prob|
                - n_votes: Total number of votes

        Example:
            >>> vote_features = extractor.extract_vote_features(prediction)
            >>> if vote_features['vote_agreement'] > 0.9:
            ...     print("Very high agreement in voting")
        """
        features = {}

        # Check if vote data available in metadata
        if 'vote_counts' in prediction.metadata:
            vote_counts = prediction.metadata['vote_counts']
            yes_count = vote_counts.get('yes', 0)
            no_count = vote_counts.get('no', 0)
            total = yes_count + no_count

            if total > 0:
                yes_prob = yes_count / total
                no_prob = no_count / total

                features['yes_prob'] = yes_prob
                features['no_prob'] = no_prob
                features['vote_margin'] = abs(yes_prob - no_prob)
                features['vote_agreement'] = max(yes_prob, no_prob)

                # Entropy: -sum(p * log2(p))
                probs = [p for p in [yes_prob, no_prob] if p > 0]
                features['vote_entropy'] = -sum(p * np.log2(p) for p in probs)

                features['n_votes'] = float(total)
            else:
                # No votes recorded
                for key in ['yes_prob', 'no_prob', 'vote_margin', 'vote_agreement', 'vote_entropy', 'n_votes']:
                    features[key] = np.nan
        else:
            # Vote features not available (not self-consistency method)
            for key in ['yes_prob', 'no_prob', 'vote_margin', 'vote_agreement', 'vote_entropy', 'n_votes']:
                features[key] = np.nan

        return features

    def extract_llm_response_features(self, prediction: PredictionResult) -> Dict[str, float]:
        """
        Extract features from LLM response text.

        Analyzes the LLM's response for text metrics and linguistic markers
        that may indicate uncertainty or confidence in the grading decision.

        Args:
            prediction: PredictionResult object with 'llm_response' in metadata

        Returns:
            Dict with features:
                - response_length: Word count in response
                - response_char_count: Character count
                - response_sentence_count: Number of sentences
                - response_avg_word_length: Average word length
                - response_lexical_diversity: Unique words / total words
                - response_readability: Flesch reading ease approximation
                - response_uncertainty_count: Count of uncertainty markers
                - response_confidence_count: Count of confidence markers
                - response_uncertainty_ratio: Uncertainty markers / word count
                - response_confidence_ratio: Confidence markers / word count

        Example:
            >>> response_features = extractor.extract_llm_response_features(prediction)
            >>> if response_features['response_uncertainty_count'] > 3:
            ...     print("LLM expressed high uncertainty")
        """
        features = {}

        # Get LLM response from metadata
        llm_response = prediction.metadata.get('llm_response', '')
        
        # Extract standard text features with response_ prefix
        text_features = self._extract_text_field_features(llm_response, 'response_')
        features.update(text_features)

        # Count linguistic markers
        features['response_uncertainty_count'] = self._count_linguistic_markers(
            llm_response, UNCERTAINTY_MARKERS
        )
        features['response_confidence_count'] = self._count_linguistic_markers(
            llm_response, CONFIDENCE_MARKERS
        )

        # Compute ratios (per 100 words for interpretability)
        word_count = features['response_length']
        if word_count > 0:
            features['response_uncertainty_ratio'] = (
                features['response_uncertainty_count'] / word_count * 100
            )
            features['response_confidence_ratio'] = (
                features['response_confidence_count'] / word_count * 100
            )
        else:
            features['response_uncertainty_ratio'] = 0.0
            features['response_confidence_ratio'] = 0.0

        return features

    def extract_multi_response_features(self, prediction: PredictionResult) -> Dict[str, float]:
        """
        Extract aggregated features from multiple LLM responses (self-consistency).

        For self-consistency methods that sample multiple responses, this computes
        mean and std of response features across all samples. The variance features
        are particularly interesting as they may indicate model uncertainty.

        Args:
            prediction: PredictionResult with 'llm_responses' list in metadata

        Returns:
            Dict with aggregated features:
                - response_*_mean: Mean feature value across responses
                - response_*_std: Std dev across responses (0 if single response)

        Example:
            >>> multi_features = extractor.extract_multi_response_features(prediction)
            >>> if multi_features['response_length_std'] > 50:
            ...     print("High variance in response lengths - potential uncertainty")
        """
        features = {}

        # Get list of responses from metadata
        llm_responses = prediction.metadata.get('llm_responses', [])

        if not llm_responses or len(llm_responses) == 0:
            # No multi-response data - return empty (single response features handled elsewhere)
            return features

        # Base feature names to aggregate (excluding ratios which we recompute)
        base_features = list(self._text_feature_names)
        marker_features = ['uncertainty_count', 'confidence_count']

        # Extract features from each response
        all_response_features = []
        per_response_uncertainty_rates = []
        per_response_confidence_rates = []
        total_words = 0.0
        total_uncertainty = 0.0
        total_confidence = 0.0
        for resp in llm_responses:
            resp_features = self._extract_text_field_features(resp, '')
            # Add marker counts
            resp_features['uncertainty_count'] = self._count_linguistic_markers(
                resp, UNCERTAINTY_MARKERS
            )
            resp_features['confidence_count'] = self._count_linguistic_markers(
                resp, CONFIDENCE_MARKERS
            )
            all_response_features.append(resp_features)

            # Compute per-response rates and pooled totals
            n_words = float(resp_features.get('length', 0.0) or 0.0)
            n_unc = float(resp_features.get('uncertainty_count', 0.0) or 0.0)
            n_conf = float(resp_features.get('confidence_count', 0.0) or 0.0)

            if np.isfinite(n_words) and n_words > 0:
                per_response_uncertainty_rates.append(n_unc / n_words * 100.0)
                per_response_confidence_rates.append(n_conf / n_words * 100.0)

            if np.isfinite(n_words):
                total_words += max(0.0, n_words)
            if np.isfinite(n_unc):
                total_uncertainty += max(0.0, n_unc)
            if np.isfinite(n_conf):
                total_confidence += max(0.0, n_conf)

        # Compute mean/std/cv for each feature
        for feat_name in base_features + marker_features:
            values = [f.get(feat_name, 0.0) for f in all_response_features]
            values = [v for v in values if v is not None and np.isfinite(v)]

            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values)) if len(values) > 1 else 0.0
                features[f'response_{feat_name}_mean'] = mean_val
                features[f'response_{feat_name}_std'] = std_val
                if np.isfinite(mean_val) and mean_val > 0:
                    features[f'response_{feat_name}_cv'] = float(std_val / mean_val)
                else:
                    features[f'response_{feat_name}_cv'] = np.nan
            else:
                features[f'response_{feat_name}_mean'] = np.nan
                features[f'response_{feat_name}_std'] = np.nan
                features[f'response_{feat_name}_cv'] = np.nan

        # Mean per-response rates (mean(count/words)), expressed per 100 words
        features['response_uncertainty_ratio_mean'] = (
            float(np.mean(per_response_uncertainty_rates)) if per_response_uncertainty_rates else 0.0
        )
        features['response_confidence_ratio_mean'] = (
            float(np.mean(per_response_confidence_rates)) if per_response_confidence_rates else 0.0
        )
        # Std of per-response rates (variability in hedging language across samples)
        features['response_uncertainty_ratio_std'] = (
            float(np.std(per_response_uncertainty_rates)) if len(per_response_uncertainty_rates) > 1 else 0.0
        )
        features['response_confidence_ratio_std'] = (
            float(np.std(per_response_confidence_rates)) if len(per_response_confidence_rates) > 1 else 0.0
        )

        # Pooled rates (total_count / total_words), expressed per 100 words
        features['response_total_words'] = float(total_words)
        features['response_uncertainty_count_total'] = float(total_uncertainty)
        features['response_confidence_count_total'] = float(total_confidence)
        if total_words > 0:
            features['response_uncertainty_ratio_pooled'] = float(total_uncertainty / total_words * 100.0)
            features['response_confidence_ratio_pooled'] = float(total_confidence / total_words * 100.0)
        else:
            features['response_uncertainty_ratio_pooled'] = 0.0
            features['response_confidence_ratio_pooled'] = 0.0

        # Number of responses (useful metadata)
        features['response_n_samples'] = len(llm_responses)

        return features

    def extract_all_features(
        self,
        instance: GradingInstance,
        prediction: PredictionResult
    ) -> Dict[str, float]:
        """
        Extract all features into single dictionary.

        Combines text, metadata, confidence, prediction, vote, and LLM response features.

        Args:
            instance: GradingInstance object
            prediction: PredictionResult object

        Returns:
            Dict with all features combined

        Example:
            >>> all_features = extractor.extract_all_features(instance, prediction)
            >>> print(f"Total features: {len(all_features)}")
        """
        features = {}

        features.update(self.extract_text_features(instance))
        features.update(self.extract_metadata_features(instance, prediction))
        features.update(self.extract_confidence_features(prediction))
        features.update(self.extract_prediction_features(prediction))
        features.update(self.extract_vote_features(prediction))

        # For self-consistency (multi-response): extract aggregated features and avoid using
        # an arbitrary single sampled response for "response_*" features.
        multi_response_features = self.extract_multi_response_features(prediction)
        if multi_response_features:
            features.update(multi_response_features)
        else:
            features.update(self.extract_llm_response_features(prediction))

        return features

    def extract_batch(
        self,
        instances: List[GradingInstance],
        predictions: List[PredictionResult]
    ) -> pd.DataFrame:
        """
        Extract features for batch of instances.

        Args:
            instances: List of GradingInstance objects
            predictions: List of PredictionResult objects (matched by instance_id)

        Returns:
            DataFrame with one row per instance, columns for all features,
            indexed by instance_id

        Example:
            >>> features_df = extractor.extract_batch(instances, predictions)
            >>> print(features_df.head())
            >>> # Analyze correlations
            >>> print(features_df.corr()['raw_confidence'].sort_values())
        """
        # Create lookup for instances
        instance_dict = {inst.id: inst for inst in instances}

        scorer = self._get_perplexity_scorer()
        stream_response_ppl = False
        if scorer is not None:
            if self.perplexity_stream_responses:
                stream_response_ppl = True
            else:
                total_response_texts = 0
                for pred in predictions:
                    llm_responses = pred.metadata.get("llm_responses")
                    if llm_responses:
                        total_response_texts += len(llm_responses)
                    else:
                        total_response_texts += 1
                    if total_response_texts > self.perplexity_response_buffer_max:
                        stream_response_ppl = True
                        break

        # Extract features for each prediction with progress bar
        rows = []
        answer_texts: List[str] = []
        response_text_groups: List[List[str]] = []
        response_ppl_by_row: List[float] = []
        response_tokens_by_row: List[float] = []
        for pred in tqdm(predictions, desc="Extracting features", unit="instance"):
            if pred.instance_id in instance_dict:
                inst = instance_dict[pred.instance_id]
                features = self.extract_all_features(inst, pred)
                features['instance_id'] = pred.instance_id
                rows.append(features)
                answer_text = "" if inst.student_answer is None else str(inst.student_answer)
                if not answer_text.strip():
                    self._log_empty_text("student_answer_empty", inst, pred)
                answer_texts.append(answer_text)
                llm_responses = pred.metadata.get("llm_responses", [])
                if stream_response_ppl and scorer is not None:
                    if llm_responses:
                        cleaned_responses = ["" if r is None else str(r) for r in llm_responses]
                        empty_indices = [
                            i for i, r in enumerate(cleaned_responses) if not str(r).strip()
                        ]
                        if empty_indices:
                            self._log_empty_text(
                                "llm_responses_empty",
                                inst,
                                pred,
                                extra={"empty_indices": empty_indices, "responses": cleaned_responses},
                            )
                        group = [r for r in cleaned_responses if r and str(r).strip()]
                    else:
                        llm_response = pred.metadata.get("llm_response", "")
                        llm_response = "" if llm_response is None else str(llm_response)
                        if not llm_response.strip():
                            self._log_empty_text("llm_response_empty", inst, pred)
                        group = [llm_response] if llm_response and llm_response.strip() else []

                    if not group:
                        response_ppl_by_row.append(np.nan)
                        response_tokens_by_row.append(0.0)
                    else:
                        group_ppl, group_tokens = scorer.score_texts(group)
                        if np.isfinite(group_ppl).any():
                            response_ppl_by_row.append(float(np.nanmean(group_ppl)))
                        else:
                            response_ppl_by_row.append(np.nan)
                        response_tokens_by_row.append(
                            float(np.nanmean(group_tokens)) if len(group_tokens) > 0 else 0.0
                        )
                else:
                    if llm_responses:
                        cleaned_responses = ["" if r is None else str(r) for r in llm_responses]
                        empty_indices = [
                            i for i, r in enumerate(cleaned_responses) if not str(r).strip()
                        ]
                        if empty_indices:
                            self._log_empty_text(
                                "llm_responses_empty",
                                inst,
                                pred,
                                extra={"empty_indices": empty_indices, "responses": cleaned_responses},
                            )
                        response_text_groups.append([r for r in cleaned_responses if r])
                    else:
                        llm_response = pred.metadata.get("llm_response", "")
                        llm_response = "" if llm_response is None else str(llm_response)
                        if not llm_response.strip():
                            self._log_empty_text("llm_response_empty", inst, pred)
                        response_text_groups.append([llm_response])

        df = pd.DataFrame(rows)

        # Add perplexity features if enabled
        if scorer is not None and not df.empty:
            answer_ppl, answer_tokens = scorer.score_texts(answer_texts)
            df["answer_perplexity"] = answer_ppl
            df["answer_perplexity_n_tokens"] = answer_tokens

            if stream_response_ppl:
                df["response_perplexity"] = np.asarray(response_ppl_by_row, dtype=float)
                df["response_perplexity_n_tokens"] = np.asarray(response_tokens_by_row, dtype=float)
            else:
                flat_responses: List[str] = []
                spans: List[Tuple[int, int]] = []
                for group in response_text_groups:
                    start = len(flat_responses)
                    flat_responses.extend(group)
                    spans.append((start, len(flat_responses)))
                resp_ppl, resp_tokens = scorer.score_texts(flat_responses)
                resp_mean = []
                resp_tokens_mean = []
                for start, end in spans:
                    if end <= start:
                        resp_mean.append(np.nan)
                        resp_tokens_mean.append(0.0)
                        continue
                    group_ppl = resp_ppl[start:end]
                    group_tokens = resp_tokens[start:end]
                    if np.isfinite(group_ppl).any():
                        resp_mean.append(float(np.nanmean(group_ppl)))
                    else:
                        resp_mean.append(np.nan)
                    resp_tokens_mean.append(float(np.nanmean(group_tokens)) if len(group_tokens) > 0 else 0.0)
                df["response_perplexity"] = np.asarray(resp_mean, dtype=float)
                df["response_perplexity_n_tokens"] = np.asarray(resp_tokens_mean, dtype=float)

        # Set instance_id as index
        if 'instance_id' in df.columns and not df.empty:
            df.set_index('instance_id', inplace=True)

        return df

    def get_numerical_features(self, features_df: pd.DataFrame) -> List[str]:
        """
        Get list of numerical feature names from DataFrame.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            List of column names with numerical dtypes

        Example:
            >>> numerical_cols = extractor.get_numerical_features(features_df)
            >>> corr_matrix = features_df[numerical_cols].corr()
        """
        return features_df.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_features(self, features_df: pd.DataFrame) -> List[str]:
        """
        Get list of categorical feature names from DataFrame.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            List of column names with categorical/object dtypes

        Example:
            >>> categorical_cols = extractor.get_categorical_features(features_df)
            >>> for col in categorical_cols:
            ...     print(f"{col}: {features_df[col].nunique()} unique values")
        """
        return features_df.select_dtypes(include=['object', 'category']).columns.tolist()

class _PerplexityScorer:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
        min_tokens: int,
        show_progress: bool,
        debug: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Perplexity feature requires torch and transformers. "
                "Install them or disable --enable-perplexity."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.model.to(self.device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.min_tokens = max(1, int(min_tokens))
        self.show_progress = show_progress
        self.debug = bool(debug)
        self.debug_max_items = 3
        self.debug_max_chars = 300

    def _debug_batch_texts(self, batch_texts: List[str], start_idx: int) -> None:
        if not batch_texts or not self.debug:
            return
        max_items = min(self.debug_max_items, len(batch_texts))
        for offset in range(max_items):
            text = batch_texts[offset]
            text = text if isinstance(text, str) else ""
            snippet = text[: self.debug_max_chars]
            suffix = "" if len(text) <= self.debug_max_chars else "...[truncated]"
            print(
                f"[perplexity debug] idx={start_idx + offset} text={snippet!r}{suffix}",
                file=sys.stderr,
            )
        for offset, text in enumerate(batch_texts):
            if not str(text).strip():
                print(
                    f"[perplexity debug] idx={start_idx + offset} EMPTY_OR_WHITESPACE",
                    file=sys.stderr,
                )

    def score_texts(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return self._score_texts_impl(texts, self.device, self.batch_size)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or self.device == "cpu":
                raise
            self.torch.cuda.empty_cache()
            if self.batch_size > 1:
                return self._score_texts_impl(texts, self.device, max(1, self.batch_size // 2))
            self.device = "cpu"
            self.model.to(self.device)
            return self._score_texts_impl(texts, self.device, max(1, min(8, self.batch_size)))

    def _score_texts_impl(self, texts: List[str], device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        n = len(texts)
        perplexities = np.full(n, np.nan, dtype=float)
        token_counts = np.zeros(n, dtype=float)
        iterator = range(0, n, batch_size)
        if self.show_progress and n > batch_size:
            iterator = tqdm(iterator, desc="Perplexity", unit="batch")
        for start in iterator:
            end = min(n, start + batch_size)
            batch_texts = [t if isinstance(t, str) else "" for t in texts[start:end]]
            self._debug_batch_texts(batch_texts, start)
            if all(not t.strip() for t in batch_texts):
                raise ValueError(
                    f"Perplexity batch has only empty/whitespace texts (rows {start}:{end})."
                )
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if input_ids.numel() == 0 or input_ids.shape[-1] == 0:
                raise ValueError(
                    f"Perplexity batch tokenization produced empty tensors (rows {start}:{end})."
                )
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_mask = attention_mask[:, 1:]
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                nll = nll * shift_mask
                token_count = shift_mask.sum(dim=1)
                token_counts[start:end] = token_count.detach().cpu().numpy().astype(float)
                valid = token_count >= self.min_tokens
                if valid.any():
                    mean_nll = nll.sum(dim=1) / token_count.clamp(min=1)
                    ppl = torch.exp(mean_nll)
                    ppl = ppl.detach().cpu().numpy()
                    valid_idx = valid.detach().cpu().numpy().astype(bool)
                    perplexities[start:end][valid_idx] = ppl[valid_idx]
        return perplexities, token_counts
