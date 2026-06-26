import pytest

from analysis.feature_extraction import FeatureExtractor
from grading_calibration.domain.models import GradingInstance, PredictionResult


def test_self_consistency_uses_multi_response_only_and_computes_rates_and_cv():
    extractor = FeatureExtractor()

    instance = GradingInstance(
        id="inst1",
        question="Q?",
        student_answer="A.",
        rubric_item="R.",
        ground_truth_label=1,
        metadata={"question_id": "q1", "student_id": "s1", "rubric_id": "r1"},
    )

    # Two responses with different lengths to distinguish mean(count/length) vs pooled totals.
    # - "maybe": 1 word, uncertainty_count=1
    # - "definitely yes indeed": 3 words, confidence_count=1
    prediction = PredictionResult(
        instance_id="inst1",
        prediction=1,
        raw_confidence=0.5,
        ground_truth=1,
        metadata={
            "llm_response": "this single response should be ignored",
            "llm_responses": ["maybe", "definitely yes indeed"],
        },
    )

    feats = extractor.extract_all_features(instance, prediction)

    # Single-response features should not be extracted when llm_responses are present
    assert "response_length" not in feats
    assert "response_uncertainty_count" not in feats
    assert "response_uncertainty_ratio" not in feats

    # Aggregated features should exist
    assert feats["response_n_samples"] == 2
    assert feats["response_length_mean"] == pytest.approx(2.0)
    assert feats["response_length_std"] == pytest.approx(1.0)
    assert feats["response_length_cv"] == pytest.approx(0.5)

    # Counts: [1, 0] for both uncertainty and confidence markers -> mean=0.5, std=0.5, cv=1.0
    assert feats["response_uncertainty_count_mean"] == pytest.approx(0.5)
    assert feats["response_uncertainty_count_std"] == pytest.approx(0.5)
    assert feats["response_uncertainty_count_cv"] == pytest.approx(1.0)
    assert feats["response_confidence_count_mean"] == pytest.approx(0.5)
    assert feats["response_confidence_count_std"] == pytest.approx(0.5)
    assert feats["response_confidence_count_cv"] == pytest.approx(1.0)

    # Per-response mean rates:
    # - uncertainty: (1/1*100 + 0/3*100)/2 = 50
    # - confidence: (0/1*100 + 1/3*100)/2 = 16.666...
    assert feats["response_uncertainty_ratio_mean"] == pytest.approx(50.0)
    assert feats["response_confidence_ratio_mean"] == pytest.approx((0.0 + (1.0 / 3.0 * 100.0)) / 2.0)

    # Pooled rates:
    # total_words=4, total_uncertainty=1, total_confidence=1
    assert feats["response_total_words"] == pytest.approx(4.0)
    assert feats["response_uncertainty_count_total"] == pytest.approx(1.0)
    assert feats["response_confidence_count_total"] == pytest.approx(1.0)
    assert feats["response_uncertainty_ratio_pooled"] == pytest.approx(25.0)
    assert feats["response_confidence_ratio_pooled"] == pytest.approx(25.0)

