from __future__ import annotations

import json

from grading_calibration.confidence.token_probability import TokenProbabilityConfidence
from grading_calibration.domain.models import GradingInstance
from grading_calibration.inference.token_probs import compute_normalized_yes_no_probabilities


def test_compute_normalized_yes_no_probabilities_matches_legacy_values() -> None:
    alternatives = [
        {"token": "yes", "logprob": -0.4},
        {"token": "no", "logprob": -1.8},
        {"token": "true", "logprob": -2.2},
    ]
    yes, no = compute_normalized_yes_no_probabilities(
        alternatives,
        ["yes", "Yes", "YES", "true", "True", "1"],
        ["no", "No", "NO", "false", "False", "0"],
    )
    assert yes.item() == 0.8253433704376221
    assert no.item() == 0.17465664446353912


def test_token_probability_sample_response_is_stable() -> None:
    instance = GradingInstance(
        id="sample-1",
        question="Question?",
        student_answer="Answer.",
        rubric_item="Rubric.",
        ground_truth_label=1,
    )
    response = {
        "text": '{"verdict": "yes"}',
        "logprobs": [
            [
                {"token": "{", "logprob": -0.01},
                {"token": "yes", "logprob": -3.0},
            ],
            [
                {"token": "yes", "logprob": -0.4},
                {"token": "no", "logprob": -1.8},
                {"token": "true", "logprob": -2.2},
            ],
        ],
    }
    result = TokenProbabilityConfidence(None)._process_single_response(
        instance,
        response,
        predict_by_probability=False,
    )
    expected = {
        "instance_id": "sample-1",
        "prediction": 1,
        "raw_confidence": 0.8253433704376221,
        "calibrated_confidence": None,
        "ground_truth": 1,
        "metadata": {
            "yes_prob": 0.8253433704376221,
            "no_prob": 0.17465664446353912,
            "method": "token_probability",
            "raw_response": '{"verdict": "yes"}',
            "parsed_answer": "yes",
        },
    }
    assert json.dumps(result.to_dict(), sort_keys=True) == json.dumps(
        expected,
        sort_keys=True,
    )
