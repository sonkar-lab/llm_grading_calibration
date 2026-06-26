from __future__ import annotations

from grading_calibration.inference.model_aliases import MODEL_ALIASES, resolve_model_alias


def test_model_aliases_include_subset_and_big_model_union() -> None:
    assert MODEL_ALIASES == {
        "qwen4b": "qwen3_4b_thinking_2507",
        "llama8b": "llama_3.1_8b_instruct",
        "gptoss20b": "gpt_oss_20b",
        "gptoss120b": "gpt_oss_120b",
        "llama70b": "llama_3.3_70b_instruct",
        "qwen80b": "qwen3_next_80b_a3b_instruct",
    }


def test_resolve_model_alias_preserves_existing_fallback_behavior() -> None:
    assert resolve_model_alias("QWEN4B") == "qwen3_4b_thinking_2507"
    assert resolve_model_alias("unknown_model") == "unknown_model"
