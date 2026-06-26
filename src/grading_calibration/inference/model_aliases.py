"""Canonical model-name aliases used by experiment scripts."""

from __future__ import annotations

MODEL_ALIASES: dict[str, str] = {
    "qwen4b": "qwen3_4b_thinking_2507",
    "llama8b": "llama_3.1_8b_instruct",
    "gptoss20b": "gpt_oss_20b",
    "gptoss120b": "gpt_oss_120b",
    "llama70b": "llama_3.3_70b_instruct",
    "qwen80b": "qwen3_next_80b_a3b_instruct",
}


def resolve_model_alias(model: str) -> str:
    """Resolve a short model nickname to the canonical experiment model name."""
    return MODEL_ALIASES.get(model.lower(), model.lower())
