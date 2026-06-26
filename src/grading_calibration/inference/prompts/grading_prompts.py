"""
Prompt templates for automated grading.

This module provides templates for different grading scenarios:
- With rubric item
- Without rubric item
- With answer key
- Without answer key
- Self-reported confidence
"""

from grading_calibration.domain.models import GradingInstance


class GradingPromptTemplate:
    """Base class for grading prompt templates."""

    @staticmethod
    def format_instance(
        instance: GradingInstance,
        include_rubric: bool = True,
        include_answer_key: bool = True,
    ) -> str:
        """
        Format a grading instance into a prompt string.

        Args:
            instance: Grading instance
            include_rubric: Whether to include rubric item
            include_answer_key: Whether to include answer key

        Returns:
            Formatted prompt string
        """
        parts = []

        # Question
        parts.append(f"Question: {instance.question}")

        # Answer key (if available and requested)
        if include_answer_key and instance.has_answer_key():
            parts.append(f"\nCorrect Answer: {instance.answer_key}")

        # Student answer
        parts.append(f"\nStudent Answer: {instance.student_answer}")

        # Rubric item (if available and requested)
        if include_rubric and instance.has_rubric():
            parts.append(f"\nRubric Item: {instance.rubric_item}")

        return "\n".join(parts)


class BinaryGradingPrompt(GradingPromptTemplate):
    """
    Prompts for binary grading (yes/no).

    This is used for evaluating whether a student answer satisfies a rubric item.
    """

    SYSTEM_PROMPT = (
        "You are an expert grading assistant. Your task is to evaluate student "
        "answers against specific rubric criteria. Return valid JSON:\n"
        '{"verdict": "yes/no"}'
    )

    @classmethod
    def create_conversation(
        cls,
        instances: list[GradingInstance],
        include_rubric: bool = True,
        include_answer_key: bool = True,
        response_format: str = "yes_no",
    ) -> list:
        """
        Create a conversation for binary grading.

        Args:
            instance: Grading instance
            include_rubric: Whether to include rubric item
            include_answer_key: Whether to include answer key
            response_format: "yes_no", "true_false", or "0_1"

        Returns:
            List of message dictionaries
        """
        # Format instance details
        instance_texts = [
            cls.format_instance(instance, include_rubric, include_answer_key)
            for instance in instances
        ]

        # Create instruction based on response format
        if response_format == "yes_no":
            instruction = (
                'Does the student answer satisfy the rubric item?\n\n'
                'IMPORTANT: The "verdict" field must be exactly "yes" or "no". '
                'No other values (like "partially") are allowed.\n\n'
                'Return valid JSON:\n'
                '{{"verdict": "yes" or "no"}}\n\n'
                'Example:\n'
                '{{"verdict": "yes"}}'
            )
        else:
            raise ValueError(f"Unknown response format: {response_format}")

        # Combine
        user_messages = [
            f"{instance_text}\n\n{instruction}" for instance_text in instance_texts
        ]

        return [
            [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            for user_message in user_messages
        ]


class SelfReportedConfidencePrompt(GradingPromptTemplate):
    """
    Prompts that ask the LLM to report its own confidence.

    This is used for Baseline 2: Self-reported confidence.
    """

    SYSTEM_PROMPT = (
        "You are an expert grading assistant. Evaluate student answers against "
        "rubric criteria and provide both your grading decision and confidence level. "
        'The "verdict" field must be exactly "yes" or "no" - no other values allowed.\n'
        "Return valid JSON:\n"
        '{"verdict": "yes" or "no", "confidence": <number>}'
    )

    @classmethod
    def create_conversation(
        cls,
        instances: list[GradingInstance],
        include_rubric: bool = True,
        include_answer_key: bool = True,
        confidence_scale: int = 10,
    ) -> list:
        """
        Create a conversation that asks for both answer and confidence.

        Args:
            instance: Grading instance
            include_rubric: Whether to include rubric item
            include_answer_key: Whether to include answer key
            confidence_scale: Maximum confidence value (e.g., 10 for 1-10 scale)

        Returns:
            List of message dictionaries
        """
        instance_texts = [
            cls.format_instance(instance, include_rubric, include_answer_key)
            for instance in instances
        ]

        instruction = (
            f"Evaluate whether the student's answer satisfies the rubric item. "
            f"Your grade must be based ONLY on whether the rubric item is met.\n\n"
            f'IMPORTANT: The "verdict" field must be exactly "yes" or "no". '
            f"No other values (like 'partially') are allowed.\n\n"
            f"Return valid JSON:\n"
            f'{{"verdict": "yes" or "no", "confidence": <1-{confidence_scale}>}}\n\n'
            f"Example:\n"
            f'{{"verdict": "yes", "confidence": 8.5}}\n\n'
            f"Where:\n"
            f'- verdict: "yes" if rubric item is satisfied, "no" otherwise (binary only)\n'
            f"- confidence: your confidence level (1-{confidence_scale})"
        )

        user_messages = [
            f"{instance_text}\n\n{instruction}" for instance_text in instance_texts
        ]

        return [
            [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            for user_message in user_messages
        ]

    @staticmethod
    def parse_response(
        response: str,
        confidence_scale: int = 10,
        log_probs: list = None,
        logger=None,
        instance_id: str = None,
    ) -> tuple:
        """
        Parse LLM response to extract answer and confidence.

        Args:
            response: LLM response text (JSON format)
            confidence_scale: Maximum confidence value
            log_probs: Optional list of [token, logprob, ...] lists for token-level analysis
            logger: Optional LogManager instance for logging parse errors
            instance_id: Optional instance ID for logging context

        Returns:
            If log_probs is None:
                Tuple of (answer, confidence)
            If log_probs is provided:
                Tuple of (answer, confidence, token_indices)
            where answer is "yes"/"no", confidence is float in [0, 1],
            and token_indices is dict with 'verdict', 'json_start', 'json_end' keys
        """
        from utils.helpers import parse_json_response

        # Parse JSON with fallback
        fallback = {"verdict": "no", "confidence": confidence_scale / 2}
        result = parse_json_response(
            response,
            fallback,
            log_probs=log_probs,
            logger=logger,
            instance_id=instance_id,
        )

        # Unpack based on whether log_probs was provided
        if log_probs is not None:
            data, token_indices = result
        else:
            data = result
            token_indices = None

        # Extract verdict and confidence
        answer = data.get("verdict", "no").lower()
        confidence_raw = float(data.get("confidence", confidence_scale / 2))

        # Normalize confidence to [0, 1]
        confidence = (confidence_raw - 1) / (confidence_scale - 1)
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        if log_probs is not None:
            return answer, confidence, token_indices
        else:
            return answer, confidence


class ChainOfThoughtPrompt(GradingPromptTemplate):
    """
    Prompts that encourage chain-of-thought reasoning before answering.

    This can improve grading quality and potentially confidence estimation.
    """

    SYSTEM_PROMPT = (
        "You are an expert grading assistant. Carefully analyze student answers "
        "step by step before making your grading decision. "
        "Return valid JSON:\n"
        '{"reasoning": "<explanation>", "verdict": "yes/no"}'
    )

    @classmethod
    def create_conversation(
        cls,
        instances: list[GradingInstance],
        include_rubric: bool = True,
        include_answer_key: bool = True,
    ) -> list:
        """
        Create a conversation with chain-of-thought instruction.

        Args:
            instance: Grading instance
            include_rubric: Whether to include rubric item
            include_answer_key: Whether to include answer key

        Returns:
            List of message dictionaries
        """
        instance_texts = [
            cls.format_instance(instance, include_rubric, include_answer_key)
            for instance in instances
        ]

        instruction = (
            "Evaluate whether the student's answer satisfies the rubric item. "
            "Your grade must be based ONLY on whether the rubric item is met.\n\n"
            'IMPORTANT: The "verdict" field must be exactly "yes" or "no". '
            "No other values (like 'partially') are allowed.\n\n"
            "Return valid JSON:\n"
            '{"reasoning": "<step-by-step explanation>", "verdict": "yes" or "no"}\n\n'
            "Example:\n"
            '{"reasoning": "The student correctly identifies photosynthesis as energy conversion.", "verdict": "yes"}\n\n'
            "Where:\n"
            "- reasoning: your step-by-step analysis\n"
            '- verdict: "yes" if rubric item is satisfied, "no" otherwise (binary only)\n\n'
            "Note: Write mathematical expressions in plain text (e.g., 'E=hf=ΔE') rather than LaTeX notation."
        )

        user_messages = [
            f"{instance_text}\n\n{instruction}" for instance_text in instance_texts
        ]

        return [
            [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            for user_message in user_messages
        ]

    @staticmethod
    def parse_response(
        response: str,
        log_probs: list = None,
        logger=None,
        instance_id: str = None,
    ):
        """
        Extract final answer from chain-of-thought response.

        Args:
            response: LLM response text (JSON format)
            log_probs: Optional list of [token, logprob, ...] lists for token-level analysis
            logger: Optional LogManager instance for logging parse errors
            instance_id: Optional instance ID for logging context

        Returns:
            If log_probs is None:
                "yes" or "no"
            If log_probs is provided:
                Tuple of (verdict, token_indices)
        """
        from utils.helpers import parse_json_response

        # Parse JSON with fallback
        fallback = {"verdict": "unknown"}
        result = parse_json_response(
            response,
            fallback,
            log_probs=log_probs,
            logger=logger,
            instance_id=instance_id,
        )

        # Unpack based on whether log_probs was provided
        if log_probs is not None:
            data, token_indices = result
        else:
            data = result
            token_indices = None

        # Extract verdict
        verdict = data.get("verdict", "unknown").lower()
        verdict = verdict if verdict in ["yes", "no"] else "unknown"

        if log_probs is not None:
            return verdict, token_indices
        else:
            return verdict


# Convenience functions


def create_grading_conversation(
    instances: list[GradingInstance],
    prompt_type: str = "binary",
    include_rubric: bool = True,
    include_answer_key: bool = True,
    **kwargs,
) -> list:
    """
    Convenience function to create grading conversation.

    Args:
        instance: Grading instance
        prompt_type: "binary", "self_reported", or "chain_of_thought"
        include_rubric: Whether to include rubric item
        include_answer_key: Whether to include answer key
        **kwargs: Additional arguments for specific prompt types

    Returns:
        Conversation (list of message dictionaries)
    """
    if prompt_type == "binary":
        return BinaryGradingPrompt.create_conversation(
            instances, include_rubric, include_answer_key, **kwargs
        )
    elif prompt_type == "self_reported":
        return SelfReportedConfidencePrompt.create_conversation(
            instances, include_rubric, include_answer_key, **kwargs
        )
    elif prompt_type == "chain_of_thought":
        return ChainOfThoughtPrompt.create_conversation(
            instances, include_rubric, include_answer_key, **kwargs
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
