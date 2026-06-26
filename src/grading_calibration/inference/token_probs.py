"""
Token probability extraction for LLMs.

This module extends the base inference engine to extract token-level probabilities,
which are used for the token probability confidence baseline.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
from .inference_llm import BatchedLLMInference


def compute_normalized_yes_no_probabilities(
    target_alternatives: List[Dict[str, Any]],
    yes_tokens: List[str],
    no_tokens: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized yes/no probabilities from token logprob alternatives."""
    yes_prob = torch.tensor(
        sum(
            torch.exp(torch.tensor(alt["logprob"])).item()
            for alt in target_alternatives
            if alt["token"].lower() in yes_tokens
        ),
        dtype=torch.float32,
    )
    no_prob = torch.tensor(
        sum(
            torch.exp(torch.tensor(alt["logprob"])).item()
            for alt in target_alternatives
            if alt["token"].lower() in no_tokens
        ),
        dtype=torch.float32,
    )

    total_prob = yes_prob + no_prob
    if total_prob.item() == 0:
        yes_prob = torch.tensor(0.5, dtype=torch.float32)
        no_prob = torch.tensor(0.5, dtype=torch.float32)
    else:
        yes_prob /= total_prob
        no_prob /= total_prob

    return yes_prob, no_prob


class TokenProbabilityExtractor:
    """
    Wrapper around BatchedLLMInference to extract token probabilities.

    For binary grading, we extract P(yes) and P(no) from the model's
    output distribution over the vocabulary.
    """

    def __init__(
        self,
        inference_engine: BatchedLLMInference,
        yes_tokens: List[str] = None,
        no_tokens: List[str] = None,
    ):
        """
        Initialize token probability extractor.

        Args:
            inference_engine: Base inference engine
            yes_tokens: List of tokens representing "yes" answer
                        (default: ["yes", "Yes", "YES", "true", "True"])
            no_tokens: List of tokens representing "no" answer
                       (default: ["no", "No", "NO", "false", "False"])
        """
        self.engine = inference_engine

        # Default tokens for yes/no
        self.yes_tokens = yes_tokens or ["yes", "Yes", "YES", "true", "True", "1"]
        self.no_tokens = no_tokens or ["no", "No", "NO", "false", "False", "0"]

    def get_token_probabilities_batch(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Dict[str, float]]:
        """
        Get token probabilities for a batch of conversations.

        Args:
            conversations: List of conversation histories

        Returns:
            List of dictionaries with format:
            {
                "yes_prob": float,
                "no_prob": float,
                "yes_tokens_probs": Dict[str, float],  # Individual yes token probs
                "no_tokens_probs": Dict[str, float],   # Individual no token probs
                "prediction": str,  # "yes" or "no"
                "confidence": float  # max(yes_prob, no_prob)
            }
        """
        if self.engine.backend == "gpt":
            return self._get_probs_gpt(conversations)
        else:
            return self._get_probs_local(conversations)

    def _get_probs_gpt(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Dict[str, float]]:
        """
        Get token probabilities for GPT API.

        GPT API provides logprobs for top tokens.
        """
        results = []

        for conv in conversations:
            try:
                # Call API with logprobs
                response = self.engine.client.chat.completions.create(
                    model=self.engine.model_name,
                    messages=conv,
                    max_tokens=1,  # We only need the first token
                    temperature=0.0,  # Deterministic
                    logprobs=True,
                    top_logprobs=20,  # Get top 20 tokens to capture yes/no
                )

                # Extract logprobs
                logprobs_data = response.choices[0].logprobs.content[0].top_logprobs

                # Build token -> probability mapping
                token_probs = {}
                for item in logprobs_data:
                    token = item.token.strip().lower()
                    prob = torch.exp(torch.tensor(item.logprob)).item()
                    token_probs[token] = prob

                # Aggregate yes/no probabilities
                yes_prob = sum(
                    token_probs.get(tok.lower(), 0.0) for tok in self.yes_tokens
                )
                no_prob = sum(
                    token_probs.get(tok.lower(), 0.0) for tok in self.no_tokens
                )

                # Normalize if neither yes nor no tokens found
                total = yes_prob + no_prob
                if total > 0:
                    yes_prob /= total
                    no_prob /= total
                else:
                    # Fallback: uniform
                    yes_prob = 0.5
                    no_prob = 0.5

                results.append(
                    {
                        "yes_prob": yes_prob,
                        "no_prob": no_prob,
                        "yes_tokens_probs": {
                            tok: token_probs.get(tok.lower(), 0.0)
                            for tok in self.yes_tokens
                        },
                        "no_tokens_probs": {
                            tok: token_probs.get(tok.lower(), 0.0)
                            for tok in self.no_tokens
                        },
                        "prediction": "yes" if yes_prob > no_prob else "no",
                        "confidence": max(yes_prob, no_prob),
                        "all_token_probs": token_probs,
                    }
                )

            except Exception as e:
                print(f"Error extracting token probabilities: {e}")
                # Fallback
                results.append(
                    {
                        "yes_prob": 0.5,
                        "no_prob": 0.5,
                        "yes_tokens_probs": {},
                        "no_tokens_probs": {},
                        "prediction": "no",
                        "confidence": 0.5,
                        "error": str(e),
                    }
                )

        return results

    def _get_probs_local(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Dict[str, float]]:
        """
        Get token probabilities for local models (HuggingFace/vLLM).

        We generate with output_scores=True to get logits.
        """
        if self.engine.backend == "vllm":
            return self._get_probs_vllm(conversations)
        else:
            return self._get_probs_hf(conversations)

    def _get_probs_hf(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Dict[str, float]]:
        """Get token probabilities using HuggingFace models."""
        # Prepare prompts
        input_texts = self.engine._prepare_prompts(conversations)
        model_inputs = self.engine.tokenizer(
            input_texts, return_tensors="pt", padding=True
        ).to(self.engine.device)

        results = []

        with torch.no_grad():
            # Generate one token with return_dict_in_generate=True
            outputs = self.engine.model.generate(
                **model_inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

            # outputs.scores is a tuple of tensors, one per generated token
            # We only generated 1 token, so scores[0] has shape (batch_size, vocab_size)
            logits = outputs.scores[0]  # (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Get token IDs for yes/no tokens
            yes_token_ids = []
            no_token_ids = []

            for tok in self.yes_tokens:
                tok_ids = self.engine.tokenizer.encode(tok, add_special_tokens=False)
                if tok_ids:
                    yes_token_ids.append(tok_ids[0])

            for tok in self.no_tokens:
                tok_ids = self.engine.tokenizer.encode(tok, add_special_tokens=False)
                if tok_ids:
                    no_token_ids.append(tok_ids[0])

            # Extract probabilities for each instance
            for i in range(len(conversations)):
                instance_probs = probs[i]

                # Sum probabilities for yes/no tokens
                yes_prob = sum(instance_probs[tid].item() for tid in yes_token_ids)
                no_prob = sum(instance_probs[tid].item() for tid in no_token_ids)

                # Normalize
                total = yes_prob + no_prob
                if total > 0:
                    yes_prob /= total
                    no_prob /= total
                else:
                    yes_prob = 0.5
                    no_prob = 0.5

                results.append(
                    {
                        "yes_prob": yes_prob,
                        "no_prob": no_prob,
                        "yes_tokens_probs": {
                            self.engine.tokenizer.decode([tid]): instance_probs[
                                tid
                            ].item()
                            for tid in yes_token_ids
                        },
                        "no_tokens_probs": {
                            self.engine.tokenizer.decode([tid]): instance_probs[
                                tid
                            ].item()
                            for tid in no_token_ids
                        },
                        "prediction": "yes" if yes_prob > no_prob else "no",
                        "confidence": max(yes_prob, no_prob),
                    }
                )

        return results

    def _get_probs_vllm(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Dict[str, float]]:
        """Get token probabilities using vLLM."""
        from vllm import SamplingParams

        # Prepare prompts
        input_texts = self.engine._prepare_prompts(conversations)

        # Sample with logprobs
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=100  # Get top 100 logprobs
        )

        outputs = self.engine.model.generate(input_texts, sampling_params)

        results = []

        for output in outputs:
            # Get logprobs for the first (and only) generated token
            token_logprobs = output.outputs[0].logprobs[0]  # Dict[token_id -> Logprob]

            # Convert to probabilities
            token_probs = {}
            for token_id, logprob_obj in token_logprobs.items():
                token_str = logprob_obj.decoded_token.strip().lower()
                prob = torch.exp(torch.tensor(logprob_obj.logprob)).item()
                token_probs[token_str] = prob

            # Aggregate yes/no probabilities
            yes_prob = sum(token_probs.get(tok.lower(), 0.0) for tok in self.yes_tokens)
            no_prob = sum(token_probs.get(tok.lower(), 0.0) for tok in self.no_tokens)

            # Normalize
            total = yes_prob + no_prob
            if total > 0:
                yes_prob /= total
                no_prob /= total
            else:
                yes_prob = 0.5
                no_prob = 0.5

            results.append(
                {
                    "yes_prob": yes_prob,
                    "no_prob": no_prob,
                    "yes_tokens_probs": {
                        tok: token_probs.get(tok.lower(), 0.0)
                        for tok in self.yes_tokens
                    },
                    "no_tokens_probs": {
                        tok: token_probs.get(tok.lower(), 0.0) for tok in self.no_tokens
                    },
                    "prediction": "yes" if yes_prob > no_prob else "no",
                    "confidence": max(yes_prob, no_prob),
                    "all_token_probs": token_probs,
                }
            )

        return results


def extract_token_probabilities(
    conversations: List[List[Dict[str, str]]],
    model_name: str,
    backend: str = "gpt",
    **inference_kwargs,
) -> List[Dict[str, float]]:
    """
    Convenience function to extract token probabilities.

    Args:
        conversations: List of conversation histories
        model_name: Model name
        backend: "gpt", "hf", or "vllm"
        **inference_kwargs: Additional arguments for BatchedLLMInference

    Returns:
        List of probability dictionaries
    """
    # Create inference engine
    engine = BatchedLLMInference(
        model_name=model_name, backend=backend, **inference_kwargs
    )

    # Create extractor
    extractor = TokenProbabilityExtractor(engine)

    # Extract probabilities
    return extractor.get_token_probabilities_batch(conversations)


if __name__ == "__main__":
    # Example usage
    test_conversations = [
        [
            {"role": "system", "content": "You are a grading assistant."},
            {
                "role": "user",
                "content": "Does this answer satisfy the rubric? Answer only 'yes' or 'no'.",
            },
        ]
    ]

    print("Testing token probability extraction...")
    results = extract_token_probabilities(
        conversations=test_conversations, model_name="gpt-4o-mini", backend="gpt"
    )

    for i, result in enumerate(results):
        print(f"\nConversation {i + 1}:")
        print(f"  Yes probability: {result['yes_prob']:.4f}")
        print(f"  No probability: {result['no_prob']:.4f}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
