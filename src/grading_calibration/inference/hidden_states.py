"""
Hidden state extraction for local models.

This module extracts internal representations (hidden states, embeddings)
from transformer models, which can be used for representation stability
analysis in confidence estimation.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
from .inference_llm import BatchedLLMInference


class HiddenStateExtractor:
    """
    Extract hidden states from transformer models.

    NOTE: This only works with local models (HuggingFace/vLLM),
    not with API-based models like GPT.
    """

    def __init__(
        self,
        inference_engine: BatchedLLMInference,
        layer_index: int = -1,
        pooling_strategy: str = "mean",
    ):
        """
        Initialize hidden state extractor.

        Args:
            inference_engine: Base inference engine (must be local, not GPT API)
            layer_index: Which layer to extract (-1 for last layer)
            pooling_strategy: How to pool token representations
                             ("mean", "max", "first", "last")
        """
        self.engine = inference_engine
        self.layer_index = layer_index
        self.pooling_strategy = pooling_strategy

        # Check that engine is local
        if self.engine.backend == "gpt":
            raise ValueError(
                "Hidden state extraction only works with local models "
                "(backend='hf' or 'vllm'), not with GPT API"
            )

    def extract_hidden_states_batch(
        self, conversations: List[List[Dict[str, str]]]
    ) -> np.ndarray:
        """
        Extract hidden states for a batch of conversations.

        Args:
            conversations: List of conversation histories

        Returns:
            numpy array of shape (batch_size, hidden_dim)
        """
        if self.engine.backend == "hf":
            return self._extract_hf(conversations)
        elif self.engine.backend == "vllm":
            # vLLM doesn't easily expose hidden states
            # This is a limitation - we'd need to modify vLLM internals
            raise NotImplementedError(
                "Hidden state extraction not yet implemented for vLLM backend. "
                "Use backend='hf' instead."
            )
        else:
            raise ValueError(f"Unknown backend: {self.engine.backend}")

    def _extract_hf(self, conversations: List[List[Dict[str, str]]]) -> np.ndarray:
        """Extract hidden states using HuggingFace models."""
        # Prepare prompts
        input_texts = self.engine._prepare_prompts(conversations)
        model_inputs = self.engine.tokenizer(
            input_texts, return_tensors="pt", padding=True
        ).to(self.engine.device)

        # Forward pass with output_hidden_states=True
        with torch.no_grad():
            outputs = self.engine.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[self.layer_index]  # (batch, seq, hidden)

        # Pool over sequence dimension
        pooled = self._pool_hidden_states(hidden_states, model_inputs["attention_mask"])

        # Convert to numpy
        return pooled.cpu().numpy()

    def _pool_hidden_states(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states over sequence dimension.

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            Pooled tensor of shape (batch_size, hidden_dim)
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling_strategy == "max":
            # Max pooling
            # Replace padding positions with -inf before max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states_masked = hidden_states.clone()
            hidden_states_masked[mask_expanded == 0] = -float("inf")
            return torch.max(hidden_states_masked, dim=1)[0]

        elif self.pooling_strategy == "first":
            # Use first token ([CLS] token in BERT-like models)
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "last":
            # Use last non-padding token
            batch_size = hidden_states.shape[0]
            last_indices = attention_mask.sum(dim=1) - 1  # Last valid token index
            return hidden_states[torch.arange(batch_size), last_indices, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")


def compute_cosine_similarity(repr1: np.ndarray, repr2: np.ndarray) -> float:
    """
    Compute cosine similarity between two representations.

    Args:
        repr1: First representation (1D or 2D array)
        repr2: Second representation (1D or 2D array)

    Returns:
        Cosine similarity (scalar or array)
    """
    # Ensure 2D
    if repr1.ndim == 1:
        repr1 = repr1.reshape(1, -1)
    if repr2.ndim == 1:
        repr2 = repr2.reshape(1, -1)

    # Compute cosine similarity
    dot_product = np.sum(repr1 * repr2, axis=1)
    norm1 = np.linalg.norm(repr1, axis=1)
    norm2 = np.linalg.norm(repr2, axis=1)

    similarity = dot_product / (norm1 * norm2 + 1e-9)

    return similarity[0] if similarity.shape[0] == 1 else similarity


def compute_euclidean_distance(repr1: np.ndarray, repr2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two representations.

    Args:
        repr1: First representation
        repr2: Second representation

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(repr1 - repr2)


def compute_representation_similarity_matrix(
    representations: List[np.ndarray], metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of representations.

    Args:
        representations: List of representation vectors
        metric: "cosine" or "euclidean"

    Returns:
        Similarity/distance matrix of shape (n, n)
    """
    n = len(representations)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if metric == "cosine":
                sim = compute_cosine_similarity(representations[i], representations[j])
            elif metric == "euclidean":
                sim = -compute_euclidean_distance(
                    representations[i], representations[j]
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")

            matrix[i, j] = sim
            matrix[j, i] = sim

    return matrix


def extract_and_compare_representations(
    inference_engine: BatchedLLMInference,
    conversations1: List[List[Dict[str, str]]],
    conversations2: List[List[Dict[str, str]]],
    layer_index: int = -1,
    pooling_strategy: str = "mean",
) -> Dict[str, Any]:
    """
    Convenience function to extract and compare representations.

    Args:
        inference_engine: HuggingFace inference engine
        conversations1: First set of conversations
        conversations2: Second set of conversations
        layer_index: Which layer to extract
        pooling_strategy: Pooling strategy

    Returns:
        Dictionary with similarity metrics
    """
    extractor = HiddenStateExtractor(
        inference_engine, layer_index=layer_index, pooling_strategy=pooling_strategy
    )

    # Extract representations
    repr1 = extractor.extract_hidden_states_batch(conversations1)
    repr2 = extractor.extract_hidden_states_batch(conversations2)

    # Compute similarities
    similarities = []
    for r1, r2 in zip(repr1, repr2):
        sim = compute_cosine_similarity(r1, r2)
        similarities.append(sim)

    return {
        "mean_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "similarities": similarities,
        "representations1": repr1,
        "representations2": repr2,
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("HIDDEN STATE EXTRACTION TEST")
    print("=" * 80)
    print("\nNote: This example requires a local HuggingFace model.")
    print("To test, initialize BatchedLLMInference with backend='hf'")
    print("and a model like 'meta-llama/Llama-2-7b-hf'")
    print("\nExample:")
    print(
        """
    from grading_calibration.inference.inference_llm import BatchedLLMInference
    from grading_calibration.inference.hidden_states import HiddenStateExtractor

    # Initialize engine with local model
    engine = BatchedLLMInference(
        model_name="meta-llama/Llama-2-7b-hf",
        backend="hf"
    )

    # Create extractor
    extractor = HiddenStateExtractor(
        engine,
        layer_index=-1,
        pooling_strategy="mean"
    )

    # Test conversations
    conversations = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hi there"}]
    ]

    # Extract hidden states
    hidden_states = extractor.extract_hidden_states_batch(conversations)
    print(f"Hidden states shape: {hidden_states.shape}")

    # Compute similarity
    sim = compute_cosine_similarity(hidden_states[0], hidden_states[1])
    print(f"Similarity between 'Hello' and 'Hi there': {sim:.4f}")
    """
    )

    print("\n" + "=" * 80)
