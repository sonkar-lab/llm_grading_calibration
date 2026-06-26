import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
print(f"Adding project root to path: {project_root}")
sys.path.insert(0, str(project_root))

from grading_calibration.inference.inference_llm import BatchedLLMInference
from utils.output_paths import add_output_args


def initialize_inference_engine(args, backend):
    """
    Initialize the inference engine with all parameters.

    Args:
        args: Parsed command line arguments
        backend: Backend to use ("gpt", "vllm", or "hf")

    Returns:
        BatchedLLMInference instance
    """
    # Build kwargs, only including non-None parameters
    inference_kwargs = {
        "model_name": args.model,
        "backend": backend,
        "batch_size": args.batch_size,
        "cache_dir": args.cache_dir,
        "num_gpus": args.num_gpus,
    }

    # Add optional sampling parameters only if specified
    if args.temperature is not None:
        inference_kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        inference_kwargs["max_tokens"] = args.max_tokens
    if args.top_p is not None:
        inference_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        inference_kwargs["top_k"] = args.top_k
    if args.min_p is not None:
        inference_kwargs["min_p"] = args.min_p
    if args.enable_thinking:
        inference_kwargs["enable_thinking"] = True
    if args.gpt_reasoning is not None:
        inference_kwargs["gpt_reasoning"] = args.gpt_reasoning

    # Add vLLM-specific parameters only if specified
    if args.vllm_max_num_seqs is not None:
        inference_kwargs["vllm_max_num_seqs"] = args.vllm_max_num_seqs
    if args.vllm_max_model_len is not None:
        inference_kwargs["vllm_max_model_len"] = args.vllm_max_model_len
    if args.vllm_gpu_memory_utilization is not None:
        inference_kwargs["vllm_gpu_memory_utilization"] = (
            args.vllm_gpu_memory_utilization
        )

    return BatchedLLMInference(**inference_kwargs)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect tutoring responses from LLM")

    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., 'gpt-5-nano-2025-08-07', 'meta-llama/Llama-2-7b', 'mistralai/Mixtral-8x7B-Instruct-v0.1')",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="self_reported",
        choices=["self_reported", "self_consistency", "token_probability"],
        help="Confidence estimation method to use",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ricechem",
        choices=["ricechem", "scientsbank", "beetle"],
        help="Dataset to use (default: ricechem)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing input data CSV files (required for ricechem)",
    )

    # Backend configuration
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["auto", "gpt", "vllm", "hf"],
        help="Inference backend: auto (detect from model name), gpt (OpenAI API), vllm (local GPU), or hf (HuggingFace)",
    )

    # Generation parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )

    # Sampling parameters (optional - use model-specific defaults if not specified)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: None, uses model-specific default)",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: None, uses model default)",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p sampling parameter (default: None, uses model default)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter (default: None, uses model default)",
    )

    parser.add_argument(
        "--min_p",
        type=float,
        default=None,
        help="Min-p sampling parameter (default: None, uses model default)",
    )

    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for compatible models (e.g., Qwen3, GPT-OSS, CWM)",
    )

    parser.add_argument(
        "--gpt_reasoning",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="GPT-OSS reasoning level (default: medium)",
    )

    # Hardware configuration (for local models)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data1/ss164/cache/",
        help="Cache directory for model downloads (local models only, default: /data1/ss164/cache/)",
    )

    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs for vLLM (default: 1)"
    )

    # vLLM-specific configuration
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to process in a batch (vLLM only, default: 256)",
    )

    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Maximum sequence length in tokens (vLLM only, default: model context length)",
    )

    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory to use for KV cache (vLLM only, default: 0.9, range: 0.0-1.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Number of samples to generate"
    )

    parser.add_argument(
        "--confidence_scale",
        type=int,
        default=10,
        help="Confidence scaling factor (default: 10)",
    )

    parser.add_argument(
        "--train_size",
        type=float,
        default=0.6,
        help="Number of training instances to use (default: 0.6 of dataset)",
    )
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.2,
        help="Number of validation instances to use (default: 0.2 of dataset)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Number of test instances to use (default: 0.2 of dataset)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: auto-generated)",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum instances to process (for testing)",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=None,
        help="Limit to first N instances for a smoke test run.",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_chunk_size",
        type=int,
        default=None,
        help="Save checkpoint after processing N instances (default: None, process all at once)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint (checkpoints stored in {method}/checkpoints/)",
    )

    parser.add_argument(
        "--top_k_probability",
        type=int,
        default=20,
        help="Top-k tokens to consider for probability-based confidence (default: 20)",
    )

    # Organized output options (shared across steps)
    add_output_args(parser)

    return parser.parse_args()
