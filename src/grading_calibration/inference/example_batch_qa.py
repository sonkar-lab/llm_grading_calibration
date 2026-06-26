#!/usr/bin/env python3
"""
Example: Batch Question Answering with CSV Input/Output

This script demonstrates the simplest use case for BatchedLLMInference:
1. Read questions from a CSV file (columns: question_id, question)
2. Generate answers using the inference engine
3. Write results back to a new CSV file (columns: question_id, question, answer)

Usage:
    # With GPT API
    python llm_inference/example_batch_qa.py \
        --input questions.csv \
        --output answers.csv \
        --model gpt-4o-mini

    # With local model (vLLM)
    python llm_inference/example_batch_qa.py \
        --input questions.csv \
        --output answers.csv \
        --model meta-llama/Llama-2-7b \
        --backend vllm \
        --num-gpus 2
"""

import argparse
import pandas as pd
from typing import List, Dict
from .inference_llm import BatchedLLMInference


def create_qa_conversation(
    question: str, system_prompt: str = None
) -> List[Dict[str, str]]:
    """
    Create a simple Q&A conversation.

    Args:
        question: The question to answer
        system_prompt: Optional system message (defaults to helpful assistant)

    Returns:
        Conversation list with system + user messages
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Answer questions clearly and concisely."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def load_questions_csv(csv_path: str) -> pd.DataFrame:
    """
    Load questions from CSV file.

    Expected columns:
        - question_id: Unique identifier for each question
        - question: The question text

    Returns:
        DataFrame with questions
    """
    print(f"Loading questions from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ["question_id", "question"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    print(f"Loaded {len(df)} questions")
    return df


def batch_answer_questions(
    questions_df: pd.DataFrame,
    inference_engine: BatchedLLMInference,
    system_prompt: str = None,
    batch_size: int = None,
) -> pd.DataFrame:
    """
    Generate answers for all questions using batched inference.

    Args:
        questions_df: DataFrame with question_id and question columns
        inference_engine: Initialized BatchedLLMInference instance
        system_prompt: Optional system prompt for Q&A
        batch_size: Optional batch size (uses engine's default if None)

    Returns:
        DataFrame with question_id, question, and answer columns
    """
    print("\nGenerating answers...")

    # Create conversations for all questions
    conversations = [
        create_qa_conversation(row["question"], system_prompt)
        for _, row in questions_df.iterrows()
    ]

    # Process in batches if needed
    if batch_size is None:
        batch_size = inference_engine.batch_size

    all_answers = []
    total_questions = len(conversations)

    for i in range(0, total_questions, batch_size):
        batch_convs = conversations[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_questions + batch_size - 1) // batch_size

        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch_convs)} questions)..."
        )

        # Generate answers for this batch
        answers = inference_engine.generate(batch_convs)
        all_answers.extend(answers)

    # Create results dataframe
    results_df = questions_df.copy()
    results_df["answer"] = all_answers

    print(f"\nGenerated {len(all_answers)} answers")
    return results_df


def save_answers_csv(results_df: pd.DataFrame, output_path: str):
    """
    Save question-answer pairs to CSV file.

    Args:
        results_df: DataFrame with question_id, question, answer columns
        output_path: Path to save CSV file
    """
    print(f"\nSaving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} question-answer pairs")


def main():
    """Main entry point for batch Q&A example."""
    parser = argparse.ArgumentParser(
        description="Batch Question Answering - CSV Input/Output Example"
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with question_id and question columns",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file for results (question_id, question, answer)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help='Model name (e.g., "gpt-4o-mini", "meta-llama/Llama-2-7b")',
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gpt",
        choices=["gpt", "hf", "vllm"],
        help="Inference backend: gpt (OpenAI API), hf (HuggingFace), or vllm",
    )

    # Generation parameters
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing"
    )

    # System prompt
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (optional)",
    )

    # Sampling parameters (optional - use model-specific defaults if not specified)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.7, or model-specific default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (model default if not specified)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter (model default if not specified)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter (model default if not specified)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter (model default if not specified)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for compatible models (e.g., Qwen3)",
    )
    parser.add_argument(
        "--gpt-reasoning",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="GPT-OSS reasoning level (default: medium)",
    )

    # Hardware configuration (for local models)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data1/ss164/cache/",
        help="Cache directory for model downloads (local models only)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs for vLLM"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BATCH QUESTION ANSWERING - CSV EXAMPLE")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model} ({args.backend})")
    print(f"Batch:  {args.batch_size} questions at a time")
    print("=" * 80)

    # Load questions from CSV
    questions_df = load_questions_csv(args.input)

    # Initialize inference engine ONCE
    print("\nInitializing inference engine...")

    # Build kwargs, only including non-None parameters
    llm_kwargs = {
        "model_name": args.model,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "cache_dir": args.cache_dir,
        "num_gpus": args.num_gpus,
    }

    # Add optional sampling parameters only if specified
    if args.temperature is not None:
        llm_kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        llm_kwargs["max_tokens"] = args.max_tokens
    if args.top_p is not None:
        llm_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        llm_kwargs["top_k"] = args.top_k
    if args.min_p is not None:
        llm_kwargs["min_p"] = args.min_p
    if args.enable_thinking:
        llm_kwargs["enable_thinking"] = True
    if args.gpt_reasoning is not None:
        llm_kwargs["gpt_reasoning"] = args.gpt_reasoning

    inference_engine = BatchedLLMInference(**llm_kwargs)

    # Generate answers for all questions
    results_df = batch_answer_questions(
        questions_df,
        inference_engine,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
    )

    # Save results to CSV
    save_answers_csv(results_df, args.output)

    # Print sample results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS (first 3)")
    print("=" * 80)
    for i, row in results_df.head(3).iterrows():
        print(f"\n[Question {row['question_id']}]")
        print(f"Q: {row['question']}")
        print(f"A: {row['answer'][:200]}{'...' if len(row['answer']) > 200 else ''}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Processed {len(results_df)} questions")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
