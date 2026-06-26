"""
Checkpoint Manager for Long-Running Inference Tasks

Handles incremental checkpoint saving and consolidation for resume capability.
Also includes LogManager for logging parse errors and full responses.
Also includes RawResponseSaver for saving raw LLM responses before parsing.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from typing import Callable, List, Dict, Any, Optional


class RawResponseSaver:
    """
    Saves raw LLM responses immediately after generation, BEFORE parsing.

    This ensures that even if parsing crashes, the raw responses are preserved
    and can be reparsed later without re-running expensive LLM inference.

    Format: JSONL file with one JSON object per line:
    {"instance_id": "...", "response_idx": 0, "response": "...", "timestamp": "..."}
    """

    def __init__(
        self,
        base_dir: str | Path,
        model_name: str,
        method: str,
        timestamp: str | None = None,
    ):
        """
        Initialize raw response saver.

        Args:
            base_dir: Base directory for raw responses
            model_name: Short model name (e.g., "qwen80b")
            method: Confidence method name (e.g., "self_consistency")
            timestamp: Optional timestamp (for resume), otherwise creates new
        """
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.method = method
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create raw_responses directory
        self.raw_dir = self.base_dir / "raw_responses" / method
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # JSONL file path
        self.jsonl_file = self.raw_dir / f"raw_{model_name}_{method}_{self.timestamp}.jsonl"
        self._write_count = 0

    def save_responses(
        self,
        instance_ids: List[str],
        responses: List[str],
        chunk_idx: Optional[int] = None,
    ) -> int:
        """
        Save raw responses immediately after LLM generation.

        Args:
            instance_ids: List of instance IDs (one per original instance)
            responses: List of raw LLM response strings
            chunk_idx: Optional chunk index for logging

        Returns:
            Number of responses saved

        Note:
            For self-consistency with n_samples, responses are interleaved:
            [inst1_sample1, inst2_sample1, ..., inst1_sample2, inst2_sample2, ...]
            Pass the original instance_ids (not duplicated), and this function
            will handle the mapping.
        """
        n_instances = len(instance_ids)
        n_responses = len(responses)
        n_samples = n_responses // n_instances if n_instances > 0 else 1

        timestamp = datetime.now().isoformat()
        saved = 0

        with open(self.jsonl_file, "a") as f:
            for resp_idx, response in enumerate(responses):
                # Map response index to instance index
                # For self-consistency: responses[i::n_instances] belong to instance i
                instance_idx = resp_idx % n_instances
                sample_idx = resp_idx // n_instances

                record = {
                    "instance_id": instance_ids[instance_idx],
                    "sample_idx": sample_idx,
                    "response_idx": resp_idx,
                    "response": response,
                    "timestamp": timestamp,
                    "chunk_idx": chunk_idx,
                }
                f.write(json.dumps(record) + "\n")
                saved += 1

        self._write_count += saved
        return saved

    def save_single_response(
        self,
        instance_id: str,
        response: str,
        sample_idx: int = 0,
    ) -> None:
        """Save a single raw response."""
        timestamp = datetime.now().isoformat()

        with open(self.jsonl_file, "a") as f:
            record = {
                "instance_id": instance_id,
                "sample_idx": sample_idx,
                "response_idx": self._write_count,
                "response": response,
                "timestamp": timestamp,
            }
            f.write(json.dumps(record) + "\n")

        self._write_count += 1

    def get_file_path(self) -> Path:
        """Get path to the JSONL file."""
        return self.jsonl_file

    def get_write_count(self) -> int:
        """Get total number of responses written."""
        return self._write_count

    @staticmethod
    def load_responses(jsonl_path: str | Path) -> List[Dict[str, Any]]:
        """
        Load all raw responses from a JSONL file.

        Args:
            jsonl_path: Path to the JSONL file

        Returns:
            List of response records
        """
        records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def group_by_instance(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Group responses by instance_id.

        Args:
            records: List of response records from load_responses()

        Returns:
            Dict mapping instance_id -> list of responses (ordered by sample_idx)
        """
        from collections import defaultdict

        grouped = defaultdict(list)
        for record in records:
            grouped[record["instance_id"]].append(record)

        # Sort by sample_idx and extract just the response text
        result = {}
        for instance_id, recs in grouped.items():
            recs.sort(key=lambda r: r.get("sample_idx", 0))
            result[instance_id] = [r["response"] for r in recs]

        return result


class LogManager:
    """Manages logging of parse errors and full LLM responses."""

    def __init__(
        self,
        base_dir: str | Path,
        model_name: str,
        method: str,
        timestamp: str | None = None,
    ):
        """
        Initialize log manager.

        Args:
            base_dir: Base directory for logs (will create logs/{method}/ structure)
            model_name: Short model name (e.g., "llama_3_1_8b_instruct")
            method: Confidence method name (e.g., "self_consistency")
            timestamp: Optional timestamp to use (for resume), otherwise creates new
        """
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.method = method
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create logs directory structure: logs/{method}/
        self.log_dir = self.base_dir / "logs" / method
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path
        self.log_file = self.log_dir / f"{model_name}_{method}_{self.timestamp}.log"
        self._initialized = False

    def _ensure_initialized(self):
        """Write header to log file on first use."""
        if not self._initialized:
            with open(self.log_file, "a") as f:
                f.write(f"{'='*80}\n")
                f.write(f"Log file for {self.model_name} - {self.method}\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"{'='*80}\n\n")
            self._initialized = True

    def log_parse_error(
        self,
        instance_id: str,
        error_message: str,
        full_response: str,
        additional_info: dict | None = None,
    ):
        """
        Log a parse error with full response.

        Args:
            instance_id: ID of the instance that failed
            error_message: The error message from parsing
            full_response: The full LLM response that failed to parse
            additional_info: Optional additional context
        """
        self._ensure_initialized()

        with open(self.log_file, "a") as f:
            f.write(f"\n{'-'*80}\n")
            f.write(f"[{datetime.now().isoformat()}] Parse Error\n")
            f.write(f"Instance ID: {instance_id}\n")
            f.write(f"Error: {error_message}\n")

            if additional_info:
                f.write("Additional Info:\n")
                for key, value in additional_info.items():
                    f.write(f"  {key}: {value}\n")

            f.write(f"\nFull Response:\n")
            f.write(f"{'-'*40}\n")
            f.write(full_response)
            f.write(f"\n{'-'*40}\n")
            f.write(f"{'-'*80}\n")

    def log_warning(self, message: str, instance_id: str | None = None):
        """Log a general warning message."""
        self._ensure_initialized()

        with open(self.log_file, "a") as f:
            timestamp = datetime.now().isoformat()
            if instance_id:
                f.write(f"[{timestamp}] WARNING ({instance_id}): {message}\n")
            else:
                f.write(f"[{timestamp}] WARNING: {message}\n")

    def get_log_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_file


class CheckpointManager:
    """Manages checkpoint saving and resumption for inference tasks."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str,
        method: str,
        dataset: str | None = None,
        timestamp: str | None = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoint files
            model_name: Short model name (e.g., "llama_3_1_8b_instruct")
            method: Confidence method name (e.g., "self_consistency")
            dataset: Optional dataset name (e.g., "ricechem", "beetle")
            timestamp: Optional timestamp to use (for resume), otherwise creates new
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.method = method
        self.dataset = dataset.lower() if dataset else None
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.batch_counter = 0
        self.seen_instance_ids = set()  # Track all instance IDs across checkpoints

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_prefix(self) -> str:
        parts = [f"checkpoint_{self.model_name}", self.method]
        if self.dataset:
            parts.append(self.dataset)
        return "_".join(parts)

    def _checkpoint_pattern(self) -> str:
        return f"{self._checkpoint_prefix()}_*_batch*.csv"

    def find_latest_checkpoint(self) -> tuple[str | None, set]:
        """
        Find the latest checkpoint file and extract completed instance IDs.

        Returns:
            Tuple of (timestamp, set of completed instance IDs)
            If no checkpoint found, returns (None, empty set)
        """
        if not self.checkpoint_dir.exists():
            return None, set()

        # Find all checkpoint files for this model and method
        checkpoint_files = sorted(self.checkpoint_dir.glob(self._checkpoint_pattern()))

        if not checkpoint_files:
            return None, set()

        # Get latest checkpoint
        latest_checkpoint = checkpoint_files[-1]
        print(f"\nFound checkpoint: {latest_checkpoint}")

        # Extract timestamp from filename: checkpoint_{model}_{method}_{TIMESTAMP}_batch{N}.csv
        filename_parts = latest_checkpoint.stem.split("_")
        timestamp = None
        for part in filename_parts:
            if "-" in part and len(part) == 15:  # YYYYMMDD-HHMMSS format
                timestamp = part
                break

        if not timestamp:
            print("Warning: Could not extract timestamp from checkpoint")
            return None, set()

        # Load ALL checkpoint files with this timestamp to get complete set of completed IDs
        timestamp_pattern = f"{self._checkpoint_prefix()}_{timestamp}_batch*.csv"
        timestamp_checkpoints = list(self.checkpoint_dir.glob(timestamp_pattern))

        completed_ids = set()
        for checkpoint_file in timestamp_checkpoints:
            try:
                df = pd.read_csv(checkpoint_file)
                completed_ids.update(df["instance_id"])
            except Exception as e:
                print(f"Warning: Could not load {checkpoint_file}: {e}")

        print(
            f"Loaded {len(completed_ids)} completed instances from {len(timestamp_checkpoints)} checkpoint(s)"
        )
        self.batch_counter = len(timestamp_checkpoints)
        return timestamp, completed_ids

    def create_checkpoint_callback(
        self, all_instances: list, split_labels: list, args
    ) -> Callable:
        """
        Create a checkpoint callback function for use with estimator.

        Args:
            all_instances: List of all instances being processed (filtered, not complete dataset)
            split_labels: Corresponding split labels for instances
            args: Parsed command line arguments

        Returns:
            Callback function with signature: (results, chunk_idx, total_chunks) -> None
        """

        def save_checkpoint(results: list, chunk_idx: int, total_chunks: int):
            """Save checkpoint after processing each chunk."""
            print(f"\nSaving checkpoint (chunk {chunk_idx + 1}/{total_chunks})...")

            # Calculate the correct offset for this chunk
            # Use the configured chunk size, not len(results), because the last chunk may be smaller
            chunk_size = args.checkpoint_chunk_size
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_instances))

            # Prepare results in same format as final output
            results_data = []
            for instance, pred, split in zip(
                all_instances[start_idx:end_idx], results, split_labels[start_idx:end_idx]
            ):
                result = {
                    "instance_id": instance.id,
                    "question": instance.question,
                    "student_answer": instance.student_answer,
                    "rubric_item": instance.rubric_item,
                    "answer_key": instance.answer_key if instance.answer_key else "",
                    "ground_truth_label": instance.ground_truth_label,
                    "prediction": pred.prediction,
                    "raw_confidence": pred.raw_confidence,
                    "is_correct": pred.is_correct(),
                    "split": split,
                    "question_id": instance.metadata.get("question_id", ""),
                    "student_id": instance.metadata.get("student_id", ""),
                    "rubric_id": instance.metadata.get("rubric_id", ""),
                }

                if args.method == "self_consistency":
                    # Add vote counts for self-consistency
                    result["vote_counts"] = pred.metadata.get("vote_counts", {})
                    for i in range(args.n_samples):
                        result[f"llm_response_{i}"] = pred.metadata.get(
                            f"raw_response_{i}", ""
                        )
                else:
                    result["llm_response"] = pred.metadata.get("raw_response", "")

                if args.method == "token_probability":
                    result["yes_prob"] = pred.metadata.get("yes_prob", None)
                    result["no_prob"] = pred.metadata.get("no_prob", None)

                results_data.append(result)

            # Check for duplicate instance IDs
            current_ids = [r["instance_id"] for r in results_data]

            # Check for duplicates within this chunk
            unique_current = set(current_ids)
            if len(current_ids) != len(unique_current):
                duplicates_in_chunk = [id for id in unique_current if current_ids.count(id) > 1]
                raise ValueError(
                    f"Duplicate instance IDs within chunk {chunk_idx + 1}: {duplicates_in_chunk}\n"
                    f"Total in chunk: {len(current_ids)}, Unique: {len(unique_current)}"
                )

            # Check for duplicates against previously saved checkpoints
            duplicates_across_checkpoints = unique_current & self.seen_instance_ids
            if duplicates_across_checkpoints:
                raise ValueError(
                    f"Duplicate instance IDs found across checkpoints in chunk {chunk_idx + 1}:\n"
                    f"  {sorted(list(duplicates_across_checkpoints))[:10]}{'...' if len(duplicates_across_checkpoints) > 10 else ''}\n"
                    f"  Total duplicates: {len(duplicates_across_checkpoints)}\n"
                    f"  These IDs were already saved in previous checkpoint(s)"
                )

            # Add current IDs to seen set
            self.seen_instance_ids.update(unique_current)

            # Save checkpoint
            df = pd.DataFrame(results_data)
            checkpoint_file = f"{self._checkpoint_prefix()}_{self.timestamp}_batch{self.batch_counter}.csv"
            checkpoint_path = self.checkpoint_dir / checkpoint_file
            df.to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved to: {checkpoint_path}")
            print(f"  ✓ All {len(current_ids)} instance IDs are unique (total unique across all checkpoints: {len(self.seen_instance_ids)})")
            self.batch_counter += 1

        return save_checkpoint

    def consolidate_checkpoints(self, output_path: str | Path) -> None:
        """
        Consolidate all checkpoint files with this timestamp into final output file.

        Args:
            output_path: Path to save the consolidated CSV file
        """
        print(f"\nConsolidating checkpoints into final file...")

        # Find all checkpoint files with this timestamp
        pattern = f"{self._checkpoint_prefix()}_{self.timestamp}_batch*.csv"
        checkpoint_files = sorted(self.checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            print(f"Warning: No checkpoint files found with timestamp {self.timestamp}")
            return

        print(f"Found {len(checkpoint_files)} checkpoint files to consolidate")

        # Load and combine all checkpoint DataFrames
        dfs = []
        for checkpoint_file in checkpoint_files:
            try:
                df = pd.read_csv(checkpoint_file)
                dfs.append(df)
                print(f"  Loaded {checkpoint_file.name}: {len(df)} instances")
            except Exception as e:
                print(f"  Warning: Could not load {checkpoint_file}: {e}")

        if not dfs:
            print("Error: No checkpoint files could be loaded")
            return

        # Concatenate all DataFrames
        final_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal instances in consolidated file: {len(final_df)}")

        # Save final file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Consolidated results saved to: {output_path}")

    def cleanup_checkpoints(self) -> None:
        """Delete checkpoint files after successful consolidation (optional)."""
        pattern = (
            f"{self._checkpoint_prefix()}_{self.timestamp}_batch*.csv"
        )
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))

        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_file.unlink()
                print(f"Deleted checkpoint: {checkpoint_file.name}")
            except Exception as e:
                print(f"Warning: Could not delete {checkpoint_file}: {e}")

        print(f"Cleaned up {len(checkpoint_files)} checkpoint files")
