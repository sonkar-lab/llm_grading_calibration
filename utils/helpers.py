"""
General helper utilities.
"""

import json
import pickle
from pathlib import Path
from typing import Any
import random
import numpy as np
import os


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch, also set torch seeds
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data as JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        filepath: Output file path
        indent: JSON indentation level
    """
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data as pickle file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def batch_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    """
    Split list into batches.

    Args:
        items: List to batch
        batch_size: Size of each batch

    Returns:
        List of batches

    Example:
        >>> batch_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys

    Returns:
        Flattened dictionary

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        {"a_b": 1, "a_c": 2, "d": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def print_header(text: str, width: int = 80, char: str = "=") -> None:
    """
    Print a formatted header.

    Args:
        text: Header text
        width: Total width
        char: Character for border
    """
    print(char * width)
    print(text.center(width))
    print(char * width)


def print_section(text: str, width: int = 80, char: str = "-") -> None:
    """
    Print a formatted section header.

    Args:
        text: Section text
        width: Total width
        char: Character for border
    """
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}")


def find_csv_folder_from_root(root_dir: str) -> Path:
    # Walk up until we find grading_calibration
    current = Path.cwd()

    while not os.path.basename(current) == root_dir:
        parent = os.path.dirname(current)
        if parent == current:  # At filesystem root, not found
            return Path()
        current = parent

    # Now search down from grading_calibration
    return Path(f"{current}/experiments/predictions_csvs/")


def _find_token_indices(
    response: str, json_str: str, log_probs: list[list], parsed_dict: dict[str, Any]
) -> dict[str, int]:
    """
    Find token indices in log_probs that correspond to the JSON response.

    Args:
        response: Full response text
        json_str: Extracted JSON string
        log_probs: List of [token, logprob, ...] lists
        parsed_dict: Parsed JSON dictionary

    Returns:
        Dictionary with token indices
    """
    indices = {}

    # Reconstruct the full text from tokens
    tokens = [token_info[0]["token"] for token_info in log_probs]
    reconstructed = "".join(tokens)

    # Find where json_str appears in the reconstructed text
    json_start_char = reconstructed.find(json_str)
    if json_start_char == -1:
        # Try finding the JSON start marker
        json_start_char = reconstructed.find("{")

    if json_start_char != -1:
        # Find which token index corresponds to this character position
        char_count = 0
        for i, token in enumerate(tokens):
            if char_count >= json_start_char:
                indices["json_start"] = i
                break
            char_count += len(token)

        # Find JSON end
        json_end_char = json_start_char + len(json_str)
        char_count = 0
        for i, token in enumerate(tokens):
            char_count += len(token)
            if char_count >= json_end_char:
                indices["json_end"] = i
                break

        # Find "verdict" value token
        if "verdict" in parsed_dict:
            verdict_value = str(parsed_dict["verdict"])
            # Look for the verdict value in the JSON string
            verdict_patterns = [
                f'"verdict":{verdict_value}',
                f'"verdict": {verdict_value}',
                f'"verdict":"{verdict_value}"',
                f'"verdict": "{verdict_value}"',
            ]
            for pattern in verdict_patterns:
                pos = json_str.find(pattern)
                if pos != -1:
                    # Find where the value starts (after the colon)
                    value_start = json_str.find(verdict_value, pos)
                    if value_start != -1:
                        # Convert to absolute character position
                        abs_pos = json_start_char + value_start
                        # Find corresponding token
                        char_count = 0
                        for i, token in enumerate(tokens):
                            if char_count >= abs_pos:
                                indices["verdict"] = i
                                break
                            char_count += len(token)
                        break

    return indices


def _patch_unescaped_quotes(json_str: str, max_patches: int = 10) -> tuple[dict | None, str]:
    """
    Try to fix unescaped inner quotes using the JSON parser's error position.

    Args:
        json_str: JSON string with potential unescaped quotes
        max_patches: Maximum number of quote patches to attempt

    Returns:
        Tuple of (parsed_obj, patched_str) or (None, last_str) if still invalid
    """
    s = json_str
    if not s:
        return None, s

    try:
        for _ in range(max_patches):
            try:
                obj = json.loads(s)
                return obj, s
            except json.JSONDecodeError as e:
                pos = e.pos  # index where parsing failed

                # Find the closest unescaped double quote at or before pos
                # Clamp i to valid index range (pos can be len(s) at end-of-string errors)
                i = min(pos, len(s) - 1) if s else -1
                while i >= 0 and s[i] != '"':
                    i -= 1
                if i <= 0:
                    # No suitable quote found; give up
                    return None, s

                # If this quote is already escaped, also give up (we'd need a different fix)
                if i > 0 and s[i - 1] == '\\':
                    return None, s

                # Insert a backslash before the problematic quote
                s = s[:i] + '\\' + s[i:]

        # Still failed after max_patches
        return None, s
    except Exception as e:
        # Safety net: never crash the entire run due to parsing issues
        print(f"Warning: Unexpected error in _patch_unescaped_quotes: {e}")
        return None, json_str


def _strip_wrappers(response: str) -> tuple[str, str, str]:
    """
    Strip wrapper markers from LLM response (channel markers, think tags, etc.).
    
    Handles:
    - Channel markers: <|channel|>final<|message|>, <|assistant|>final, etc.
    - Clutter: <|end|>, <|start|>
    - Think tags: </think>
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Tuple of (cleaned_response, raw_response, extraction_method_hint)
        - cleaned_response: Response with wrappers stripped
        - raw_response: Original response preserved for token indexing
        - extraction_method_hint: "channel_marker" if markers found, else "unknown"
    """
    raw_response = response
    extraction_method_hint = "unknown"
    
    # Strip channel markers (handle multiple model formats)
    markers = ["<|channel|>final<|message|>", "<|assistant|>final<|message|>", "<|assistant|>final"]
    for marker in markers:
        if marker in response:
            extraction_method_hint = "channel_marker"
            start = response.rfind(marker) + len(marker)
            end = response.find("<|return|>", start)
            response = response[start:end if end != -1 else None].strip()
            break

    # Strip any leftover <|end|>/<|start|> clutter (loop handles multiples)
    while True:
        stripped = response.lstrip()
        if stripped.startswith("<|end|>"):
            response = stripped[7:]
        elif stripped.startswith("<|start|>"):
            response = stripped[9:]
        else:
            response = stripped
            break

    # Handle <think> tags from reasoning models - extract content after the tag
    if "</think>" in response:
        think_end = response.rfind("</think>")
        response = response[think_end + 8:].strip()
    
    return response, raw_response, extraction_method_hint


def _extract_json_snippet(response: str, extraction_method_hint: str = "unknown") -> tuple[str, dict | None, str]:
    """
    Extract JSON substring from preprocessed LLM response.
    
    Tries extraction strategies in priority order:
    1. ```json markdown blocks
    2. Response starting with { (uses raw_decode for robustness)
    3. Fallback search for last valid JSON object
    4. Plain ``` markdown blocks
    
    Args:
        response: Preprocessed response text (after wrapper stripping)
        extraction_method_hint: Hint from preprocessing (e.g., "channel_marker")
        
    Returns:
        Tuple of (json_str, decoded_obj, extraction_method)
        - json_str: Extracted JSON substring
        - decoded_obj: Already-parsed dict if raw_decode succeeded, else None
        - extraction_method: String describing which method was used
        
    Raises:
        ValueError: If no JSON found in response
    """
    extraction_method = extraction_method_hint
    decoded_obj = None
    
    # Priority order: ```json > starts with { > fallback search > plain ```
    if "```json" in response:
        if extraction_method == "unknown":
            extraction_method = "markdown_json"
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            json_str = response[start:end].strip()
        else:
            json_str = response[start:].strip()
            
    elif response.strip().startswith("{"):
        if extraction_method == "unknown":
            extraction_method = "brace_matching"
        # Use raw_decode for robust parsing (handles braces inside strings)
        response_stripped = response.strip()
        decoder = json.JSONDecoder()
        try:
            obj, end_idx = decoder.raw_decode(response_stripped)
            if isinstance(obj, dict):
                json_str = response_stripped[:end_idx]
                decoded_obj = obj
            else:
                # Not a dict, fall back to full string
                json_str = response_stripped
        except json.JSONDecodeError:
            # raw_decode failed, use full string and let repair logic handle it
            json_str = response_stripped
            
    else:
        if extraction_method == "unknown":
            extraction_method = "fallback_search"
        # Look for LAST valid JSON object using proper JSON decoding
        brace_positions = [i for i, c in enumerate(response) if c == "{"]

        if not brace_positions:
            # No braces found - try plain markdown as last resort
            if "```" in response:
                if extraction_method not in ("channel_marker",):
                    extraction_method = "markdown_plain"
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
                else:
                    json_str = response[start:].strip()
            else:
                raise ValueError("No JSON found in response")
        else:
            # Use JSONDecoder.raw_decode for robust parsing
            decoder = json.JSONDecoder()
            json_str = None
            
            for start in reversed(brace_positions):
                try:
                    obj, end_idx = decoder.raw_decode(response[start:])
                    if isinstance(obj, dict):
                        json_str = response[start:start + end_idx]
                        decoded_obj = obj
                        break
                except json.JSONDecodeError:
                    continue

            if json_str is None:
                # All JSON decode attempts failed - try plain markdown as last resort
                if "```" in response:
                    if extraction_method not in ("channel_marker",):
                        extraction_method = "markdown_plain"
                    start = response.find("```") + 3
                    end = response.find("```", start)
                    if end != -1:
                        json_str = response[start:end].strip()
                    else:
                        json_str = response[start:].strip()
                else:
                    raise ValueError("No JSON found in response")

    return json_str.strip(), decoded_obj, extraction_method


def _parse_or_repair(json_str: str, decoded_obj: dict | None = None) -> tuple[dict | None, str | None, str | None]:
    """
    Parse JSON string, applying repair steps if initial parse fails.
    
    Repair steps (applied only after initial parse failure):
    1. Escape newlines/tabs/carriage returns
    2. Truncate after last closing brace
    3. Escape unescaped backslashes (e.g., LaTeX)
    4. Fix erroneous escaped quotes
    5. Remove trailing commas
    6. Patch unescaped inner quotes
    7. Convert single quotes (if no double quotes present)
    
    Args:
        json_str: Extracted JSON string
        decoded_obj: Already-decoded dict if available (skips parsing)
        
    Returns:
        Tuple of (parsed_dict, after_regex_clean, cleaned_variant)
        - parsed_dict: Successfully parsed dict, or None if all attempts fail
        - after_regex_clean: JSON after regex cleaning (for debugging)
        - cleaned_variant: Final cleaned JSON variant (for debugging)
    """
    import re
    
    after_regex_clean = None
    cleaned = None
    
    # If we already have a decoded object from raw_decode, use it directly
    if decoded_obj is not None:
        return decoded_obj, None, None
    
    # Try to parse directly first before any mutation
    try:
        parsed_dict = json.loads(json_str)
        return parsed_dict, None, None
    except json.JSONDecodeError:
        pass
    
    # Parse failed - now apply cleaning steps
    # Apply escaping to handle newlines/tabs within JSON strings
    json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    # Ensure the JSON string ends properly with closing brace
    last_brace = json_str.rfind("}")
    if last_brace != -1 and last_brace < len(json_str) - 1:
        json_str = json_str[:last_brace + 1]

    # Clean unescaped backslashes (e.g., from LaTeX notation like \( or \Delta)
    json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)

    # Fix erroneous escaped quotes before field-ending commas
    json_str = re.sub(r'\\" \s*,\s*"', r'", "', json_str)
    json_str = re.sub(r',\\" \s*\\"', r'", "', json_str)

    after_regex_clean = json_str

    # Try to parse after cleaning
    try:
        return json.loads(json_str), after_regex_clean, None
    except json.JSONDecodeError:
        pass
    
    # Remove trailing commas
    cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
    try:
        return json.loads(cleaned), after_regex_clean, cleaned
    except json.JSONDecodeError:
        pass
    
    # Try fixing unescaped inner quotes
    patched_obj, patched_str = _patch_unescaped_quotes(cleaned)
    if patched_obj is not None:
        return patched_obj, after_regex_clean, patched_str
    
    # Last resort: convert single quotes if no double quotes present
    if '"' not in cleaned and "'" in cleaned:
        candidate = cleaned.replace("'", '"')
        try:
            return json.loads(candidate), after_regex_clean, candidate
        except json.JSONDecodeError:
            pass
    
    return None, after_regex_clean, cleaned


def _normalize_dict(parsed_dict: dict) -> dict:
    """
    Normalize parsed dictionary: lowercase keys.
    
    Args:
        parsed_dict: Raw parsed dictionary
        
    Returns:
        Dictionary with lowercase keys
    """
    return {k.lower(): v for k, v in parsed_dict.items()}


def _extract_verdict_fallback(parsed_dict: dict, json_str_original: str, full_response: str = None) -> tuple[dict, bool]:
    """
    Extract verdict from string values if "verdict" key is missing.

    Searches for:
    1. "Verdict: yes/no" pattern inside any string value
    2. Malformed keys like "verdict: \"yes\"" (LLM JSON generation error)
    3. JSON-style "verdict": "yes/no" in extracted JSON string
    4. JSON-style "verdict": "yes/no" in full original response (for verdicts outside extracted snippet)

    Args:
        parsed_dict: Normalized parsed dictionary
        json_str_original: Extracted JSON string for regex fallback
        full_response: Full original response (optional, for verdicts outside JSON snippet)

    Returns:
        Tuple of (updated_dict, verdict_found)
    """
    import re

    if "verdict" in parsed_dict:
        return parsed_dict, False

    verdict_found = False

    # 1) Look for "Verdict: yes/no" style inside any string value
    for value in parsed_dict.values():
        if isinstance(value, str):
            # Match patterns: 'verdict: yes', 'verdict: "yes"', 'verdict:"yes"', 'verdict":"yes"', etc.
            # Handle quotes before AND after colon: verdict":"yes, verdict": "yes, verdict" : "yes, etc.
            match = re.search(r'verdict\s*["\']?\s*:\s*["\']?\s*(\w+)', value, re.IGNORECASE | re.MULTILINE)
            if match:
                potential_verdict = match.group(1).lower()
                if potential_verdict in ["yes", "no"]:
                    parsed_dict["verdict"] = potential_verdict
                    verdict_found = True
                    break

    # 2) Look for malformed keys like 'verdict: "yes"' or 'verdict: "no"'
    #    This handles LLM JSON generation errors where the key itself contains the verdict
    if not verdict_found:
        for key in parsed_dict.keys():
            if isinstance(key, str):
                # Match patterns like: 'verdict: "yes"' or "verdict: 'yes'" or 'verdict:"yes"'
                match = re.search(r'verdict\s*:\s*["\']?(yes|no)["\']?', key, re.IGNORECASE)
                if match:
                    potential_verdict = match.group(1).lower()
                    parsed_dict["verdict"] = potential_verdict
                    verdict_found = True
                    print(f"Warning: Extracted verdict from malformed JSON key '{key}': {potential_verdict}")
                    break

    # 3) If still not found, look for a JSON-style "verdict": "yes/no" in extracted JSON string
    #    Handle both regular quotes and escaped quotes (\" or \\")
    if not verdict_found:
        match = re.search(
            r'\\*["\']verdict\\*["\']\s*:\s*\\*["\'](yes|no)\\*["\']',
            json_str_original,
            re.IGNORECASE,
        )
        if match:
            parsed_dict["verdict"] = match.group(1).lower()
            verdict_found = True

    # 4) If still not found and full response provided, search in full response
    #    This handles cases like: {"reasoning":"..."} "verdict":"no"}
    #    where verdict is outside the extracted JSON snippet
    #    Also handles escaped quotes like: rubric,\"verdict\": \"no\"}
    if not verdict_found and full_response is not None:
        match = re.search(
            r'\\*["\']verdict\\*["\']\s*:\s*\\*["\'](yes|no)\\*["\']',
            full_response,
            re.IGNORECASE,
        )
        if match:
            parsed_dict["verdict"] = match.group(1).lower()
            verdict_found = True
            print(f"Warning: Extracted verdict from outside JSON snippet: {match.group(1)}")

    return parsed_dict, verdict_found


def _ensure_verdict(result_dict: dict, fallback_dict: dict[str, str]) -> dict:
    """
    Ensure result dictionary has a verdict key.
    
    Args:
        result_dict: Current result dictionary
        fallback_dict: Fallback dictionary with default values
        
    Returns:
        Dictionary with guaranteed "verdict" key
    """
    if "verdict" not in result_dict:
        result_dict["verdict"] = fallback_dict.get("verdict", "no")
    return result_dict


def _compute_token_indices(
    raw_response: str,
    json_str_original: str,
    log_probs: list[dict],
    parsed_dict: dict
) -> dict[str, int]:
    """
    Compute token indices for JSON elements in log_probs.
    
    Args:
        raw_response: Original unstripped response (for accurate offsets)
        json_str_original: Unmodified JSON substring
        log_probs: List of token log probability entries
        parsed_dict: Parsed JSON dictionary
        
    Returns:
        Dictionary with token indices for 'json_start', 'json_end', 'verdict'
    """
    return _find_token_indices(raw_response, json_str_original, log_probs, parsed_dict)


def _log_parse_failure(
    error: Exception,
    raw_response: str,
    extraction_method: str,
    extracted_json: str | None,
    after_regex_clean: str | None,
    cleaned: str | None,
    logger,
    instance_id: str | None,
    preview_chars: int = 500
) -> None:
    """
    Log parse failure details to console and optional logger.
    
    Args:
        error: The exception that occurred
        raw_response: Original LLM response
        extraction_method: Which extraction method was attempted
        extracted_json: Extracted JSON string (if available)
        after_regex_clean: JSON after regex cleaning (if available)
        cleaned: Final cleaned variant (if available)
        logger: Optional LogManager instance
        instance_id: Optional instance ID for logging context
    """
    error_msg = f"Failed to parse JSON response: {error}"
    print(f"Warning: {error_msg}")

    if cleaned is not None:
        print(f"cleaned: {cleaned}")

    try:
        preview_chars = int(preview_chars)
    except (TypeError, ValueError):
        preview_chars = 500
    if preview_chars < 0:
        preview_chars = 0

    if logger and instance_id:
        debug_info = {"extraction_method": extraction_method}
        if extracted_json is not None:
            debug_info["extracted_json"] = extracted_json
        if after_regex_clean is not None:
            debug_info["after_regex_clean"] = after_regex_clean
        if cleaned is not None:
            debug_info["after_comma_clean"] = cleaned

        logger.log_parse_error(
            instance_id=instance_id,
            error_message=str(error),
            full_response=raw_response,
            additional_info=debug_info,
        )
    else:
        truncated = raw_response[:preview_chars] + "..." if len(raw_response) > preview_chars else raw_response
        print(f"Raw response (first {preview_chars} chars): {repr(truncated)}")


def parse_json_response(
    response: str,
    fallback_dict: dict[str, str],
    log_probs: list[dict] | None = None,
    logger=None,
    instance_id: str | None = None,
    preview_chars: int = 500,
) -> dict | tuple[dict, dict]:
    """
    Parse JSON response with fallback for malformed JSON.
    
    Orchestrates the parsing pipeline:
    1. Strip wrapper markers (channel markers, think tags, etc.)
    2. Extract JSON snippet from response
    3. Parse or repair the JSON
    4. Normalize keys and extract verdict fallbacks
    5. Compute token indices if log_probs provided

    Args:
        response: Raw text response from LLM
        fallback_dict: Dictionary to return if parsing fails
        log_probs: Optional list of lists where each inner list is [token, logprob, ...]
        logger: Optional LogManager instance for logging parse errors
        instance_id: Optional instance ID for logging context

    Returns:
        If log_probs is None:
            Dict: Parsed JSON dictionary
        If log_probs is provided:
            Tuple[Dict, Dict]: (parsed_dict, token_indices) where token_indices contains:
                - 'json_start': Index of first JSON token in log_probs
                - 'json_end': Index of last JSON token in log_probs
                - 'verdict': Index of verdict value token (if found)
    """
    try:
        preview_chars = int(preview_chars)
    except (TypeError, ValueError):
        preview_chars = 500
    if preview_chars < 0:
        preview_chars = 0

    # Initialize tracking variables for error reporting
    extraction_method = "unknown"
    extracted_json_original = None
    after_regex_clean = None
    cleaned = None
    raw_response = response
    
    try:
        # 1. Strip wrapper markers
        cleaned_response, raw_response, extraction_method = _strip_wrappers(response)
        
        # 2. Extract JSON snippet
        json_str, decoded_obj, extraction_method = _extract_json_snippet(
            cleaned_response, extraction_method
        )
        json_str_original = json_str
        extracted_json_original = json_str
        
        # 3. Parse or repair
        parsed_dict, after_regex_clean, cleaned = _parse_or_repair(json_str, decoded_obj)
        
        # 4. Normalize and extract verdict fallbacks
        # IMPORTANT: Try fallback extraction EVEN if parsing failed (parsed_dict is None)
        # This allows us to extract verdict from malformed JSON using regex on the raw response
        if parsed_dict is not None:
            parsed_dict = _normalize_dict(parsed_dict)
        else:
            # Parsing failed completely - start with empty dict for fallback extraction
            parsed_dict = {}

        parsed_dict, verdict_found = _extract_verdict_fallback(
            parsed_dict, extracted_json_original, full_response=raw_response
        )
        if verdict_found:
            print(f"Warning: Extracted verdict from text using fallback parsing: {parsed_dict['verdict']}")
        
        # 5. Build result dict with fallback
        result_dict = parsed_dict if parsed_dict is not None else dict(fallback_dict)
        result_dict = _ensure_verdict(result_dict, fallback_dict)
        
        # 6. Compute token indices if requested
        if log_probs is not None:
            token_indices = {}
            if parsed_dict is not None:
                token_indices = _compute_token_indices(
                    raw_response, json_str_original, log_probs, parsed_dict
                )
            return result_dict, token_indices
        else:
            return result_dict

    except (json.JSONDecodeError, ValueError) as e:
        # Log the failure
        _log_parse_failure(
            error=e,
            raw_response=raw_response,
            extraction_method=extraction_method,
            extracted_json=extracted_json_original,
            after_regex_clean=after_regex_clean,
            cleaned=cleaned,
            logger=logger,
            instance_id=instance_id,
            preview_chars=preview_chars,
        )

        # Return fallback dict with raw response
        result = dict(fallback_dict)
        result["raw_response"] = raw_response
        print("Using fallback dictionary for this response.")
        print("Fallback dict:\n", result)
        print("-" * 40)
        return result if log_probs is None else (result, {})

    except Exception as e:
        # CRITICAL: Catch-all to ensure parsing NEVER crashes the entire run
        # This handles IndexError, KeyError, AttributeError, etc.
        print(f"ERROR: Unexpected exception in parse_json_response: {type(e).__name__}: {e}")
        print(f"Response (first {preview_chars} chars): {response[:preview_chars] if response else 'None'}")

        # Log if logger available
        if logger and instance_id:
            logger.log_parse_error(
                instance_id=instance_id,
                error_message=f"Unexpected {type(e).__name__}: {e}",
                full_response=response if response else "",
                additional_info={"exception_type": type(e).__name__}
            )

        # Return fallback dict - NEVER crash
        result = dict(fallback_dict)
        result["raw_response"] = response if response else ""
        result["parse_error"] = f"{type(e).__name__}: {e}"
        print("Using fallback dictionary due to unexpected error.")
        return result if log_probs is None else (result, {})
