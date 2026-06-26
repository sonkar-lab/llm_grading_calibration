# LLM Inference Module

**Single Responsibility:** Batched LLM inference (conversations → responses)

This module provides a unified interface for batched LLM generation across different backends (OpenAI API, HuggingFace, vLLM) without worrying about implementation details.

## Quick Start - Simple Example

### 1. Prepare Your Data

Create a CSV file with questions:

```csv
question_id,question
1,What is the capital of France?
2,Explain photosynthesis in one sentence.
3,What is 15 multiplied by 23?
```

### 2. Run Batch Q&A

```bash
# Using GPT API
python llm_inference/example_batch_qa.py \
    --input llm_inference/sample_questions.csv \
    --output my_answers.csv \
    --model gpt-4o-mini \
    --batch-size 10

# Using local model with vLLM
python llm_inference/example_batch_qa.py \
    --input llm_inference/sample_questions.csv \
    --output my_answers.csv \
    --model meta-llama/Llama-2-7b \
    --backend vllm \
    --num-gpus 2 \
    --cache-dir /data1/ss164/cache/
```

### 3. Get Results

Output CSV will have your answers:

```csv
question_id,question,answer
1,What is the capital of France?,The capital of France is Paris.
2,Explain photosynthesis in one sentence.,"Photosynthesis is the process..."
3,What is 15 multiplied by 23?,15 multiplied by 23 equals 345.
```

**That's it!** No need to think about model loading, API calls, or batching logic.

---

## Files in This Directory

- **`inference_llm.py`** - Core inference engine (don't modify this)
- **`example_batch_qa.py`** - Simple example: CSV questions → CSV answers
- **`sample_questions.csv`** - Sample input data for testing
- **`README.md`** - This file

---

## For Collaborators: Using the Inference Engine

### The Pattern (3 Simple Steps)

```python
from grading_calibration.inference.inference_llm import BatchedLLMInference

# Step 1: Initialize inference engine ONCE
inference = BatchedLLMInference(
    model_name="gpt-4o-mini",
    backend="gpt",
    temperature=0.7,
    batch_size=8
)

# Step 2: Create conversations
conversations = [
    [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "What is 2+2?"}
    ],
    [
        {"role": "user", "content": "Explain gravity"}
    ]
]

# Step 3: Get answers
answers = inference.generate(conversations)
# Returns: ["The answer is 4.", "Gravity is a force..."]
```

### What You Need to Know

#### ✅ What to Provide

1. **Conversations**: List of message lists
   ```python
   [
       [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
       [{"role": "user", "content": "..."}],
       ...
   ]
   ```

2. **Model name**: `"gpt-4o-mini"`, `"meta-llama/Llama-2-7b"`, etc.

3. **Backend**: `"gpt"` (API) or `"hf"` (HuggingFace) or `"vllm"`

#### ❌ What You DON'T Need to Worry About

- Model loading/initialization
- API rate limiting
- Batching logic
- Token management
- Backend-specific code
- Error handling

---

## Common Use Cases

### Use Case 1: Question Answering

See `example_batch_qa.py` - complete working example with CSV input/output.

### Use Case 2: Custom Prompts in Your Pipeline

```python
from grading_calibration.inference.inference_llm import BatchedLLMInference
import pandas as pd

# Initialize ONCE
inference = BatchedLLMInference(model_name="gpt-4o-mini", backend="gpt")

# Read your data
df = pd.read_csv("my_data.csv")

# Create custom conversations
conversations = []
for _, row in df.iterrows():
    conv = [
        {"role": "system", "content": "You are a student simulation..."},
        {"role": "user", "content": f"Explain your reasoning for: {row['question']}"}
    ]
    conversations.append(conv)

# Generate responses
responses = inference.generate(conversations)

# Save results
df['response'] = responses
df.to_csv("results.csv", index=False)
```

### Use Case 3: Process Different Batches with Same Model

```python
# Initialize once
inference = BatchedLLMInference(model_name="gpt-4o-mini", backend="gpt")

# Process batch 1
batch1_conversations = [...]
batch1_responses = inference.generate(batch1_conversations)

# Process batch 2 (NO reloading! Same model)
batch2_conversations = [...]
batch2_responses = inference.generate(batch2_conversations)

# Process batch 3 (Still same model!)
batch3_conversations = [...]
batch3_responses = inference.generate(batch3_conversations)
```

### Use Case 4: Override Parameters Per Call

```python
# Initialize with defaults
inference = BatchedLLMInference(
    model_name="gpt-4o-mini",
    backend="gpt",
    temperature=0.7,  # Default
    max_tokens=512    # Default
)

# Use default settings
responses1 = inference.generate(conversations1)

# Override temperature for creative generation
responses2 = inference.generate(
    conversations2,
    temperature=1.0  # More creative
)

# Override for deterministic output
responses3 = inference.generate(
    conversations3,
    temperature=0.0  # Deterministic
)
```

---

## Command-Line Arguments

### For `example_batch_qa.py`

#### Required
- `--input`: Input CSV path (must have `question_id` and `question` columns)
- `--output`: Output CSV path (will have `question_id`, `question`, `answer`)

#### Model Configuration
- `--model`: Model name (default: `gpt-4o-mini`)
- `--backend`: `gpt`, `hf`, or `vllm` (default: `gpt`)

#### Generation Parameters
- `--temperature`: Sampling temperature 0.0-1.0 (default: `0.7`)
- `--batch-size`: Questions per batch (default: `8`)
- `--max-tokens`: Max tokens per answer (default: `512`)

#### Optional
- `--system-prompt`: Custom system message
- `--cache-dir`: Model download directory (local models only)
- `--num-gpus`: GPUs for vLLM (default: `1`)

### Examples

```bash
# Simple GPT run
python llm_inference/example_batch_qa.py \
    --input questions.csv \
    --output answers.csv

# Custom system prompt
python llm_inference/example_batch_qa.py \
    --input questions.csv \
    --output answers.csv \
    --system-prompt "You are a physics tutor. Explain concepts clearly."

# High creativity
python llm_inference/example_batch_qa.py \
    --input questions.csv \
    --output answers.csv \
    --temperature 1.0

# Large batch processing
python llm_inference/example_batch_qa.py \
    --input questions.csv \
    --output answers.csv \
    --batch-size 32

# Local model with vLLM
python llm_inference/example_batch_qa.py \
    --input questions.csv \
    --output answers.csv \
    --model meta-llama/Llama-2-7b \
    --backend vllm \
    --num-gpus 4
```

---

## Supported Backends

### 1. GPT (OpenAI API)

**When to use:** You have API access and want fast, high-quality results

```python
inference = BatchedLLMInference(
    model_name="gpt-4o-mini",  # or gpt-4, gpt-3.5-turbo, etc.
    backend="gpt"
)
```

**Supported models:**
- `gpt-4`, `gpt-4o-mini`, `gpt-3.5-turbo`
- `gpt-5-mini-2025-08-07`
- `deepseek-chat`, `deepseek-reasoner`

### 2. HuggingFace Transformers

**When to use:** Running locally, small-medium models, single GPU

```python
inference = BatchedLLMInference(
    model_name="meta-llama/Llama-2-7b-hf",
    backend="hf",
    cache_dir="/path/to/cache"
)
```

### 3. vLLM

**When to use:** Running locally, large models, multi-GPU, need speed

```python
inference = BatchedLLMInference(
    model_name="meta-llama/Llama-2-70b-hf",
    backend="vllm",
    num_gpus=4,
    cache_dir="/path/to/cache"
)
```

**Optimized for:**
- Qwen3, DeepSeek, Phi, GPT-OSS, CWM models
- Multi-GPU tensor parallelism
- High-throughput batched inference

---