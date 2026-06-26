"""
Batched LLM Inference Engine
Single responsibility: Convert conversations -> responses
Supports: OpenAI API, HuggingFace Transformers, vLLM
"""

##
## TODO: FINISH SUPPORT FOR RETURNING LOGPROBS
##
##
import time
from typing import List, Dict, Any, Optional
from multiprocessing.pool import ThreadPool
from openai import OpenAI, OpenAIError
import torch
import torch.nn.functional as F
import os


class BatchedLLMInference:
    """
    Unified inference engine for batched LLM generation.
    Supports GPT API, HuggingFace transformers, and vLLM backends.

    Usage:
        # Initialize once - model loaded into memory
        inference = BatchedLLMInference(
            model_name="gpt-4",
            backend="gpt",
            temperature=0.7,
            batch_size=8
        )

        # Use many times - no reloading
        conversations = [[{"role": "user", "content": "Hello"}], ...]
        responses = inference.generate(conversations)
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "gpt",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        batch_size: int = 8,
        cache_dir: Optional[str] = None,
        num_gpus: int = 1,
        no_cuda: bool = False,
        # Model-specific settings
        enable_thinking: bool = False,
        gpt_reasoning: str = "medium",
        # vLLM-specific settings
        vllm_multiproc_method: str = "spawn",
        vllm_max_num_seqs: Optional[int] = None,
        vllm_max_model_len: Optional[int] = None,
        vllm_gpu_memory_utilization: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize inference engine with model/client.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "meta-llama/Llama-2-7b")
            backend: "gpt" (OpenAI API), "hf" (HuggingFace), or "vllm"
            temperature: Sampling temperature (0.0-1.0), None uses model-specific default
            max_tokens: Maximum tokens to generate, None uses model-specific default
            top_p: Top-p sampling parameter, None uses model-specific default
            top_k: Top-k sampling parameter, None uses model-specific default
            min_p: Min-p sampling parameter, None uses model-specific default
            batch_size: Batch size for generation (GPT: parallel threads, local: true batching)
            cache_dir: Directory for model downloads (local models only)
            num_gpus: Number of GPUs for vLLM tensor parallelism
            no_cuda: Force CPU usage even if CUDA available
            enable_thinking: Enable thinking/reasoning mode (Qwen3, GPT-OSS, CWM models)
            gpt_reasoning: GPT-OSS reasoning level ("low", "medium", "high")
            vllm_multiproc_method: vLLM multiprocessing start method ("spawn" or "fork", default: "spawn")
                                   Use "spawn" to avoid CUDA re-initialization errors
            vllm_max_num_seqs: Maximum number of sequences to process in a batch (vLLM only, default: 256)
            vllm_max_model_len: Maximum sequence length in tokens (vLLM only, default: model's context length)
            vllm_gpu_memory_utilization: Fraction of GPU memory to use (vLLM only, default: 0.9, range: 0.0-1.0)
            **kwargs: Additional backend-specific parameters
                if vllm kwargs:
                    Args:
                        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
                        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
                            if available, and "slow" will always use the slow tokenizer.
                        skip_tokenizer_init: If true, skip initialization of tokenizer and
                            detokenizer. Expect valid prompt_token_ids and None for prompt
                            from the input.
                        allowed_local_media_path: Allowing API requests to read local images
                            or videos from directories specified by the server file system.
                            This is a security risk. Should only be enabled in trusted
                            environments.
                        allowed_media_domains: If set, only media URLs that belong to this
                            domain can be used for multi-modal inputs.
                        dtype: The data type for the model weights and activations. Currently,
                            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
                            the `dtype` attribute of the Transformers model's config. However,
                            if the `dtype` in the config is `float32`, we will use `float16` instead.
                        quantization: The method used to quantize the model weights. Currently,
                            we support "awq", "gptq", and "fp8" (experimental).
                            If None, we first check the `quantization_config` attribute in the
                            model config file. If that is None, we assume the model weights are
                            not quantized and use `dtype` to determine the data type of
                            the weights.
                        revision: The specific model version to use. It can be a branch name,
                            a tag name, or a commit id.
                        tokenizer_revision: The specific tokenizer version to use. It can be a
                            branch name, a tag name, or a commit id.
                        seed: The seed to initialize the random number generator for sampling.
                        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
                            reserve for the model weights, activations, and KV cache. Higher
                            values will increase the KV cache size and thus improve the model's
                            throughput. However, if the value is too high, it may cause out-of-
                            memory (OOM) errors.
                        kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,
                            this is set to None and vllm can automatically infer the kv cache
                            size based on gpu_memory_utilization. However, users may want to
                            manually specify the kv cache memory size. kv_cache_memory_bytes
                            allows more fine-grain control of how much memory gets used when
                            compared with using gpu_memory_utilization. Note that
                            kv_cache_memory_bytes (when not-None) ignores
                            gpu_memory_utilization
                        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
                            This can be used for temporarily storing the states of the requests
                            when their `best_of` sampling parameters are larger than 1. If all
                            requests will have `best_of=1`, you can safely set this to 0.
                            Noting that `best_of` is only supported in V0. Otherwise, too small
                            values may cause out-of-memory (OOM) errors.
                        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
                            the model weights. This virtually increases the GPU memory space
                            you can use to hold the model weights, at the cost of CPU-GPU data
                            transfer for every forward pass.
                        enforce_eager: Whether to enforce eager execution. If True, we will
                            disable CUDA graph and always execute the model in eager mode.
                            If False, we will use CUDA graph and eager execution in hybrid.
                        disable_custom_all_reduce: See
                            [ParallelConfig][vllm.config.ParallelConfig].
                        hf_token: The token to use as HTTP bearer authorization for remote files
                            . If `True`, will use the token generated when running
                            `huggingface-cli login` (stored in `~/.huggingface`).
                        hf_overrides: If a dictionary, contains arguments to be forwarded to the
                            HuggingFace config. If a callable, it is called to update the
                            HuggingFace config.
                        mm_processor_kwargs: Arguments to be forwarded to the model's processor
                            for multi-modal data, e.g., image processor. Overrides for the
                            multi-modal processor obtained from `AutoProcessor.from_pretrained`.
                            The available overrides depend on the model that is being run.
                            For example, for Phi-3-Vision: `{"num_crops": 4}`.
                        pooler_config: Initialize non-default pooling config for the pooling
                            model. e.g. `PoolerConfig(pooling_type="mean", normalize=False)`.
                        override_pooler_config: [DEPRECATED] Use `pooler_config` instead. This
                            argument is deprecated and will be removed in v0.12.0 or v1.0.0,
                            whichever is sooner.
                        compilation_config: Either an integer or a dictionary. If it is an
                            integer, it is used as the mode of compilation optimization. If it
                            is a dictionary, it can specify the full compilation configuration.
                        **kwargs: Arguments for [`EngineArgs`][vllm.EngineArgs].

        """
        self.model_name = model_name
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.num_gpus = num_gpus
        self.enable_thinking = enable_thinking
        self.gpt_reasoning = gpt_reasoning
        self.vllm_multiproc_method = vllm_multiproc_method
        self.vllm_max_num_seqs = vllm_max_num_seqs
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.kwargs = kwargs

        # Model type detection
        self.is_deepseek = "deepseek" in model_name.lower()
        self.is_qwen3 = "qwen3" in model_name.lower()
        self.is_phi = "phi" in model_name.lower()
        self.is_gpt_oss = "gpt-oss" in model_name.lower()
        self.is_cwm = "cwm" in model_name.lower()

        print(f"Initializing BatchedLLMInference:")
        print(f"  Model: {model_name}")
        print(f"  Backend: {backend}")
        print(
            f"  Temperature: {temperature if temperature is not None else 'model default'}"
        )
        print(
            f"  Max tokens: {max_tokens if max_tokens is not None else 'model default'}"
        )
        print(f"  Top-p: {top_p if top_p is not None else 'model default'}")
        print(f"  Top-k: {top_k if top_k is not None else 'model default'}")
        print(f"  Min-p: {min_p if min_p is not None else 'model default'}")
        print(f"  Batch size: {batch_size}")

        if backend == "vllm":
            print(f"  vLLM Settings:")
            print(
                f"    max_num_seqs: {vllm_max_num_seqs if vllm_max_num_seqs is not None else 'default (256)'}"
            )
            print(
                f"    max_model_len: {vllm_max_model_len if vllm_max_model_len is not None else 'model default'}"
            )
            print(
                f"    gpu_memory_utilization: {vllm_gpu_memory_utilization if vllm_gpu_memory_utilization is not None else 'default (0.9)'}"
            )
        # Initialize model/client based on backend
        if backend == "gpt":
            self._init_gpt_client()
        else:
            self._init_local_model(no_cuda)

        print("Initialization complete!")

    def _init_gpt_client(self):
        """Initialize OpenAI client for API calls."""
        if self.model_name.startswith("deepseek"):
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
            print("Initialized DeepSeek API client")
        else:
            # Use official OpenAI API
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it with: export OPENAI_API_KEY='your-key-here'"
                )

            self.client = OpenAI(api_key=api_key)
            print("Initialized OpenAI API client")

        self.model = None
        self.tokenizer = None
        self.device = None

    def _init_local_model(self, no_cuda: bool):
        """Initialize local model and tokenizer for HF/vLLM backends."""
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Set vLLM multiprocessing method BEFORE any CUDA operations
        # This fixes "Cannot re-initialize CUDA in forked subprocess" errors
        if self.backend == "vllm":
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = self.vllm_multiproc_method
            print(f"Set VLLM_WORKER_MULTIPROC_METHOD={self.vllm_multiproc_method}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
        print(f"Using device: {self.device}")
        print(f"Loading model {self.model_name}...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                print("Warning: Tokenizer missing pad_token, setting to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model based on backend
            if self.backend == "hf":
                model_kwargs = {
                    "torch_dtype": (
                        torch.bfloat16 if self.device.type == "cuda" else torch.float32
                    ),
                }
                if self.cache_dir:
                    model_kwargs["cache_dir"] = self.cache_dir
                if self.device.type == "cuda":
                    model_kwargs["device_map"] = "auto"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_kwargs
                )

                if self.device.type == "cpu":
                    self.model.to(self.device)

            elif self.backend == "vllm":
                from vllm import LLM

                vllm_kwargs = {
                    "model": self.model_name,
                    "download_dir": self.cache_dir,
                    "tensor_parallel_size": self.num_gpus,
                    "trust_remote_code": True,
                    **self.kwargs,
                }
                # Add optional vLLM-specific parameters if provided
                if self.vllm_max_num_seqs is not None:
                    vllm_kwargs["max_num_seqs"] = self.vllm_max_num_seqs
                if self.vllm_max_model_len is not None:
                    vllm_kwargs["max_model_len"] = self.vllm_max_model_len
                if self.vllm_gpu_memory_utilization is not None:
                    vllm_kwargs["gpu_memory_utilization"] = (
                        self.vllm_gpu_memory_utilization
                    )

                # Model-specific vLLM settings (can be overridden by explicit parameters)
                if self.is_deepseek and "max_model_len" not in vllm_kwargs:
                    vllm_kwargs["max_model_len"] = 32768

                print(f"vLLM initialization with args:\n{vllm_kwargs}")
                self.model = LLM(**vllm_kwargs)

            print("Model loaded successfully")
            self.client = None

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(
        self,
        conversations: list[list[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "nl",
        top_k_log_probs: int | None = None,
        seed: Optional[int] = None,
    ) -> List[str] | List[dict[str, Any]]:
        """
        Generate responses for a batch of conversations.

        Args:
            conversations: List of conversation histories
                Example: [
                    [{"role": "system", "content": "You are helpful"},
                     {"role": "user", "content": "What is 2+2?"}],
                    [{"role": "user", "content": "Explain gravity"}],
                    ...
                ]
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call
            task_type: Task type hint ("nl" or "code") for model-specific tuning

        Returns:
            List of generated response strings, one per conversation
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.backend == "gpt":
            return self._generate_gpt_batch(
                conversations, temp, max_tok, top_k_log_probs, seed
            )
        else:
            return self._generate_local_batch(
                conversations, temp, max_tok, task_type, top_k_log_probs, seed
            )

    def _generate_gpt_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        top_k_log_probs: int | None = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate using GPT API with parallel ThreadPool."""
        pool = ThreadPool(processes=self.batch_size)

        start_time = time.time()
        print(f"Starting GPT API calls for {len(conversations)} conversations...")

        responses = pool.starmap(
            self._run_gpt4_single,
            [
                (conv, self.model_name, max_tokens, temperature, top_k_log_probs, seed)
                for conv in conversations
            ],
        )

        pool.close()
        pool.join()

        end_time = time.time()
        print(f"API calls completed in {end_time - start_time:.2f} seconds")

        return responses

    def _run_gpt4_single(
        self,
        conversation: List[Dict[str, str]],
        engine: str,
        max_tokens: int,
        temperature: float,
        top_k_log_probs: int | None = None,
        seed: Optional[int] = None,
    ) -> str:
        """Run single GPT API call with error handling."""
        try:
            if engine.startswith("gpt-4"):
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=top_k_log_probs is not None,
                    top_k_log_probs=top_k_log_probs,
                    seed=seed,
                )
            elif engine.startswith("gpt-5"):
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    logprobs=top_k_log_probs is not None,
                    top_k_log_probs=top_k_log_probs,
                    seed=seed,
                    max_completion_tokens=(
                        max_tokens if max_tokens is not None else 4096
                    ),
                )
            elif engine.startswith("deepseek"):
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    temperature=0.0,
                    max_tokens=4096,
                    stream=False,
                    logprobs=top_k_log_probs is not None,
                    top_k_log_probs=top_k_log_probs,
                    seed=seed,
                )
            else:
                # Default case
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=top_k_log_probs is not None,
                    top_k_log_probs=top_k_log_probs,
                    seed=seed,
                )

            return (
                response.choices[0].message.content.strip()
                if top_k_log_probs is None
                else {
                    "text": response.choices[0].message.content.strip(),
                    "logprobs": response.choices[0].logprobs,
                }
            )
        except OpenAIError as exc:
            print(f"OpenAI error occurred: {exc}, retrying...")
            return "FAIL"

    def _generate_local_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        task_type: str,
        top_k_log_probs: int | None = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate using local model (HuggingFace or vLLM backend)."""
        import torch

        # Apply chat templates and prepare prompts
        input_texts = self._prepare_prompts(conversations)

        if self.backend == "hf":
            return self._generate_hf(
                input_texts, temperature, max_tokens, task_type, top_k_log_probs, seed
            )
        elif self.backend == "vllm":
            return self._generate_vllm(
                input_texts, temperature, max_tokens, task_type, top_k_log_probs, seed
            )

    def _get_sampling_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "nl",
        top_k_log_probs: int | None = None,
    ) -> Dict[str, Any]:
        """
        Resolve sampling parameters with priority:
        1. User-provided instance variables (self.temperature, self.top_p, etc.)
        2. Model-specific defaults
        3. General defaults

        Args:
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            task_type: Task type hint (nl or code) for model-specific tuning

        Returns:
            Dictionary with resolved sampling parameters
        """
        # Start with model-specific defaults
        if self.is_qwen3:
            if self.enable_thinking:
                defaults = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0,
                    "max_tokens": 32768,
                }
            else:
                defaults = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0,
                    "max_tokens": 32768,
                }
        elif self.is_phi:
            defaults = {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": None,
                "min_p": None,
                "max_tokens": 4096,
            }
        elif self.is_gpt_oss:
            gpt_oss_temp = 0.3 if task_type == "code" else 1.0
            defaults = {
                "temperature": gpt_oss_temp,
                "top_p": 1.0,
                "top_k": None,
                "min_p": None,
                "max_tokens": 131072,
            }
        elif self.is_cwm:
            defaults = {
                "temperature": 0.3,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": None,
                "max_tokens": 2048,
            }
        else:
            # Generic defaults
            defaults = {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": None,
                "max_tokens": 8192,
            }

        # Resolve parameters with priority: call param > instance var > model default
        params = {}

        # Temperature: call param > instance var > model default
        params["temperature"] = (
            temperature
            if temperature is not None
            else (
                self.temperature
                if self.temperature is not None
                else defaults["temperature"]
            )
        )
        if temperature is None and self.temperature is None:
            print(f"Using default temperature={params['temperature']} for model")

        # Max tokens: call param > instance var > model default
        params["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (
                self.max_tokens
                if self.max_tokens is not None
                else defaults["max_tokens"]
            )
        )
        if max_tokens is None and self.max_tokens is None:
            print(f"Using default max_tokens={params['max_tokens']} for model")

        # Top-p: instance var > model default
        params["top_p"] = self.top_p if self.top_p is not None else defaults["top_p"]
        if self.top_p is None:
            print(f"Using default top_p={params['top_p']} for model")

        # Top-k: instance var > model default
        params["top_k"] = self.top_k if self.top_k is not None else defaults["top_k"]
        if self.top_k is None:
            print(f"Using default top_k={params['top_k']} for model")

        # Min-p: instance var > model default
        params["min_p"] = self.min_p if self.min_p is not None else defaults["min_p"]
        if self.min_p is None:
            print(f"Using default min_p={params['min_p']} for model")

        params["logprobs"] = top_k_log_probs

        return params

    def _prepare_prompts(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        """Apply model-specific chat templates to conversations."""
        # Handle models that don't support system messages
        if self.is_qwen3 or self.is_deepseek:
            modified_conversations = []
            for conversation in conversations:
                system_msg = next(
                    (msg for msg in conversation if msg["role"] == "system"), None
                )
                user_msg = next(
                    (msg for msg in conversation if msg["role"] == "user"), None
                )

                system_content = system_msg["content"] if system_msg else ""
                user_content = user_msg["content"] if user_msg else ""

                combined_content = (
                    f"{system_content}\n\n{user_content}"
                    if system_content
                    else user_content
                )
                modified_conversations.append(
                    [{"role": "user", "content": combined_content}]
                )

            if self.is_qwen3:
                return [
                    self.tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                    for conv in modified_conversations
                ]
            else:  # DeepSeek
                return [
                    self.tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True
                    )
                    for conv in modified_conversations
                ]

        # GPT-OSS with reasoning
        elif self.is_gpt_oss:
            modified_conversations = []
            for conversation in conversations:
                system_msg = next(
                    (msg for msg in conversation if msg["role"] == "system"), None
                )
                user_msg = next(
                    (msg for msg in conversation if msg["role"] == "user"), None
                )

                system_content = system_msg["content"] if system_msg else ""
                user_content = user_msg["content"] if user_msg else ""

                if self.enable_thinking:  # Enable reasoning mode for GPT-OSS
                    modified_conversations.append(
                        [
                            {
                                "role": "system",
                                "content": f"{system_content}\n\nReasoning: {self.gpt_reasoning}",
                            },
                            {"role": "user", "content": user_content},
                        ]
                    )
                else:
                    modified_conversations.append(
                        [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content},
                        ]
                    )

            return [
                self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                for conv in modified_conversations
            ]

        # CWM with thinking
        elif self.is_cwm:
            return [
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                for conv in conversations
            ]

        # Standard chat template
        else:
            return [
                self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                for conv in conversations
            ]

    def _generate_hf(
        self,
        input_texts: List[str],
        temperature: float,
        max_tokens: int,
        task_type: str,
        top_k_log_probs: int | None = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate using HuggingFace transformers."""
        import torch

        model_inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding=True
        ).to(self.device)
        input_ids = model_inputs.input_ids
        prompt_token_lengths = [len(input_ids[j]) for j in range(len(input_ids))]

        # Get resolved sampling parameters (user > model defaults > general defaults)
        params = self._get_sampling_params(temperature, max_tokens, task_type)

        # Model-specific max_tokens adjustments
        max_token_scaling = 10
        max_model_tokens = self.tokenizer.model_max_length

        if self.is_qwen3:
            print("Using special settings for Qwen models")
            max_model_tokens = 32768
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling),
            )
            print(
                f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}"
            )
            print(
                f"Using temperature={params['temperature']} ({'thinking' if self.enable_thinking else 'non-thinking'} mode)"
            )

        elif self.is_phi:
            print("Using special settings for Phi models")
            adjusted_max_tokens = 4096
            print(
                f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}"
            )

        elif self.is_gpt_oss:
            print("Using special settings for GPT-OSS models")
            max_model_tokens = 131072
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling),
            )
            print(
                f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}"
            )
            print(
                f"Using temperature={params['temperature']} for task_type={task_type}"
            )

        elif self.is_cwm:
            print("Using custom settings for CWM generation")
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling),
            )
            print(
                f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}"
            )
            print(f"Using temperature={params['temperature']}")

        else:
            print("Using generic settings for model generation")
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling),
            )
            print(
                f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}"
            )

        # Build generation kwargs with resolved parameters
        # Only include non-None parameters
        gen_kwargs = {
            **model_inputs,
            "temperature": params["temperature"],
            "max_new_tokens": adjusted_max_tokens,
            "do_sample": True if params["temperature"] > 0 else False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": top_k_log_probs is not None,
            "output_scores": top_k_log_probs is not None,
        }

        if params["top_p"] is not None:
            gen_kwargs["top_p"] = params["top_p"]
        if params["top_k"] is not None:
            gen_kwargs["top_k"] = params["top_k"]
        if params["min_p"] is not None:
            gen_kwargs["min_p"] = params["min_p"]
        if params["logprobs"] is not None:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**gen_kwargs)

        if top_k_log_probs is not None:
            # Extract scores and sequences from the GenerateDecoderOnlyOutput
            scores = generated_ids.scores  # Tuple of (batch_size, vocab_size) tensors
            generated_sequences = (
                generated_ids.sequences
            )  # (batch_size, seq_len) tensor

            results = []
            for batch_idx in range(len(input_ids)):
                # Decode the generated text
                prompt_len = len(input_ids[batch_idx])
                gen_tokens = generated_sequences[batch_idx][prompt_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

                # Extract log probs for each generated token
                token_logprobs = []
                for step_idx, step_scores in enumerate(scores):
                    if step_idx >= len(gen_tokens):
                        break

                    # Get logits for this batch item and convert to log probs
                    logits = step_scores[batch_idx]  # Shape: (vocab_size,)
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get the actual generated token
                    token_id = gen_tokens[step_idx].item()
                    token_logprob = log_probs[token_id].item()

                    # Get top-k alternatives
                    top_values, top_indices = torch.topk(
                        log_probs, k=min(top_k_log_probs, len(log_probs))
                    )

                    token_logprobs.append(
                        {
                            "token": self.tokenizer.decode([token_id]),
                            "logprob": token_logprob,
                            "top_logprobs": [
                                {
                                    "token": self.tokenizer.decode([idx.item()]),
                                    "logprob": val.item(),
                                }
                                for val, idx in zip(top_values, top_indices)
                            ],
                        }
                    )

                results.append({"text": text, "logprobs": token_logprobs})

            return results
        else:
            # Decode outputs without log probs
            generated_ids = [
                output_ids[len(input_ids[i]) :]
                for i, output_ids in enumerate(generated_ids)
            ]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return responses

    def _generate_vllm(
        self,
        input_texts: List[str],
        temperature: float,
        max_tokens: int,
        task_type: str,
        top_k_log_probs: int | None = None,
        seed: Optional[int] = None,
    ) -> list[str] | list[dict[str, Any]]:
        """Generate using vLLM backend."""
        from vllm import SamplingParams

        # Get resolved sampling parameters (user > model defaults > general defaults)
        params = self._get_sampling_params(temperature, max_tokens, task_type)

        # Build SamplingParams with resolved values
        # Only include non-None parameters
        sampling_kwargs = {
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "logprobs": top_k_log_probs,
            "seed": seed,
        }

        if params["top_p"] is not None:
            sampling_kwargs["top_p"] = params["top_p"]
        if params["top_k"] is not None:
            sampling_kwargs["top_k"] = params["top_k"]
        if params["min_p"] is not None:
            sampling_kwargs["min_p"] = params["min_p"]
        if params["logprobs"] is not None:
            sampling_kwargs["logprobs"] = params["logprobs"]

        sampling_params = SamplingParams(**sampling_kwargs)

        # Generate
        responses = []
        for batch_start in range(0, len(input_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(input_texts))
            batch_inputs = input_texts[batch_start:batch_end]
            outputs = self.model.generate(batch_inputs, sampling_params)
            if top_k_log_probs is not None:
                for output in outputs:
                    text = output.outputs[0].text

                    token_logprobs = []

                    # output.outputs[0].logprobs is a list of dicts: {token_id: Logprob object}
                    for position_logprobs in output.outputs[0].logprobs:
                        # Extract top-k tokens with their log probs
                        top_tokens = [
                            {
                                "token": logprob_obj.decoded_token,
                                "logprob": logprob_obj.logprob,
                                "rank": logprob_obj.rank,
                            }
                            for token_id, logprob_obj in position_logprobs.items()
                        ]
                        # Sort by rank to maintain top-k order
                        top_tokens.sort(key=lambda x: x["rank"])
                        token_logprobs.append(top_tokens)

                    responses.append({"text": text, "logprobs": token_logprobs})
            else:
                responses.extend([self.tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False) for output in outputs])

        return responses


# ============================================================================
# CLI for standalone testing
# ============================================================================


def add_params():
    """Argument parser for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batched LLM Inference Engine - Standalone Testing"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4",
        help='Model name or path (e.g., "gpt-4", "meta-llama/Llama-2-7b")',
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gpt",
        choices=["hf", "vllm", "gpt"],
        help="Inference backend: hf (HuggingFace), vllm, or gpt (OpenAI API)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data1/ss164/cache/",
        help="Cache directory for model downloads (local models only)",
    )

    # Generation parameters
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0-1.0, higher=more diverse)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--bsz", type=int, default=8, help="Batch size for generation")

    # Hardware configuration
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA even if available"
    )

    # Model-specific flags
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking/reasoning mode (Qwen3, GPT-OSS, CWM)",
    )
    parser.add_argument(
        "--gpt_reasoning",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="GPT-OSS reasoning level",
    )

    # vLLM-specific flags
    parser.add_argument(
        "--vllm_multiproc_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork"],
        help="vLLM multiprocessing method (default: spawn, fixes CUDA errors)",
    )
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
        help="Fraction of GPU memory to use (vLLM only, default: 0.9, range: 0.0-1.0)",
    )

    # Testing
    parser.add_argument(
        "--test", action="store_true", help="Run with test conversations"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = add_params()

    print("=" * 80)
    print("BATCHED LLM INFERENCE ENGINE - STANDALONE TEST")
    print("=" * 80)

    # Initialize inference engine
    inference = BatchedLLMInference(
        model_name=args.model_name,
        backend=args.backend,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        batch_size=args.bsz,
        cache_dir=args.cache_dir,
        num_gpus=args.num_gpus,
        no_cuda=args.no_cuda,
        enable_thinking=args.enable_thinking,
        gpt_reasoning=args.gpt_reasoning,
        vllm_multiproc_method=args.vllm_multiproc_method,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    # Test with sample conversations
    if args.test:
        print("\n" + "=" * 80)
        print("RUNNING TEST GENERATION")
        print("=" * 80)

        test_conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            [{"role": "user", "content": "Explain gravity in one sentence."}],
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is the Pythagorean theorem?"},
            ],
        ]

        print(
            f"\nGenerating responses for {len(test_conversations)} test conversations..."
        )
        responses = inference.generate(test_conversations)

        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)
        for i, (conv, resp) in enumerate(zip(test_conversations, responses)):
            print(f"\nConversation {i+1}:")
            print(f"  User: {conv[-1]['content']}")
            print(f"  Assistant: {resp}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
    else:
        print("\nInference engine initialized successfully!")
        print("Use --test flag to run with sample conversations")
        print(
            "Or import this module and use the BatchedLLMInference class in your code"
        )
