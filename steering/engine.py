"""
Steering Engine — Single-model inference with per-agent concept steering.

Replaces the multi-model committee with a single Gemma-2-9B-it loaded in 4-bit,
using RFM concept vectors from neural_controllers to steer per agent personality.
"""

import asyncio
import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Add neural_controllers to path
NC_PATH = Path(__file__).parent.parent / "neural_controllers"
sys.path.insert(0, str(NC_PATH))

DIRECTIONS_DIR = Path(__file__).parent / "directions"
MODEL_ID = "Qwen/Qwen3-4B"


class SteeringEngine:
    """Single Gemma-2-9B model with per-agent RFM concept steering."""

    def __init__(self, device="cuda", load_in_4bit=True):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.controller = None
        self.concept_directions: Dict[str, dict] = {}  # concept_name -> directions dict
        self._loaded = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._composite_cache: Dict[str, dict] = {}  # "agent|role" -> composite directions
        self._vram_reservation_id = None  # GPU queue v2 reservation

        logger.info(f"SteeringEngine initialized (model={MODEL_ID}, 4bit={load_in_4bit})")

    def load_model(self):
        """Load Gemma-2-9B-it in 4-bit quantization."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading {MODEL_ID} in 4-bit...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            token=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Initialize neural controller
        from neural_controllers import NeuralController
        self.controller = NeuralController(
            self.model,
            self.tokenizer,
            rfm_iters=8,
            batch_size=2,
            n_components=5,
            control_method='rfm',
        )

        self._loaded = True
        
        # Register VRAM reservation with GPU queue v2
        # Use nvidia-smi for actual VRAM (PyTorch undercounts by ~2x due to CUDA context)
        try:
            sys.path.insert(0, "/home/andus/.openclaw/workspace/gpu-queue")
            from queue_manager_v2 import gpu_reserve
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            vram_used = 0
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) == 2 and int(parts[0].strip()) == os.getpid():
                    vram_used = int(parts[1].strip())
                    break
            if vram_used == 0:
                vram_used = int(torch.cuda.memory_allocated() / 1e6) + 500
            # Add 200 MB headroom for inference spikes
            vram_budget = vram_used + 200
            self._vram_reservation_id = gpu_reserve("steering_engine", vram_mb=vram_budget, pid=os.getpid())
            logger.info(f"Registered VRAM reservation: {vram_budget}MB actual (nvidia-smi: {vram_used}MB, id={self._vram_reservation_id})")
        except Exception as e:
            logger.warning(f"Could not register VRAM reservation: {e}")
        
        logger.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def load_concept(self, concept_name: str):
        """Load a pre-trained concept direction from disk."""
        if concept_name in self.concept_directions:
            return

        direction_file = DIRECTIONS_DIR / f"rfm_{concept_name}_qwen3_4b.pkl"
        if not direction_file.exists():
            logger.warning(f"No direction file for concept '{concept_name}': {direction_file}")
            return

        import pickle
        with open(direction_file, 'rb') as f:
            directions = pickle.load(f)
        self.concept_directions[concept_name] = directions
        logger.info(f"Loaded concept direction: {concept_name}")

    def load_all_concepts(self):
        """Load all available concept directions."""
        if not DIRECTIONS_DIR.exists():
            logger.warning(f"Directions directory not found: {DIRECTIONS_DIR}")
            return

        for pkl_file in DIRECTIONS_DIR.glob("rfm_*_qwen3_4b.pkl"):
            concept = pkl_file.stem.replace("rfm_", "").replace("_qwen3_4b", "")
            self.load_concept(concept)

        logger.info(f"Loaded {len(self.concept_directions)} concept directions")

    def generate(
        self,
        prompt: str,
        agent_name: str = "",
        agent_concepts: Optional[Dict[str, float]] = None,
        pipeline_role: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text with per-agent, per-pipeline concept steering.

        Args:
            prompt: The input prompt
            agent_name: Name of the agent (for logging)
            agent_concepts: Dict of {concept_name: coefficient}. If None and agent_name
                           is provided, auto-resolves from agent profiles.
            pipeline_role: Pipeline role (social/spatial/temporal/emotional/memory/dialogue/judge).
                          When provided with agent_name, applies role-specific steering.
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        if not self._loaded:
            self.load_model()

        # Auto-resolve steering from agent name + pipeline role
        if agent_concepts is None and agent_name:
            if pipeline_role:
                from .agent_concepts import get_pipeline_steering
                agent_concepts = get_pipeline_steering(agent_name, pipeline_role)
                logger.debug(f"Steering {agent_name}/{pipeline_role}: {agent_concepts}")
            else:
                from .agent_concepts import get_steering_config
                agent_concepts = get_steering_config(agent_name)

        # If no concepts to steer, generate normally
        if not agent_concepts:
            return self._generate_plain(prompt, max_new_tokens, temperature)

        # Build cache key from agent + pipeline role
        cache_key = f"{agent_name}|{pipeline_role}" if agent_name else ""

        # Apply multi-concept steering via hook composition
        return self._generate_steered(
            prompt, agent_concepts, max_new_tokens, temperature, cache_key=cache_key
        )

    def _tokenize_chat(self, prompt: str):
        """Tokenize prompt using chat template, return input_ids tensor on device."""
        messages = [{"role": "user", "content": prompt}]
        # Disable Qwen3 thinking mode — we need direct answers, not internal reasoning
        template_kwargs = {"return_tensors": "pt", "add_generation_prompt": True}
        if "Qwen" in MODEL_ID:
            template_kwargs["enable_thinking"] = False
        result = self.tokenizer.apply_chat_template(
            messages, **template_kwargs
        )
        # apply_chat_template may return a BatchEncoding or raw tensor depending on version
        if hasattr(result, 'input_ids'):
            input_ids = result.input_ids.to(self.model.device)
        elif isinstance(result, torch.Tensor):
            input_ids = result.to(self.model.device)
        else:
            input_ids = result["input_ids"].to(self.model.device) if isinstance(result, dict) else torch.tensor(result).unsqueeze(0).to(self.model.device)
        return input_ids

    def _decode_new_tokens(self, input_ids, output_ids) -> str:
        """Decode only the newly generated tokens (strip the input)."""
        import re
        new_ids = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
        # Debug: log raw output before stripping
        if '<think>' in text:
            logger.debug(f"Raw output (first 500 chars): {text[:500]}")
        
        # Strip Qwen3 thinking blocks
        # Handle closed blocks
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        # Handle unclosed blocks (if max_new_tokens hit mid-thought) - remove everything
        if '<think>' in text:
            logger.warning(f"Unclosed <think> block detected — output truncated mid-thought. Increase max_new_tokens.")
            text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        
        result = text.strip()
        if not result:
            logger.warning(f"Empty result after stripping think blocks. Raw length was {len(text)}")
        return result

    def _generate_plain(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate without steering."""
        # Qwen3 needs more tokens for thinking
        if "Qwen" in MODEL_ID and max_new_tokens < 1024:
            max_new_tokens = 2048
        
        input_ids = self._tokenize_chat(prompt)
        outputs = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        return self._decode_new_tokens(input_ids, outputs)

    def _get_composite_directions(self, agent_concepts: Dict[str, float], cache_key: str = "") -> dict:
        """Compute or retrieve cached composite directions for a concept mix."""
        if cache_key and cache_key in self._composite_cache:
            return self._composite_cache[cache_key]

        composite_directions = {}
        for concept_name, coef in agent_concepts.items():
            if concept_name not in self.concept_directions:
                continue
            directions = self.concept_directions[concept_name]
            for layer, direction in directions.items():
                if isinstance(direction, torch.Tensor):
                    d = direction.float()
                else:
                    import numpy as np
                    d = torch.from_numpy(direction).float()

                scaled = d * coef
                if layer in composite_directions:
                    composite_directions[layer] = composite_directions[layer] + scaled
                else:
                    composite_directions[layer] = scaled

        if cache_key and composite_directions:
            self._composite_cache[cache_key] = composite_directions
            logger.debug(f"Cached composite directions for '{cache_key}' ({len(composite_directions)} layers)")

        return composite_directions

    def _generate_steered(
        self,
        prompt: str,
        agent_concepts: Dict[str, float],
        max_new_tokens: int,
        temperature: float,
        cache_key: str = "",
    ) -> str:
        """Generate with multi-concept steering via summed direction hooks."""
        # Qwen3 needs more tokens for thinking
        if "Qwen" in MODEL_ID and max_new_tokens < 1024:
            max_new_tokens = 2048

        from generation_utils import hook_model, clear_hooks

        composite_directions = self._get_composite_directions(agent_concepts, cache_key)

        if not composite_directions:
            return self._generate_plain(prompt, max_new_tokens, temperature)

        # Hook and generate
        layers_to_control = list(composite_directions.keys())
        hooks = hook_model(self.model, composite_directions, layers_to_control, control_coef=1.0)

        try:
            input_ids = self._tokenize_chat(prompt)
            outputs = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            return self._decode_new_tokens(input_ids, outputs)
        finally:
            clear_hooks(hooks)

    async def generate_async(self, **kwargs) -> str:
        """Async wrapper — runs generate() in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: self.generate(**kwargs))

    def unload(self):
        """Free GPU memory and release VRAM reservation."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.controller
            self.model = None
            self.tokenizer = None
            self.controller = None
            self._loaded = False
            torch.cuda.empty_cache()
            
            # Release VRAM reservation
            if self._vram_reservation_id:
                try:
                    sys.path.insert(0, "/home/andus/.openclaw/workspace/gpu-queue")
                    from queue_manager_v2 import gpu_release
                    gpu_release(self._vram_reservation_id)
                    logger.info(f"Released VRAM reservation {self._vram_reservation_id}")
                except Exception as e:
                    logger.warning(f"Could not release VRAM reservation: {e}")
                self._vram_reservation_id = None
            
            logger.info("Model unloaded, VRAM freed")
