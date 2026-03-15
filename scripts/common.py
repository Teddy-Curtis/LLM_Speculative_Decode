import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DRAFT_MODEL = "distilgpt2"
DEFAULT_TARGET_MODEL = "gpt2"


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared sampling flags used by all CLI entry points."""
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=7)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch so benchmark runs are easier to compare."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick an explicit device if provided, otherwise prefer CUDA when available."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(model_name: str):
    """
    Load a tokenizer and ensure it has a usable padding token.

    GPT-style tokenizers often omit a dedicated pad token. For this project we
    use the EOS token as padding so batching or future prompt padding does not
    immediately fail.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: torch.device):
    """Load a causal LM, move it onto the requested device, and switch to eval mode."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Tokenize a single prompt string and move the resulting IDs onto the model device."""
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"].to(device)


def probs_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Convert logits into a probability distribution with optional temperature and top-k sampling.

    Example:
        logits shape: [1, vocab_size]
        return shape: [1, vocab_size]

    If `top_k=50`, all but the 50 highest-logit tokens are masked out before
    the softmax. This matches the usual "sample only from the top candidates"
    decoding rule.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    return torch.softmax(logits, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token ID from a probability distribution."""
    return torch.multinomial(probs, num_samples=1)


def select_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    greedy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert logits into both a probability distribution and a chosen next token.

    In sampling mode we apply temperature/top-k and sample from the resulting
    distribution. In greedy mode we still materialize the probability
    distribution for acceptance-ratio calculations, but we choose the argmax
    token deterministically.
    """
    probs = probs_from_logits(logits, temperature, top_k)
    if greedy:
        token = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        token = sample_from_probs(probs)
    return probs, token


def synchronize_if_needed(device: torch.device) -> None:
    """
    Force CUDA work to finish before timing.

    PyTorch GPU kernels launch asynchronously. Without synchronization, a timing
    measurement can stop before the GPU has actually finished the work we want
    to measure.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TraceRecorder:
    """
    Collect a per-token timeline for later visualization.

    Each recorded event represents a token that has become part of the committed
    output sequence. For baseline decoding that is just "one model step
    completed". For speculative decoding it means "this token is now accepted
    into the final answer", which may happen after a block verification pass.
    """

    def __init__(self, tokenizer, prompt: str, metadata: dict):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.metadata = metadata
        self.events = []
        self.response = ""

    def record(self, token_id: int, elapsed_s: float, status: str) -> None:
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        self.events.append(
            {
                "token_id": int(token_id),
                "token_text": token_text,
                "elapsed_s": float(elapsed_s),
                "status": status,
            }
        )

    def set_response(self, response: str) -> None:
        self.response = response

    def write(self, output_path: str) -> None:
        payload = {
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
            "events": self.events,
        }
        write_json(payload, output_path)


def timed_call_start(device: torch.device) -> float:
    synchronize_if_needed(device)
    return time.perf_counter()


def timed_call_end(device: torch.device, start_time: float) -> float:
    synchronize_if_needed(device)
    return time.perf_counter() - start_time


def prime_model_cache(model, input_ids: torch.Tensor):
    """
    Run the full prompt once to initialize the KV cache and return the next-token logits.

    Example:
        input_ids: [prompt tokens]
        output logits: distribution for the next token after the prompt
        output cache: cached keys/values representing the entire prompt

    This is the expensive "read the whole prefix" step. After this point we try
    to advance the cache one token at a time instead of replaying the full prompt.
    """
    outputs = model(input_ids=input_ids, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


def advance_model_cache(model, next_input_ids: torch.Tensor, past_key_values):
    """
    Advance an existing KV cache by exactly the provided token(s).

    In our scripts this is usually called with shape `[1, 1]`, meaning:
    "the prefix is already cached; now update that cache as if we appended this
    one new token."
    """
    outputs = model(
        input_ids=next_input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.logits[:, -1, :], outputs.past_key_values


def trim_past_key_values(past_key_values, sequence_length: int):
    """
    Return a copy of the cache cropped back to a shorter prefix length.

    This matters in speculative decoding when a proposed draft token is rejected.
    At that point we have a cache representing:

        prefix + accepted draft tokens + rejected token + maybe more draft tokens

    but we only want:

        prefix + accepted draft tokens

    In newer `transformers` versions the cache is an object with a `.crop()`
    method, so we preserve that object type. The tuple fallback is kept for
    compatibility with older cache formats.
    """
    if hasattr(past_key_values, "crop") and hasattr(past_key_values, "get_seq_length"):
        trimmed_cache = copy.deepcopy(past_key_values)
        trimmed_cache.crop(sequence_length)
        return trimmed_cache

    trimmed = []
    for layer in past_key_values:
        trimmed_layer = []
        for tensor in layer:
            if tensor is None:
                trimmed_layer.append(None)
            elif tensor.dim() >= 3:
                trimmed_layer.append(tensor[..., :sequence_length, :])
            else:
                trimmed_layer.append(tensor)
        trimmed.append(tuple(trimmed_layer))
    return tuple(trimmed)


def write_json(payload, output_path: str) -> None:
    """Persist benchmark results in a stable JSON format for later comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
