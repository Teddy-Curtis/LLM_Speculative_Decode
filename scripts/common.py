import argparse
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"].to(device)


def probs_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    return torch.softmax(logits, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call_start(device: torch.device) -> float:
    synchronize_if_needed(device)
    return time.perf_counter()


def timed_call_end(device: torch.device, start_time: float) -> float:
    synchronize_if_needed(device)
    return time.perf_counter() - start_time


def prime_model_cache(model, input_ids: torch.Tensor):
    outputs = model(input_ids=input_ids, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


def advance_model_cache(model, next_input_ids: torch.Tensor, past_key_values):
    outputs = model(
        input_ids=next_input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.logits[:, -1, :], outputs.past_key_values


def trim_past_key_values(past_key_values, sequence_length: int):
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
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
