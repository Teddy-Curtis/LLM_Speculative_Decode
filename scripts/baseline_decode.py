import argparse
import time

import torch

from common import (
    DEFAULT_TARGET_MODEL,
    TraceRecorder,
    advance_model_cache,
    add_sampling_args,
    encode_prompt,
    load_model,
    load_tokenizer,
    prime_model_cache,
    resolve_device,
    select_next_token,
    set_seed,
    synchronize_if_needed,
    timed_call_end,
    timed_call_start,
)


@torch.no_grad()
def autoregressive_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    greedy: bool,
    eos_token_id: int | None = None,
    trace_recorder=None,
    trace_start_time: float | None = None,
    device: torch.device | None = None,
):
    """
    Standard left-to-right decoding using one model and its KV cache.

    Example flow:
    1. Read the full prompt once to build the cache.
    2. Sample the next token from the current logits.
    3. Feed only that token back into the model with the cached prefix.
    4. Repeat until `max_new_tokens` have been produced.

    This is the baseline we compare speculative decoding against.
    """
    generated = input_ids.clone()
    logits, past_key_values = prime_model_cache(model, generated)

    for _ in range(max_new_tokens):
        # `logits` always represents "what should come next after the current prefix".
        _, next_token = select_next_token(logits, temperature, top_k, greedy)
        generated = torch.cat([generated, next_token], dim=1)
        logits, past_key_values = advance_model_cache(model, next_token, past_key_values)
        if trace_recorder is not None and trace_start_time is not None and device is not None:
            synchronize_if_needed(device)
            trace_recorder.record(
                token_id=next_token.item(),
                elapsed_s=time.perf_counter() - trace_start_time,
                status="emitted",
            )
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline autoregressive decoding.")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trace-output", default=None)
    add_sampling_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device)
    input_ids = encode_prompt(tokenizer, args.prompt, device)
    trace_recorder = None
    if args.trace_output:
        trace_recorder = TraceRecorder(
            tokenizer=tokenizer,
            prompt=args.prompt,
            metadata={
                "method": "baseline",
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "greedy": args.greedy,
            },
        )

    start = timed_call_start(device)
    # We time only the actual generation call, not model loading or tokenization.
    generated = autoregressive_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
        eos_token_id=tokenizer.eos_token_id,
        trace_recorder=trace_recorder,
        trace_start_time=start,
        device=device,
    )
    latency = timed_call_end(device, start)

    completion = generated[:, input_ids.shape[1] :]
    text = tokenizer.decode(completion[0], skip_special_tokens=True)
    tokens_per_second = completion.shape[1] / latency if latency > 0 else float("inf")

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"generated_tokens={completion.shape[1]}")
    print(f"latency_s={latency:.4f}")
    print(f"tokens_per_second={tokens_per_second:.4f}")
    print("completion:")
    print(text)
    if trace_recorder is not None:
        trace_recorder.write(args.trace_output)
        print(f"trace_output={args.trace_output}")


if __name__ == "__main__":
    main()
