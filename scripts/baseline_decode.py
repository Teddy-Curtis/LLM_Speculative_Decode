import argparse

import torch

from common import (
    DEFAULT_TARGET_MODEL,
    add_sampling_args,
    encode_prompt,
    load_model,
    load_tokenizer,
    probs_from_logits,
    resolve_device,
    sample_from_probs,
    set_seed,
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
):
    generated = input_ids.clone()
    past_key_values = None

    for _ in range(max_new_tokens):
        model_inputs = generated if past_key_values is None else generated[:, -1:]
        outputs = model(
            input_ids=model_inputs,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_probs = probs_from_logits(outputs.logits[:, -1, :], temperature, top_k)
        next_token = sample_from_probs(next_token_probs)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline autoregressive decoding.")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None)
    add_sampling_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, device)
    input_ids = encode_prompt(tokenizer, args.prompt, device)

    start = timed_call_start(device)
    generated = autoregressive_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
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


if __name__ == "__main__":
    main()

