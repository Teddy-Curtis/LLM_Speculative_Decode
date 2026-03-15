import argparse
import statistics
from typing import Dict, List

from baseline_decode import autoregressive_generate
from common import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    add_sampling_args,
    encode_prompt,
    load_model,
    load_tokenizer,
    resolve_device,
    set_seed,
    timed_call_end,
    timed_call_start,
    write_json,
)
from speculative_decode import speculative_generate


DEFAULT_PROMPTS = [
    "Explain speculative decoding in simple terms.",
    "Write a short paragraph about why batching can improve throughput.",
    "PyTorch modules are useful because",
]


def run_baseline(model, input_ids, max_new_tokens, temperature, top_k, device):
    """Run one baseline decode and return metrics in a JSON-friendly dict."""
    start = timed_call_start(device)
    generated = autoregressive_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    latency = timed_call_end(device, start)
    new_tokens = generated.shape[1] - input_ids.shape[1]
    return {
        "latency_s": latency,
        "tokens_per_second": new_tokens / latency if latency > 0 else float("inf"),
        "generated_tokens": new_tokens,
    }


def run_speculative(
    draft_model,
    target_model,
    input_ids,
    max_new_tokens,
    num_draft_tokens,
    temperature,
    top_k,
    device,
):
    """Run one speculative decode and return metrics in a JSON-friendly dict."""
    start = timed_call_start(device)
    _, drafted_tokens, accepted_tokens, generated_tokens, acceptance_rate = speculative_generate(
        draft_model=draft_model,
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    latency = timed_call_end(device, start)
    return {
        "latency_s": latency,
        "tokens_per_second": generated_tokens / latency if latency > 0 else float("inf"),
        "generated_tokens": generated_tokens,
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": acceptance_rate,
    }


def summarize(results: List[Dict[str, float]], key: str) -> float:
    """Average one metric across prompts."""
    return statistics.mean(result[key] for result in results)


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark baseline vs speculative decoding.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--output", default=None)
    add_sampling_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    prompts = args.prompts or DEFAULT_PROMPTS
    tokenizer = load_tokenizer(args.target_model)
    draft_model = load_model(args.draft_model, device)
    target_model = load_model(args.target_model, device)

    baseline_results = []
    speculative_results = []

    for prompt in prompts:
        # Each prompt is benchmarked independently so we can average across a
        # small fixed prompt set rather than overfitting to one lucky example.
        input_ids = encode_prompt(tokenizer, prompt, device)
        baseline_results.append(
            run_baseline(
                model=target_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
        )
        speculative_results.append(
            run_speculative(
                draft_model=draft_model,
                target_model=target_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                num_draft_tokens=args.num_draft_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
        )

    baseline_tps = summarize(baseline_results, "tokens_per_second")
    speculative_tps = summarize(speculative_results, "tokens_per_second")
    baseline_latency = summarize(baseline_results, "latency_s")
    speculative_latency = summarize(speculative_results, "latency_s")
    acceptance_rate = summarize(speculative_results, "acceptance_rate")
    speedup = speculative_tps / baseline_tps if baseline_tps > 0 else None

    payload = {
        "device": str(device),
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "draft_block_size": args.num_draft_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "prompts": prompts,
        "baseline": baseline_results,
        "speculative": speculative_results,
        "summary": {
            "avg_baseline_tokens_per_second": baseline_tps,
            "avg_baseline_latency_s": baseline_latency,
            "avg_speculative_tokens_per_second": speculative_tps,
            "avg_speculative_latency_s": speculative_latency,
            "avg_acceptance_rate": acceptance_rate,
            "speedup_vs_baseline": speedup,
        },
    }

    # The printed output is meant for quick human inspection in Colab.
    # The optional JSON file is meant for later comparison across runs.
    print(f"device={device}")
    print(f"draft_model={args.draft_model}")
    print(f"target_model={args.target_model}")
    print(f"num_prompts={len(prompts)}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"draft_block_size={args.num_draft_tokens}")
    print()
    print("baseline:")
    print(f"  avg_tokens_per_second={baseline_tps:.4f}")
    print(f"  avg_latency_s={baseline_latency:.4f}")
    print()
    print("speculative:")
    print(f"  avg_tokens_per_second={speculative_tps:.4f}")
    print(f"  avg_latency_s={speculative_latency:.4f}")
    print(f"  avg_acceptance_rate={acceptance_rate:.4f}")
    if speedup is not None:
        print(f"  speedup_vs_baseline={speedup:.4f}x")

    if args.output:
        write_json(payload, args.output)
        print()
        print(f"saved_results={args.output}")


if __name__ == "__main__":
    main()
