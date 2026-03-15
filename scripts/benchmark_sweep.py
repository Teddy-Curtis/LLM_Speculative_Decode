import argparse

from benchmark import DEFAULT_PROMPTS, summarize
from common import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    encode_prompt,
    load_model,
    load_tokenizer,
    resolve_device,
    set_seed,
    write_json,
)
from benchmark import run_baseline, run_speculative


def build_parser():
    parser = argparse.ArgumentParser(description="Sweep speculative draft block sizes.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-block-sizes", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--output", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
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
    encoded_prompts = []
    for prompt in prompts:
        input_ids = encode_prompt(tokenizer, prompt, device)
        encoded_prompts.append((prompt, input_ids))
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

    baseline_tps = summarize(baseline_results, "tokens_per_second")
    baseline_latency = summarize(baseline_results, "latency_s")

    sweep_results = []
    print(f"device={device}")
    print(f"draft_model={args.draft_model}")
    print(f"target_model={args.target_model}")
    print(f"num_prompts={len(prompts)}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print()
    print("baseline:")
    print(f"  avg_tokens_per_second={baseline_tps:.4f}")
    print(f"  avg_latency_s={baseline_latency:.4f}")
    print()
    print("sweep:")

    for block_size in args.draft_block_sizes:
        speculative_results = []
        for _, input_ids in encoded_prompts:
            speculative_results.append(
                run_speculative(
                    draft_model=draft_model,
                    target_model=target_model,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    num_draft_tokens=block_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                )
            )

        speculative_tps = summarize(speculative_results, "tokens_per_second")
        speculative_latency = summarize(speculative_results, "latency_s")
        acceptance_rate = summarize(speculative_results, "acceptance_rate")
        speedup = speculative_tps / baseline_tps if baseline_tps > 0 else None

        result = {
            "draft_block_size": block_size,
            "avg_tokens_per_second": speculative_tps,
            "avg_latency_s": speculative_latency,
            "avg_acceptance_rate": acceptance_rate,
            "speedup_vs_baseline": speedup,
        }
        sweep_results.append(result)

        print(f"  block_size={block_size}")
        print(f"    avg_tokens_per_second={speculative_tps:.4f}")
        print(f"    avg_latency_s={speculative_latency:.4f}")
        print(f"    avg_acceptance_rate={acceptance_rate:.4f}")
        if speedup is not None:
            print(f"    speedup_vs_baseline={speedup:.4f}x")

    if args.output:
        write_json(
            {
                "device": str(device),
                "draft_model": args.draft_model,
                "target_model": args.target_model,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "prompts": prompts,
                "baseline": {
                    "per_prompt": baseline_results,
                    "avg_tokens_per_second": baseline_tps,
                    "avg_latency_s": baseline_latency,
                },
                "sweep": sweep_results,
            },
            args.output,
        )
        print()
        print(f"saved_results={args.output}")


if __name__ == "__main__":
    main()
