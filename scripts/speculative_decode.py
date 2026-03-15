import argparse

import torch

from common import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    advance_model_cache,
    add_sampling_args,
    encode_prompt,
    load_model,
    load_tokenizer,
    prime_model_cache,
    probs_from_logits,
    resolve_device,
    sample_from_probs,
    set_seed,
    timed_call_end,
    timed_call_start,
    trim_past_key_values,
)


@torch.no_grad()
def generate_draft_tokens(
    draft_model,
    draft_past_key_values,
    next_logits: torch.Tensor,
    num_draft_tokens: int,
    temperature: float,
    top_k: int,
):
    proposal_tokens = []
    proposal_probs = []
    past_key_values = draft_past_key_values
    logits = next_logits

    for _ in range(num_draft_tokens):
        probs = probs_from_logits(logits, temperature, top_k)
        token = sample_from_probs(probs)
        proposal_tokens.append(token)
        proposal_probs.append(probs)
        logits, past_key_values = advance_model_cache(draft_model, token, past_key_values)

    stacked_tokens = torch.cat(proposal_tokens, dim=1)
    stacked_probs = torch.stack(proposal_probs, dim=1)
    return stacked_tokens, stacked_probs, past_key_values, logits


def sample_remainder(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    remainder = torch.clamp(target_probs - draft_probs, min=0.0)
    normalizer = remainder.sum(dim=-1, keepdim=True)
    fallback = normalizer.squeeze(-1) <= 0
    if fallback.any():
        remainder = remainder.clone()
        remainder[fallback] = target_probs[fallback]
        normalizer = remainder.sum(dim=-1, keepdim=True)
    remainder = remainder / normalizer
    return sample_from_probs(remainder)


@torch.no_grad()
def speculative_generate(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    num_draft_tokens: int,
    temperature: float,
    top_k: int,
):
    generated = input_ids.clone()
    drafted_tokens = 0
    accepted_tokens = 0
    draft_next_logits, draft_past_key_values = prime_model_cache(draft_model, generated)
    target_next_logits, target_past_key_values = prime_model_cache(target_model, generated)

    while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
        remaining = max_new_tokens - (generated.shape[1] - input_ids.shape[1])
        current_block = min(num_draft_tokens, remaining)
        prefix_length = generated.shape[1]

        draft_tokens, draft_probs, proposed_draft_cache, proposed_draft_logits = generate_draft_tokens(
            draft_model=draft_model,
            draft_past_key_values=draft_past_key_values,
            next_logits=draft_next_logits,
            num_draft_tokens=current_block,
            temperature=temperature,
            top_k=top_k,
        )
        drafted_tokens += current_block

        target_outputs = target_model(
            input_ids=draft_tokens,
            past_key_values=target_past_key_values,
            use_cache=True,
        )
        proposed_target_cache = target_outputs.past_key_values
        target_probs = probs_from_logits(target_outputs.logits, temperature, top_k)
        bonus_logits = target_outputs.logits[:, -1, :]

        all_accepted = True
        accepted_in_block = 0
        for step in range(current_block):
            proposed_token = draft_tokens[:, step]
            q_probs = target_probs[:, step, :]
            p_probs = draft_probs[:, step, :]

            q_token_prob = q_probs.gather(1, proposed_token.unsqueeze(1)).squeeze(1)
            p_token_prob = p_probs.gather(1, proposed_token.unsqueeze(1)).squeeze(1)
            accept_prob = torch.minimum(
                torch.ones_like(q_token_prob),
                q_token_prob / torch.clamp(p_token_prob, min=1e-12),
            )

            if torch.rand(1, device=generated.device).item() <= accept_prob.item():
                generated = torch.cat([generated, proposed_token.unsqueeze(1)], dim=1)
                accepted_tokens += 1
                accepted_in_block += 1
            else:
                correction_token = sample_remainder(q_probs, p_probs)
                generated = torch.cat([generated, correction_token], dim=1)
                all_accepted = False
                accepted_prefix_cache = trim_past_key_values(
                    proposed_target_cache,
                    prefix_length + accepted_in_block,
                )
                target_next_logits, target_past_key_values = advance_model_cache(
                    target_model,
                    correction_token,
                    accepted_prefix_cache,
                )

                refreshed_logits, refreshed_cache = prime_model_cache(draft_model, generated)
                draft_next_logits = refreshed_logits
                draft_past_key_values = refreshed_cache
                break

        if all_accepted and generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            target_past_key_values = proposed_target_cache
            target_next_logits = bonus_logits

            bonus_probs = probs_from_logits(target_next_logits, temperature, top_k)
            bonus_token = sample_from_probs(bonus_probs)
            generated = torch.cat([generated, bonus_token], dim=1)
            target_next_logits, target_past_key_values = advance_model_cache(
                target_model,
                bonus_token,
                target_past_key_values,
            )
            draft_next_logits, draft_past_key_values = advance_model_cache(
                draft_model,
                bonus_token,
                proposed_draft_cache,
            )
        elif all_accepted:
            target_past_key_values = proposed_target_cache
            target_next_logits = bonus_logits
            draft_past_key_values = proposed_draft_cache
            draft_next_logits = proposed_draft_logits

    new_tokens = generated.shape[1] - input_ids.shape[1]
    acceptance_rate = accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0
    return generated[:, : input_ids.shape[1] + max_new_tokens], drafted_tokens, accepted_tokens, new_tokens, acceptance_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual speculative decoding.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--device", default=None)
    add_sampling_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.target_model)
    draft_model = load_model(args.draft_model, device)
    target_model = load_model(args.target_model, device)
    input_ids = encode_prompt(tokenizer, args.prompt, device)

    start = timed_call_start(device)
    generated, drafted_tokens, accepted_tokens, new_tokens, acceptance_rate = speculative_generate(
        draft_model=draft_model,
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    latency = timed_call_end(device, start)

    completion = generated[:, input_ids.shape[1] :]
    text = tokenizer.decode(completion[0], skip_special_tokens=True)
    tokens_per_second = new_tokens / latency if latency > 0 else float("inf")

    print(f"draft_model={args.draft_model}")
    print(f"target_model={args.target_model}")
    print(f"device={device}")
    print(f"generated_tokens={new_tokens}")
    print(f"drafted_tokens={drafted_tokens}")
    print(f"accepted_tokens={accepted_tokens}")
    print(f"acceptance_rate={acceptance_rate:.4f}")
    print(f"latency_s={latency:.4f}")
    print(f"tokens_per_second={tokens_per_second:.4f}")
    print("completion:")
    print(text)


if __name__ == "__main__":
    main()
