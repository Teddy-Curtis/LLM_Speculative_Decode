import argparse

import time

import torch

from common import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    TraceRecorder,
    advance_model_cache,
    add_sampling_args,
    encode_prompt,
    load_model,
    load_tokenizer,
    prime_model_cache,
    probs_from_logits,
    resolve_device,
    sample_from_probs,
    select_next_token,
    set_seed,
    synchronize_if_needed,
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
    greedy: bool,
):
    """
    Ask the draft model to propose a short block of future tokens.

    Inputs:
    - `draft_past_key_values`: cache for the already accepted prefix
    - `next_logits`: distribution for the next token after that prefix

    Returns:
    - `stacked_tokens`: proposed tokens, shape `[1, block_size]`
    - `stacked_probs`: draft probabilities for each proposal step
    - updated cache/logits after rolling the draft model forward through the block

    Example:
        If the accepted prefix is "The capital of France is", this function might
        propose `[" Paris", ",", " and", " it"]` one token at a time.
    """
    proposal_tokens = []
    proposal_probs = []
    past_key_values = draft_past_key_values
    logits = next_logits

    for _ in range(num_draft_tokens):
        # Sample from the draft model's current next-token distribution.
        probs, token = select_next_token(logits, temperature, top_k, greedy)
        proposal_tokens.append(token)
        proposal_probs.append(probs)
        # Roll the draft cache forward so the next loop iteration sees a longer prefix.
        logits, past_key_values = advance_model_cache(draft_model, token, past_key_values)

    stacked_tokens = torch.cat(proposal_tokens, dim=1)
    stacked_probs = torch.stack(proposal_probs, dim=1)
    return stacked_tokens, stacked_probs, past_key_values, logits


def sample_remainder(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """
    Sample from the "correction" distribution used after a rejection.

    In speculative decoding, if the target rejects a proposed draft token, we do
    not simply sample from the target distribution again. Instead we sample from
    the positive part of:

        target_probs - draft_probs

    This preserves the correct target-model distribution overall.
    """
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
    greedy: bool,
    eos_token_id: int | None = None,
    trace_recorder=None,
    trace_start_time: float | None = None,
    device: torch.device | None = None,
):
    """
    Generate tokens with manual speculative decoding.

    High-level algorithm:
    1. Prime both draft and target models on the same prompt.
    2. Let the draft model propose a short block of `num_draft_tokens`.
    3. Run the target model over that block in one go using the cached prefix.
    4. For each proposed token, accept or reject it using the speculative rule.
    5. If all proposals are accepted, sample one extra "bonus" token from the target.
    6. If a proposal is rejected, crop the target cache back to the accepted prefix,
       append the correction token, and update the draft model on the new sequence.

    Small example:
        Prompt tokens: [A, B]
        Draft proposes: [C, D, E]
        Target verifies:
        - accept C
        - reject D
        Then the final continuation for this block becomes:
            [C, correction_token]
        and the remaining proposals after D are discarded.
    """
    generated = input_ids.clone()
    drafted_tokens = 0
    accepted_tokens = 0
    # Both models start from the exact same prompt prefix.
    draft_next_logits, draft_past_key_values = prime_model_cache(draft_model, generated)
    target_next_logits, target_past_key_values = prime_model_cache(target_model, generated)

    while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
        remaining = max_new_tokens - (generated.shape[1] - input_ids.shape[1])
        current_block = min(num_draft_tokens, remaining)
        prefix_length = generated.shape[1]

        # Step 1: the draft model cheaply proposes several future tokens.
        draft_tokens, draft_probs, proposed_draft_cache, proposed_draft_logits = generate_draft_tokens(
            draft_model=draft_model,
            draft_past_key_values=draft_past_key_values,
            next_logits=draft_next_logits,
            num_draft_tokens=current_block,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        drafted_tokens += current_block

        # Step 2: the target model verifies the entire proposed block against the
        # current accepted prefix using its own KV cache.
        target_outputs = target_model(
            input_ids=draft_tokens,
            past_key_values=target_past_key_values,
            use_cache=True,
        )
        proposed_target_cache = target_outputs.past_key_values
        # `target_outputs.logits[:, j, :]` is the distribution *after* consuming
        # `draft_tokens[:, j]`. For verification we instead need:
        #
        # - step 0: distribution after the accepted prefix and before draft token 0
        # - step 1: distribution after draft token 0 and before draft token 1
        # - ...
        #
        # So the correct verification logits are:
        #   [target_next_logits, target_outputs.logits[:, :-1, :]]
        verification_logits = torch.cat(
            [target_next_logits.unsqueeze(1), target_outputs.logits[:, :-1, :]],
            dim=1,
        )
        target_probs = probs_from_logits(verification_logits, temperature, top_k)
        bonus_logits = target_outputs.logits[:, -1, :]

        all_accepted = True
        accepted_in_block = 0
        for step in range(current_block):
            proposed_token = draft_tokens[:, step]
            q_probs = target_probs[:, step, :]
            p_probs = draft_probs[:, step, :]
            target_greedy_token = torch.argmax(q_probs, dim=-1, keepdim=True)

            # The acceptance probability is min(1, q(x) / p(x)) where:
            # - q is the target-model distribution
            # - p is the draft-model distribution
            q_token_prob = q_probs.gather(1, proposed_token.unsqueeze(1)).squeeze(1)
            p_token_prob = p_probs.gather(1, proposed_token.unsqueeze(1)).squeeze(1)
            accept_prob = torch.minimum(
                torch.ones_like(q_token_prob),
                q_token_prob / torch.clamp(p_token_prob, min=1e-12),
            )

            if greedy:
                accept_token = torch.equal(proposed_token.unsqueeze(1), target_greedy_token)
            else:
                accept_token = torch.rand(1, device=generated.device).item() <= accept_prob.item()

            if accept_token:
                # Accepted draft tokens become part of the final output exactly as proposed.
                generated = torch.cat([generated, proposed_token.unsqueeze(1)], dim=1)
                accepted_tokens += 1
                accepted_in_block += 1
                if trace_recorder is not None and trace_start_time is not None and device is not None:
                    synchronize_if_needed(device)
                    trace_recorder.record(
                        token_id=proposed_token.item(),
                        elapsed_s=time.perf_counter() - trace_start_time,
                        status="accepted_draft",
                    )
                if eos_token_id is not None and proposed_token.item() == eos_token_id:
                    all_accepted = False
                    break
            else:
                # On rejection we:
                # 1. crop the verified target cache back to the accepted prefix,
                # 2. sample a correction token from the remainder distribution,
                # 3. append that correction token,
                # 4. crop the draft cache to the same accepted prefix and advance it
                #    with the correction token instead of re-reading the full sequence.
                if greedy:
                    # In greedy mode we force speculative decoding to follow the
                    # target model's greedy path exactly so visual comparisons can
                    # show the same text appearing at different times.
                    correction_token = target_greedy_token
                else:
                    correction_token = sample_remainder(q_probs, p_probs)
                generated = torch.cat([generated, correction_token], dim=1)
                all_accepted = False
                if trace_recorder is not None and trace_start_time is not None and device is not None:
                    synchronize_if_needed(device)
                    trace_recorder.record(
                        token_id=correction_token.item(),
                        elapsed_s=time.perf_counter() - trace_start_time,
                        status="target_correction",
                    )
                if eos_token_id is not None and correction_token.item() == eos_token_id:
                    return (
                        generated[:, : input_ids.shape[1] + max_new_tokens],
                        drafted_tokens,
                        accepted_tokens,
                        generated.shape[1] - input_ids.shape[1],
                        accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0,
                    )
                accepted_prefix_cache = trim_past_key_values(
                    proposed_target_cache,
                    prefix_length + accepted_in_block,
                )
                target_next_logits, target_past_key_values = advance_model_cache(
                    target_model,
                    correction_token,
                    accepted_prefix_cache,
                )
                accepted_draft_cache = trim_past_key_values(
                    proposed_draft_cache,
                    prefix_length + accepted_in_block,
                )
                draft_next_logits, draft_past_key_values = advance_model_cache(
                    draft_model,
                    correction_token,
                    accepted_draft_cache,
                )
                break

        if all_accepted and generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # If the target accepted the entire block, we get one extra token "for free"
            # from the target model's final verification logits.
            target_past_key_values = proposed_target_cache
            target_next_logits = bonus_logits

            _, bonus_token = select_next_token(target_next_logits, temperature, top_k, greedy)
            generated = torch.cat([generated, bonus_token], dim=1)
            if trace_recorder is not None and trace_start_time is not None and device is not None:
                synchronize_if_needed(device)
                trace_recorder.record(
                    token_id=bonus_token.item(),
                    elapsed_s=time.perf_counter() - trace_start_time,
                    status="target_bonus",
                )
            if eos_token_id is not None and bonus_token.item() == eos_token_id:
                break
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
            # If we accepted the full block but have already reached `max_new_tokens`,
            # keep the advanced caches without sampling the bonus token.
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
    parser.add_argument("--trace-output", default=None)
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
    trace_recorder = None
    if args.trace_output:
        trace_recorder = TraceRecorder(
            tokenizer=tokenizer,
            prompt=args.prompt,
            metadata={
                "method": "speculative",
                "draft_model": args.draft_model,
                "target_model": args.target_model,
                "max_new_tokens": args.max_new_tokens,
                "num_draft_tokens": args.num_draft_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "greedy": args.greedy,
            },
        )

    start = timed_call_start(device)
    # Only the decoding loop is timed so the benchmark is not dominated by setup.
    generated, drafted_tokens, accepted_tokens, new_tokens, acceptance_rate = speculative_generate(
        draft_model=draft_model,
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft_tokens,
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
    if trace_recorder is not None:
        trace_recorder.set_response(text)
        trace_recorder.write(args.trace_output)
        print(f"trace_output={args.trace_output}")


if __name__ == "__main__":
    main()
