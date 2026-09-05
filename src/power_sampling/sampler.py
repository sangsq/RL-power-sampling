"""Batched vLLM implementation of autoregressive MCMC power sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from vllm import SamplingParams


@dataclass
class Chain:
    prompt_ids: list[int]
    tokens: list[int] = field(default_factory=list)
    base_logprobs: list[float] = field(default_factory=list)
    proposal_logprobs: list[float] = field(default_factory=list)
    attempts: int = 0
    accepts: int = 0
    log_ratios: list[float] = field(default_factory=list)
    sampled_tokens: int = 0
    done: bool = False

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / self.attempts if self.attempts else 0.0


def mh_log_ratio(
    alpha: float,
    proposed_base: list[float],
    current_base: list[float],
    proposed_q: list[float],
    current_q: list[float],
) -> float:
    """Log MH ratio for a resampled suffix targeting p(x)^alpha."""
    return alpha * (sum(proposed_base) - sum(current_base)) + sum(current_q) - sum(proposed_q)


def chosen_logprobs(output) -> list[float]:
    token_ids = output.token_ids
    return [position[token].logprob for token, position in zip(token_ids, output.logprobs)]


class PowerSampler:
    """Sequential block sampler using low-temperature suffix proposals."""

    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id

    def generate(
        self,
        contexts: list[list[int]],
        lengths: list[int],
        temperature: float,
        seeds: list[int],
    ) -> list[tuple[list[int], list[float], str | None]]:
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=length,
                logprobs=0,
                seed=seed,
                detokenize=False,
            )
            for length, seed in zip(lengths, seeds)
        ]
        prompts = [{"prompt_token_ids": context} for context in contexts]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [
            (list(item.outputs[0].token_ids), chosen_logprobs(item.outputs[0]), item.outputs[0].finish_reason)
            for item in outputs
        ]

    def score(self, sequences: list[list[int]], starts: list[int]) -> list[list[float]]:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=0,
            detokenize=False,
        )
        prompts = [{"prompt_token_ids": sequence} for sequence in sequences]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        scores = []
        for output, sequence, start in zip(outputs, sequences, starts):
            positions = output.prompt_logprobs
            scores.append(
                [positions[index][sequence[index]].logprob for index in range(start, len(sequence))]
            )
        return scores

    def standard(
        self,
        prompt_ids: list[list[int]],
        max_tokens: int,
        temperature: float,
        seed: int,
        independent_seeds: bool = True,
    ) -> list[Chain]:
        generated = self.generate(
            prompt_ids,
            [max_tokens] * len(prompt_ids),
            temperature,
            (
                [seed + index * 10_007 for index in range(len(prompt_ids))]
                if independent_seeds
                else [seed] * len(prompt_ids)
            ),
        )
        chains = []
        for prompt, (tokens, q_logprobs, _) in zip(prompt_ids, generated):
            chain = Chain(prompt_ids=prompt, tokens=tokens, proposal_logprobs=q_logprobs)
            chain.sampled_tokens = len(tokens)
            chains.append(chain)
        return chains

    def power(
        self,
        prompt_ids: list[list[int]],
        *,
        alpha: float,
        proposal_temperature: float,
        mcmc_steps: int,
        block_size: int,
        max_tokens: int,
        seed: int,
        suffix_policy: str = "full",
    ) -> list[Chain]:
        if alpha < 1 or proposal_temperature <= 0 or mcmc_steps < 1:
            raise ValueError("Require alpha >= 1, proposal_temperature > 0, and mcmc_steps >= 1")
        if suffix_policy not in {"full", "official_overlap"}:
            raise ValueError("suffix_policy must be full or official_overlap")
        rngs = [random.Random(seed + index * 10_007) for index in range(len(prompt_ids))]
        chains = [Chain(prompt_ids=list(prompt)) for prompt in prompt_ids]

        for block in range(math.ceil(max_tokens / block_size)):
            active = [index for index, chain in enumerate(chains) if not chain.done and len(chain.tokens) < max_tokens]
            if not active:
                break
            lengths = [min(block_size, max_tokens - len(chains[index].tokens)) for index in active]
            contexts = [chains[index].prompt_ids + chains[index].tokens for index in active]
            extensions = self.generate(
                contexts,
                lengths,
                proposal_temperature,
                [rngs[index].randrange(2**31) for index in active],
            )
            sequences = []
            starts = []
            for index, (tokens, q_logprobs, _) in zip(active, extensions):
                chain = chains[index]
                chain.tokens.extend(tokens)
                chain.proposal_logprobs.extend(q_logprobs)
                chain.sampled_tokens += len(tokens)
                sequences.append(chain.prompt_ids + chain.tokens)
                starts.append(len(chain.prompt_ids))
            base_scores = self.score(sequences, starts)
            for index, scores in zip(active, base_scores):
                chains[index].base_logprobs = scores

            for step in range(mcmc_steps):
                active_step = [index for index in active if not chains[index].done and chains[index].tokens]
                if not active_step:
                    break
                cuts = [rngs[index].randrange(len(chains[index].tokens)) for index in active_step]
                contexts = [
                    chains[index].prompt_ids + chains[index].tokens[:cut]
                    for index, cut in zip(active_step, cuts)
                ]
                lengths = [len(chains[index].tokens) - cut for index, cut in zip(active_step, cuts)]
                proposals = self.generate(
                    contexts,
                    lengths,
                    proposal_temperature,
                    [rngs[index].randrange(2**31) for index in active_step],
                )
                proposed_tokens = [
                    chains[index].tokens[:cut] + generated[0]
                    for index, cut, generated in zip(active_step, cuts, proposals)
                ]
                proposed_q = [
                    chains[index].proposal_logprobs[:cut] + generated[1]
                    for index, cut, generated in zip(active_step, cuts, proposals)
                ]
                proposed_base = self.score(
                    [chains[index].prompt_ids + tokens for index, tokens in zip(active_step, proposed_tokens)],
                    [len(chains[index].prompt_ids) for index in active_step],
                )

                for index, cut, tokens, q_scores, p_scores, generated in zip(
                    active_step, cuts, proposed_tokens, proposed_q, proposed_base, proposals
                ):
                    chain = chains[index]
                    end = len(tokens) if suffix_policy == "official_overlap" else None
                    log_ratio = mh_log_ratio(
                        alpha,
                        p_scores[cut:end],
                        chain.base_logprobs[cut:end],
                        q_scores[cut:end],
                        chain.proposal_logprobs[cut:end],
                    )
                    # Algebraically exact; avoids accumulated BF16 differences
                    # between vLLM generation and prompt-scoring code paths.
                    if alpha == 1.0 and proposal_temperature == 1.0:
                        log_ratio = 0.0
                    chain.attempts += 1
                    chain.log_ratios.append(log_ratio)
                    chain.sampled_tokens += len(generated[0])
                    if math.log(max(rngs[index].random(), 1e-300)) < min(0.0, log_ratio):
                        chain.tokens = tokens
                        chain.proposal_logprobs = q_scores
                        chain.base_logprobs = p_scores
                        chain.accepts += 1
                    if self.eos_id in chain.tokens:
                        eos = chain.tokens.index(self.eos_id) + 1
                        chain.tokens = chain.tokens[:eos]
                        chain.proposal_logprobs = chain.proposal_logprobs[:eos]
                        chain.base_logprobs = chain.base_logprobs[:eos]
                        chain.done = True
        return chains
