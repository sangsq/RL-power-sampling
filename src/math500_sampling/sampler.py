"""Batched vLLM implementation of autoregressive MCMC power sampling."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Callable

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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Chain":
        return cls(**values)


def mh_log_ratio(
    alpha: float,
    proposed_base: list[float],
    current_base: list[float],
    proposed_q: list[float],
    current_q: list[float],
) -> float:
    """Log MH ratio for suffix resampling with target p(x)^alpha."""
    return alpha * (sum(proposed_base) - sum(current_base)) + sum(current_q) - sum(proposed_q)


def chosen_logprobs(output) -> list[float]:
    return [position[token].logprob for token, position in zip(output.token_ids, output.logprobs)]


def json_rng_state(state):
    return [json_rng_state(value) for value in state] if isinstance(state, tuple) else state


def restore_rng_state(state):
    return tuple(restore_rng_state(value) for value in state) if isinstance(state, list) else state


Checkpoint = Callable[[int, list[Chain], list], None]


class PowerSampler:
    """Grow sequences by blocks, then update random suffixes with MH."""

    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.eos_id = tokenizer.eos_token_id

    def generate(
        self,
        contexts: list[list[int]],
        lengths: list[int],
        temperature: float,
        seeds: list[int],
    ) -> list[tuple[list[int], list[float]]]:
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=length,
                logprobs=0,
                seed=seed,
                detokenize=False,
            )
            for length, seed in zip(lengths, seeds, strict=True)
        ]
        outputs = self.llm.generate(
            [{"prompt_token_ids": context} for context in contexts], params, use_tqdm=False
        )
        return [
            (list(request.outputs[0].token_ids), chosen_logprobs(request.outputs[0]))
            for request in outputs
        ]

    def score(self, sequences: list[list[int]], starts: list[int]) -> list[list[float]]:
        """Score chosen tokens under the untempered base model."""
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False)
        outputs = self.llm.generate(
            [{"prompt_token_ids": sequence} for sequence in sequences], params, use_tqdm=False
        )
        scores = []
        for output, sequence, start in zip(outputs, sequences, starts, strict=True):
            scores.append(
                [
                    output.prompt_logprobs[index][sequence[index]].logprob
                    for index in range(start, len(sequence))
                ]
            )
        return scores

    def standard(
        self,
        prompt_ids: list[list[int]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> list[Chain]:
        generated = self.generate(
            prompt_ids,
            [max_tokens] * len(prompt_ids),
            temperature,
            [seed + index * 10_007 for index in range(len(prompt_ids))],
        )
        chains = []
        for prompt_ids_i, (tokens, logprobs) in zip(prompt_ids, generated, strict=True):
            chains.append(
                Chain(
                    prompt_ids=prompt_ids_i,
                    tokens=tokens,
                    proposal_logprobs=logprobs,
                    sampled_tokens=len(tokens),
                    done=self.eos_id in tokens,
                )
            )
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
        suffix_policy: str = "official_overlap",
        chains: list[Chain] | None = None,
        rng_states: list | None = None,
        start_block: int = 0,
        checkpoint: Checkpoint | None = None,
    ) -> list[Chain]:
        if alpha < 1 or proposal_temperature <= 0 or mcmc_steps < 1:
            raise ValueError("Require alpha >= 1, proposal_temperature > 0, and mcmc_steps >= 1")
        if suffix_policy not in {"full", "official_overlap"}:
            raise ValueError("suffix_policy must be full or official_overlap")
        blocks = math.ceil(max_tokens / block_size)
        if not 0 <= start_block <= blocks:
            raise ValueError("start_block is outside the block schedule")

        chains = chains or [Chain(prompt_ids=list(ids)) for ids in prompt_ids]
        if len(chains) != len(prompt_ids):
            raise ValueError("Checkpoint chain count does not match prompts")
        rngs = [random.Random(seed + index * 10_007) for index in range(len(prompt_ids))]
        if rng_states is not None:
            for rng, state in zip(rngs, rng_states, strict=True):
                rng.setstate(restore_rng_state(state))

        for block in range(start_block, blocks):
            active = [
                index
                for index, chain in enumerate(chains)
                if not chain.done and len(chain.tokens) < max_tokens
            ]
            if not active:
                if checkpoint:
                    checkpoint(blocks, chains, [json_rng_state(rng.getstate()) for rng in rngs])
                break

            lengths = [min(block_size, max_tokens - len(chains[i].tokens)) for i in active]
            contexts = [chains[i].prompt_ids + chains[i].tokens for i in active]
            extensions = self.generate(
                contexts,
                lengths,
                proposal_temperature,
                [rngs[i].randrange(2**31) for i in active],
            )
            for index, (tokens, q_logprobs) in zip(active, extensions, strict=True):
                chains[index].tokens.extend(tokens)
                chains[index].proposal_logprobs.extend(q_logprobs)
                chains[index].sampled_tokens += len(tokens)
            base_scores = self.score(
                [chains[i].prompt_ids + chains[i].tokens for i in active],
                [len(chains[i].prompt_ids) for i in active],
            )
            for index, scores in zip(active, base_scores, strict=True):
                chains[index].base_logprobs = scores

            for _ in range(mcmc_steps):
                active_step = [i for i in active if not chains[i].done and chains[i].tokens]
                if not active_step:
                    break
                cuts = [rngs[i].randrange(len(chains[i].tokens)) for i in active_step]
                proposals = self.generate(
                    [chains[i].prompt_ids + chains[i].tokens[:cut] for i, cut in zip(active_step, cuts)],
                    [len(chains[i].tokens) - cut for i, cut in zip(active_step, cuts)],
                    proposal_temperature,
                    [rngs[i].randrange(2**31) for i in active_step],
                )
                proposed_tokens = [
                    chains[i].tokens[:cut] + generated[0]
                    for i, cut, generated in zip(active_step, cuts, proposals, strict=True)
                ]
                proposed_q = [
                    chains[i].proposal_logprobs[:cut] + generated[1]
                    for i, cut, generated in zip(active_step, cuts, proposals, strict=True)
                ]
                proposed_base = self.score(
                    [chains[i].prompt_ids + tokens for i, tokens in zip(active_step, proposed_tokens)],
                    [len(chains[i].prompt_ids) for i in active_step],
                )

                for index, cut, tokens, q_scores, p_scores, generated in zip(
                    active_step,
                    cuts,
                    proposed_tokens,
                    proposed_q,
                    proposed_base,
                    proposals,
                    strict=True,
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
                    if alpha == proposal_temperature == 1.0:
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

            if checkpoint:
                checkpoint(block + 1, chains, [json_rng_state(rng.getstate()) for rng in rngs])
        return chains
