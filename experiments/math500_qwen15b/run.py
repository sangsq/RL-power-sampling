"""Run one resumable MATH500 sampling condition on physical GPU 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-Math-1.5B"
DATA = ROOT / "data" / "MATH500.json"
DATA_SHA256 = "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
OFFICIAL_COMMIT = "720a8e9d084c87a630595e316f5260f1d7c3446c"
ENGINE_KEYS = {
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "overwrite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("base", "low_temp", "power"), required=True)
    parser.add_argument("--name")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--proposal-temperature", type=float, default=0.25)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=192)
    parser.add_argument("--suffix-policy", choices=("official_overlap", "full"), default="official_overlap")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    (ROOT / "notes").mkdir(exist_ok=True)
    with (ROOT / "notes" / "ACTIVITY_LOG.md").open("a") as handle:
        handle.write(f"- {timestamp()} — {message}\n")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compatible(left: dict, right: dict) -> bool:
    """Engine throughput settings may change when resuming; sampling settings may not."""
    return {key: value for key, value in left.items() if key not in ENGINE_KEYS} == {
        key: value for key, value in right.items() if key not in ENGINE_KEYS
    }


def engine_settings(config: dict) -> dict:
    return {key: config[key] for key in ENGINE_KEYS - {"overwrite"}}


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal 1; physical GPU 0 is out of scope")
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise RuntimeError("MATH500 data hash does not match the official repository")
    if args.max_tokens % args.block_size and args.condition == "power":
        raise ValueError("Use a block size that divides max_tokens for the official protocol")

    from transformers import AutoTokenizer
    from vllm import LLM

    from math500_sampling.grading import grade, load_math500, prompt
    from math500_sampling.sampler import Chain, PowerSampler

    dataset = load_math500(DATA)
    end = min(args.offset + args.limit, len(dataset))
    examples = dataset[args.offset:end]
    if not examples:
        raise ValueError("The requested dataset slice is empty")
    name = args.name or f"{args.condition}_seed{args.seed}_offset{args.offset}_n{len(examples)}"
    result_path = ROOT / "results" / f"{name}.json"
    checkpoint_path = ROOT / "results" / "checkpoints" / f"{name}.json"
    config = {
        **vars(args),
        "name": name,
        "model": MODEL,
        "dataset": "official MATH500",
        "dataset_sha256": DATA_SHA256,
        "official_commit": OFFICIAL_COMMIT,
        "prompt": "official Qwen Math COT boxed prompt",
        "physical_gpu": 1,
        "vllm_logprobs_mode": "processed_logprobs",
    }

    if result_path.is_file() and not args.overwrite:
        previous = json.loads(result_path.read_text())
        if not compatible(previous.get("config", {}), config):
            raise RuntimeError(f"Existing result has a different config: {result_path}")
        print(json.dumps(previous["summary"], indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = [prompt(example["prompt"]) for example in examples]
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    if max(map(len, prompt_ids)) + args.max_tokens > 4096:
        raise ValueError("Prompt plus completion exceeds the configured 4,096-token context")

    resume_block, chains, rng_states, previous_elapsed = 0, None, None, 0.0
    engine_history = [engine_settings(config)]
    if checkpoint_path.is_file() and not args.overwrite:
        checkpoint = json.loads(checkpoint_path.read_text())
        if not compatible(checkpoint.get("config", {}), config):
            raise RuntimeError(f"Existing checkpoint has a different config: {checkpoint_path}")
        resume_block = checkpoint["next_block"]
        chains = [Chain.from_dict(values) for values in checkpoint["chains"]]
        rng_states = checkpoint["rng_states"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        engine_history = checkpoint.get(
            "engine_history", [engine_settings(checkpoint["config"])]
        )
        if engine_history[-1] != engine_settings(config):
            engine_history.append(engine_settings(config))

    log(
        f"Starting {name}: condition={args.condition}, questions={args.offset}:{end}, "
        f"seed={args.seed}, resume_block={resume_block}, config={config}"
    )
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        logprobs_mode="processed_logprobs",
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    sampler = PowerSampler(llm, tokenizer)
    started = time.time()

    def save_checkpoint(next_block, current_chains, current_rng_states):
        elapsed = previous_elapsed + time.time() - started
        atomic_json(
            checkpoint_path,
            {
                "config": config,
                "next_block": next_block,
                "elapsed_seconds": elapsed,
                "engine_history": engine_history,
                "rng_states": current_rng_states,
                "chains": [chain.to_dict() for chain in current_chains],
            },
        )
        log(f"Checkpointed {name}: next_block={next_block}, elapsed={elapsed / 60:.1f} min")

    if args.condition == "power":
        chains = sampler.power(
            prompt_ids,
            alpha=args.alpha,
            proposal_temperature=args.proposal_temperature,
            mcmc_steps=args.mcmc_steps,
            block_size=args.block_size,
            max_tokens=args.max_tokens,
            seed=args.seed,
            suffix_policy=args.suffix_policy,
            chains=chains,
            rng_states=rng_states,
            start_block=resume_block,
            checkpoint=save_checkpoint,
        )
    else:
        temperature = args.proposal_temperature if args.condition == "low_temp" else args.temperature
        chains = sampler.standard(prompt_ids, args.max_tokens, temperature, args.seed)

    elapsed = previous_elapsed + time.time() - started
    rows = []
    for local_index, (example, text, chain) in enumerate(zip(examples, prompts, chains, strict=True)):
        response = tokenizer.decode(chain.tokens, skip_special_tokens=True)
        correct, parsed = grade(response, example["answer"])
        rows.append(
            {
                "dataset_index": args.offset + local_index,
                "id": example["id"],
                "source": example["source"],
                "question": example["prompt"],
                "prompt": text,
                "ground_truth": example["answer"],
                "response": response,
                "parsed_answer": parsed,
                "correct": correct,
                "response_tokens": len(chain.tokens),
                "sampled_tokens": chain.sampled_tokens,
                "attempts": chain.attempts,
                "accepts": chain.accepts,
                "acceptance_rate": chain.acceptance_rate,
                "mean_log_ratio": mean(chain.log_ratios) if chain.log_ratios else None,
            }
        )

    summary = {
        "n": len(rows),
        "accuracy": mean([row["correct"] for row in rows]),
        "parse_rate": mean([row["parsed_answer"] is not None for row in rows]),
        "mean_response_tokens": mean([row["response_tokens"] for row in rows]),
        "mean_sampled_tokens": mean([row["sampled_tokens"] for row in rows]),
        "mean_acceptance_rate": mean([row["acceptance_rate"] for row in rows]),
        "elapsed_seconds": elapsed,
        "blocks": math.ceil(args.max_tokens / args.block_size) if args.condition == "power" else None,
    }
    atomic_json(
        result_path,
        {"config": config, "engine_history": engine_history, "summary": summary, "rows": rows},
    )
    log(
        f"Finished {name}: accuracy={summary['accuracy']:.4f}, parse={summary['parse_rate']:.4f}, "
        f"sampled_tokens={summary['mean_sampled_tokens']:.1f}, elapsed={elapsed / 60:.1f} min; "
        f"saved {result_path.relative_to(ROOT)}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
