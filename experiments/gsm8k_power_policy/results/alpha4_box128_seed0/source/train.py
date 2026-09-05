"""Train one fixed-alpha LoRA policy, with a wall-clock cap below 30 minutes."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import threading
import time

import torch

from power_sampling.grading import load_jsonl
from power_sampling.hf_policy import MODEL, REVISION, generate, load_policy, update

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "gsm8k_qwen05b/data/gsm8k/train.jsonl"


def format_prompt(template, question):
    if template.count("{question}") != 1:
        raise ValueError("Prompt template must contain exactly one {question}")
    return template.replace("{question}", question)


def epoch_indices(indices, seed, epoch):
    order = indices.copy()
    if epoch:
        random.Random(seed + epoch).shuffle(order)
    return order


def save_checkpoint(model, tokenizer, optimizer, folder, step, config):
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    torch.save({"optimizer": optimizer.state_dict(), "step": step,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), "config": config}, folder / "state.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=1680)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1740 or args.steps < 1:
        parser.error("Require 0 < max-seconds <= 1740 and positive steps")
    if args.prompts_per_step < 1 or 1024 % args.prompts_per_step:
        parser.error("prompts-per-step must be a positive divisor of 1024")
    if min(args.generation_batch, args.microbatch, args.max_tokens) < 1:
        parser.error("Batch sizes and max-tokens must be positive")
    template = args.prompt_file.read_text() if args.prompt_file else "{question}"
    format_prompt(template, "")
    started = time.monotonic()
    deadline = started + args.max_seconds
    # A daemon watchdog also bounds a stuck CUDA operation; normal exit saves first.
    watchdog = threading.Timer(args.max_seconds + 30, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    result_dir = ROOT / "results" / args.name
    checkpoint_dir = ROOT / "checkpoints" / args.name
    result_dir.mkdir(parents=True, exist_ok=False)
    source_dir = result_dir / "source"
    source_dir.mkdir()
    source_hashes = {}
    for path in (Path(__file__), ROOT.parents[1] / "src/power_sampling/hf_policy.py"):
        (source_dir / path.name).write_bytes(path.read_bytes())
        source_hashes[str(path.relative_to(ROOT.parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (result_dir / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(DATA)
    dev_questions = {row["question"] for row in rows[:256]}
    candidates = [i for i in range(256, len(rows)) if rows[i]["question"] not in dev_questions]
    indices = random.Random(0).sample(candidates, 1024)
    random.Random(args.seed).shuffle(indices)
    config = {**vars(args), "prompt_file": str(args.prompt_file.resolve()) if args.prompt_file else None,
              "prompt_template": template, "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "model": MODEL, "revision": REVISION, "rank": 8,
              "lora_alpha": 16, "target_modules": "all-linear", "group_size": 4,
              "epoch_order": "initial train_indices; later epochs shuffle them with Random(seed + epoch)",
              "train_indices": indices, "dev_indices": list(range(256)),
              "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
              "question_hashes": [hashlib.sha256(rows[i]["question"].encode()).hexdigest() for i in indices],
              "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")}}
    (result_dir / "config.json").write_text(json.dumps(config, indent=2))
    model, tokenizer = load_policy(trainable=True)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in name for name, _ in trainable)
    config["trainable_parameters"] = sum(p.numel() for _, p in trainable)
    config["trainable_names"] = [name for name, _ in trainable]
    config["targeted_modules"] = model.targeted_module_names
    (result_dir / "config.json").write_text(json.dumps(config, indent=2))
    prompts = {i: tokenizer.encode(format_prompt(template, rows[i]["question"]), add_special_tokens=False) for i in indices}
    steps_per_epoch = len(indices) // args.prompts_per_step
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    save_checkpoint(model, tokenizer, optimizer, checkpoint_dir / "step_0000", 0, config)
    print(json.dumps({"event": "start", "trainable_parameters": config["trainable_parameters"],
                      "modules": len(config["targeted_modules"]), "budget_seconds": args.max_seconds}), flush=True)
    step = 0
    reason = "step_limit"
    durations = []
    try:
        with (result_dir / "metrics.jsonl").open("w", buffering=1) as metrics, (result_dir / "rollouts.jsonl").open("w", buffering=1) as rollouts:
            for step_index in range(args.steps):
                # Reserve time for one complete update and checkpoint serialization.
                allowance = max(30, max(durations[-3:], default=0) * 1.2)
                if time.monotonic() + allowance >= deadline:
                    reason = "time_budget"
                    break
                batch_started = time.monotonic()
                epoch, batch_in_epoch = divmod(step_index, steps_per_epoch)
                order = epoch_indices(indices, args.seed, epoch)
                offset = batch_in_epoch * args.prompts_per_step
                batch_indices = order[offset:offset + args.prompts_per_step]
                qs = [prompts[i] for i in batch_indices for _ in range(4)]
                responses = generate(model, tokenizer, qs, args.max_tokens,
                                     batch_size=args.generation_batch, deadline=deadline - 15)
                generation_seconds = time.monotonic() - batch_started
                if time.monotonic() + 15 >= deadline:
                    reason = "time_budget"
                    break
                stats = update(model, optimizer, qs, responses, tokenizer.pad_token_id,
                               alpha=args.alpha, microbatch=args.microbatch)
                step = step_index + 1
                lengths = [len(x) for x in responses]
                ended = [x[-1] == tokenizer.eos_token_id for x in responses]
                old, base, advantages = (stats.pop(k) for k in ("old_logprobs", "base_logprobs", "advantages"))
                for i, tokens in enumerate(responses):
                    rollouts.write(json.dumps({"step": step, "dataset_index": batch_indices[i // 4],
                                              "token_ids": tokens, "old_logprob": old[i], "base_logprob": base[i],
                                              "advantage": advantages[i], "eos": ended[i]}) + "\n")
                duration = time.monotonic() - batch_started
                durations.append(duration)
                stats.update(step=step, epoch=epoch, batch_in_epoch=batch_in_epoch,
                             prompts=args.prompts_per_step, responses=len(responses),
                             elapsed_seconds=time.monotonic() - started,
                             update_seconds=duration, generation_seconds=generation_seconds,
                             mean_length=sum(lengths) / len(lengths), median_length=float(torch.tensor(lengths).median()),
                             sampled_tokens=sum(lengths), eos_rate=sum(ended) / len(ended),
                             first_token_eos_rate=sum(x[0] == tokenizer.eos_token_id for x in responses) / len(responses),
                             short_rate=sum(n < 16 for n in lengths) / len(lengths),
                             truncated_rate=1 - sum(ended) / len(ended),
                             peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
                metrics.write(json.dumps(stats) + "\n")
                print(json.dumps(stats), flush=True)
                if step in (32, 64) or step % 128 == 0:
                    save_checkpoint(model, tokenizer, optimizer, checkpoint_dir / f"step_{step:04d}", step, config)
    except TimeoutError:
        reason = "time_budget_discarded_rollout"
    except (FloatingPointError, RuntimeError) as error:
        # Preserve evidence and the last parameter state; no silent retries or retuning.
        reason = f"error: {error}"
    finally:
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(model, tokenizer, optimizer, checkpoint_dir / "final", step, config)
        summary = {"completed_steps": step, "reason": reason, "wall_seconds": time.monotonic() - started,
                   "final_checkpoint": str(checkpoint_dir / "final"), "max_seconds": args.max_seconds}
        (result_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        watchdog.cancel()
    if reason.startswith("error:"):
        raise RuntimeError(reason)


if __name__ == "__main__":
    main()
