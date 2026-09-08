"""Train one fixed-alpha LoRA policy under an explicit wall-clock budget."""

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

from power_sampling.grading import ground_truth, load_jsonl
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
    parser.add_argument("--stop-string")
    parser.add_argument("--hour-budget", action="store_true", help="Explicitly authorized one-hour run")
    parser.add_argument("--track-accuracy", action="store_true")
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--eval-questions", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=20260905)
    args = parser.parse_args()
    cap = 3570 if args.hour_budget else 1740
    if not 0 < args.max_seconds <= cap or args.steps < 1:
        parser.error(f"Require 0 < max-seconds <= {cap} and positive steps")
    if args.eval_every < 1 or not 1 <= args.eval_questions <= 256:
        parser.error("Require positive eval-every and 1 <= eval-questions <= 256")
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
    for path in (Path(__file__), ROOT / "monitor_accuracy.py", ROOT / "analyze.py",
                 ROOT.parents[1] / "src/power_sampling/hf_policy.py",
                 ROOT.parents[1] / "src/power_sampling/grading.py"):
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
              "accuracy_monitoring": {"enabled": args.track_accuracy,
                  "indices": list(range(args.eval_questions)), "split": "train (held out from updates)",
                  "every_steps": args.eval_every, "initial_and_final": True,
                  "samples_per_question": 1, "temperature": 1.0, "seed": args.eval_seed,
                  "rng_isolated": True, "included_in_wall_budget": True,
                  "primary": "explicit numeric answer, allowing nested boxed braces",
                  "secondary": "historical strict boxed grader", "used_for_updates_or_selection": False},
              "decoding": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "typical_p": 1.0,
                           "repetition_penalty": 1.0, "do_sample": True, "num_beams": 1,
                           "stop_string": args.stop_string, "include_stop_token": True,
                           "max_new_tokens": args.max_tokens, "physical_gpu": 1,
                           "dtype": "bfloat16", "attention": "sdpa", "add_special_tokens": False},
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
    evaluation_steps = []
    evaluation_seconds = 0.0
    if args.track_accuracy:
        from experiments.gsm8k_power_policy.monitor_accuracy import answer_scores, evaluate_accuracy
        eval_rows = rows[:args.eval_questions]
        eval_prompts = [tokenizer.encode(format_prompt(template, row["question"]), add_special_tokens=False)
                        for row in eval_rows]

    def monitor():
        nonlocal evaluation_seconds
        result = evaluate_accuracy(model, tokenizer, eval_rows, eval_prompts, step=step,
                                   args=args, folder=result_dir, started=started, deadline=deadline - 10)
        evaluation_seconds += result["evaluation_seconds"]
        evaluation_steps.append(step)

    try:
        if args.track_accuracy:
            monitor()
        with (result_dir / "metrics.jsonl").open("w", buffering=1) as metrics, (result_dir / "rollouts.jsonl").open("w", buffering=1) as rollouts:
            for step_index in range(args.steps):
                # Reserve time for one complete update and checkpoint serialization.
                allowance = max(60 if args.track_accuracy else 30, max(durations[-3:], default=0) * 1.2)
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
                                     batch_size=args.generation_batch, deadline=deadline - 15,
                                     stop_string=args.stop_string)
                generation_seconds = time.monotonic() - batch_started
                if time.monotonic() + 15 >= deadline:
                    reason = "time_budget"
                    break
                stats = update(model, optimizer, qs, responses, tokenizer.pad_token_id,
                               alpha=args.alpha, microbatch=args.microbatch)
                step = step_index + 1
                lengths = [len(x) for x in responses]
                ended = [x[-1] == tokenizer.eos_token_id for x in responses]
                stopped = [bool(args.stop_string and args.stop_string in tokenizer.decode(x, skip_special_tokens=True))
                           for x in responses]
                old, base, advantages = (stats.pop(k) for k in ("old_logprobs", "base_logprobs", "advantages"))
                accuracy = [answer_scores(tokenizer.decode(tokens, skip_special_tokens=True),
                                         ground_truth(rows[batch_indices[i // 4]]))
                            for i, tokens in enumerate(responses)] if args.track_accuracy else []
                for i, tokens in enumerate(responses):
                    rollouts.write(json.dumps({"step": step, "dataset_index": batch_indices[i // 4],
                                              "token_ids": tokens, "old_logprob": old[i], "base_logprob": base[i],
                                              "advantage": advantages[i], "eos": ended[i],
                                              "stop_string_reached": stopped[i],
                                              **(accuracy[i] if accuracy else {})}) + "\n")
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
                             stop_string_rate=sum(stopped) / len(stopped),
                             truncated_rate=sum(not (e or s) for e, s in zip(ended, stopped)) / len(ended),
                             peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
                if accuracy:
                    stats.update(train_answer_accuracy=sum(r["answer_correct"] for r in accuracy) / len(accuracy),
                                 train_boxed_accuracy=sum(r["boxed_correct"] for r in accuracy) / len(accuracy))
                metrics.write(json.dumps(stats) + "\n")
                print(json.dumps(stats), flush=True)
                if step in (32, 64) or step % 128 == 0:
                    save_checkpoint(model, tokenizer, optimizer, checkpoint_dir / f"step_{step:04d}", step, config)
                if args.track_accuracy and step % args.eval_every == 0 and time.monotonic() + 45 < deadline:
                    monitor()
    except TimeoutError:
        reason = "time_budget_discarded_rollout"
    except (FloatingPointError, RuntimeError) as error:
        # Preserve evidence and the last parameter state; no silent retries or retuning.
        reason = f"error: {error}"
    finally:
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(model, tokenizer, optimizer, checkpoint_dir / "final", step, config)
        if args.track_accuracy and step not in evaluation_steps and not reason.startswith("error:"):
            try:
                if time.monotonic() + 15 < deadline:
                    monitor()
            except TimeoutError:
                print(json.dumps({"event": "final_evaluation_skipped", "reason": "deadline"}), flush=True)
        summary = {"completed_steps": step, "reason": reason, "wall_seconds": time.monotonic() - started,
                   "final_checkpoint": str(checkpoint_dir / "final"), "max_seconds": args.max_seconds,
                   "evaluation_steps": evaluation_steps, "evaluation_seconds": evaluation_seconds,
                   "final_evaluation_complete": step in evaluation_steps}
        (result_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        watchdog.cancel()
    if reason.startswith("error:"):
        raise RuntimeError(reason)


if __name__ == "__main__":
    main()
