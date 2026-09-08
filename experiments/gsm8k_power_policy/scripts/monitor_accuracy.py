"""Read-only answer monitoring; labels never enter the policy update."""

from power_sampling.evaluation import answer_scores

import json
import random
import time

import torch

from power_sampling.grading import ground_truth
from power_sampling.hf_policy import generate


def evaluate_accuracy(model, tokenizer, rows, prompts, *, step, args, folder, started, deadline):
    """Evaluate a fixed development slice without advancing training RNG states."""
    begin = time.monotonic()
    python_rng = random.getstate()
    try:
        devices = [model.device] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.eval_seed)
            random.seed(args.eval_seed)
            responses = generate(model, tokenizer, prompts, args.max_tokens,
                                 temperature=1.0, batch_size=args.generation_batch,
                                 deadline=deadline, stop_string=args.stop_string)
    finally:
        random.setstate(python_rng)
    records = []
    for i, (row, tokens) in enumerate(zip(rows, responses, strict=True)):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        records.append({"step": step, "dataset_index": i, "question": row["question"],
                        "ground_truth": ground_truth(row), "token_ids": tokens, "response": text,
                        **answer_scores(text, ground_truth(row))})
    metrics = {"step": step, "questions": len(records), "seed": args.eval_seed,
               "elapsed_seconds": time.monotonic() - started,
               "evaluation_seconds": time.monotonic() - begin,
               "answer_accuracy": sum(r["answer_correct"] for r in records) / len(records),
               "boxed_accuracy": sum(r["boxed_correct"] for r in records) / len(records),
               "parse_rate": sum(r["parsed_answer"] is not None for r in records) / len(records),
               "mean_length": sum(map(len, responses)) / len(responses),
               "sampled_tokens": sum(map(len, responses))}
    with (folder / "accuracy_responses.jsonl").open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    with (folder / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(metrics) + "\n")
    print(json.dumps({"event": "development_accuracy", **metrics}), flush=True)
    return metrics
