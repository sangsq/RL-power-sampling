"""Evaluate frozen HF baselines or one saved adapter, with resumable output."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from power_sampling.grading import canonical_answer, grade, ground_truth, load_jsonl
from power_sampling.hf_policy import MODEL, REVISION, generate, load_policy, score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "gsm8k_qwen05b/data/gsm8k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--score", action="store_true", help="Measure alpha=2 objective for T=1 adapter samples")
    args = parser.parse_args()
    if args.score and (not args.adapter or args.temperature != 1):
        parser.error("--score requires a T=1 adapter policy")
    folder = ROOT / "results" / "evaluation" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "adapter": str(args.adapter.resolve()) if args.adapter else None,
              "model": MODEL, "revision": REVISION}
    if args.adapter:
        config["adapter_sha256"] = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    config_path = folder / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing evaluation has a different configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    output = folder / "responses.jsonl"
    completed = load_jsonl(output) if output.exists() else []
    examples = load_jsonl(DATA / f"{args.split}.jsonl")[:args.limit]
    assert [row["index"] for row in completed] == list(range(len(completed)))
    if len(completed) < len(examples):
        model, tokenizer = load_policy(args.adapter)
        batch_questions = max(1, args.batch_size // args.samples)
        if len(completed) % batch_questions:
            raise ValueError("Resume requires a complete generation batch")
        started = time.monotonic()
        with output.open("a", buffering=1) as handle:
            for start in range(len(completed), len(examples), batch_questions):
                # Distinct batch seeds make an interrupted evaluation reproducible.
                torch.manual_seed(args.seed + start * 10007)
                batch = examples[start:start + batch_questions]
                prompts = [tokenizer.encode(row["question"], add_special_tokens=False) for row in batch]
                repeated = [q for q in prompts for _ in range(args.samples)]
                tokens = generate(model, tokenizer, repeated, args.max_tokens,
                                  args.temperature, args.batch_size)
                if args.score:
                    lp = score(model, repeated, tokens, tokenizer.pad_token_id)
                    lp0 = score(model, repeated, tokens, tokenizer.pad_token_id, reference=True)
                for j, example in enumerate(batch):
                    group = tokens[j * args.samples:(j + 1) * args.samples]
                    responses = [tokenizer.decode(x, skip_special_tokens=True) for x in group]
                    grades = [grade(x, ground_truth(example)) for x in responses]
                    row = {"index": start + j, "question": example["question"],
                           "ground_truth": ground_truth(example), "responses": responses,
                           "token_ids": group, "correct": [x[0] for x in grades],
                           "parsed_answers": [x[1] for x in grades],
                           "canonical_answers": [canonical_answer(x) for x in responses],
                           "lengths": [len(x) for x in group],
                           "eos": [x[-1] == tokenizer.eos_token_id for x in group],
                           "first_token_eos": [x[0] == tokenizer.eos_token_id for x in group]}
                    if args.score:
                        offset = slice(j * args.samples, (j + 1) * args.samples)
                        row.update(policy_logprobs=lp[offset].tolist(), base_logprobs=lp0[offset].tolist())
                    handle.write(json.dumps(row) + "\n")
                    completed.append(row)
                print(json.dumps({"evaluation": args.name, "completed": len(completed),
                                  "total": len(examples), "session_seconds": time.monotonic() - started}), flush=True)
    from analyze import summarize
    summary = summarize(completed)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
