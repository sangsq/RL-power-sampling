"""Matched raw-prompt vLLM pass@N evaluation of the base and saved power policy."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time

from power_sampling.grading import grade, ground_truth, load_jsonl
from power_sampling.hf_policy import MODEL, REVISION
from experiments.gsm8k_power_policy.analyze import extract_explicit_answer

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "gsm8k_qwen05b/data/gsm8k/test.jsonl"
ADAPTER = ROOT / "checkpoints/alpha2_seed0/final"
CONDITIONS = {"base_t1": 1.0, "base_t03": 0.3, "policy_t1": 1.0}


def pass_at_k(n, correct, k):
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError("Require 1 <= k <= n and 0 <= correct <= n")
    return 1.0 if n - correct < k else 1 - math.comb(n - correct, k) / math.comb(n, k)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="raw_500_n32")
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--request-batch", type=int, default=8)
    parser.add_argument("--condition", choices=list(CONDITIONS), nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Expose only physical GPU 1")
    if not 1 <= args.questions <= 1319 or not 1 <= args.samples <= 32 or args.request_batch < 1:
        parser.error("Invalid evaluation size")
    rows = load_jsonl(DATA)
    indices = sorted(random.Random(args.subset_seed).sample(range(len(rows)), args.questions))
    folder = ROOT / "results/pass_at_n" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL, "revision": REVISION, "adapter": str(ADAPTER),
              "adapter_sha256": digest(ADAPTER / "adapter_model.safetensors"),
              "dataset_sha256": digest(DATA), "indices": indices,
              "samples": args.samples, "subset_seed": args.subset_seed, "seed": args.seed,
              "seed_policy": "seed + original test index, paired across conditions",
              "prompt": "raw question, add_special_tokens=False", "max_tokens": 512,
              "stop": "EOS only; no stop strings or forced EOS", "conditions": CONDITIONS,
              "request_batch": args.request_batch, "dtype": "bfloat16",
              "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
              "engine": {"max_model_len": 1024, "max_num_seqs": 256,
                         "gpu_memory_utilization": 0.8, "enforce_eager": True,
                         "enable_prefix_caching": True, "generation_config": "vllm",
                         "enable_lora": True, "max_lora_rank": 8, "lora_dtype": "bfloat16"},
              "versions": {name: version(name) for name in ("vllm", "torch", "transformers", "peft")},
              "source_sha256": {p.name: digest(p) for p in
                                (Path(__file__), ROOT / "analyze.py")}}
    config_path = folder / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing run has a different configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    prompts = [{"prompt_token_ids": tokenizer.encode(rows[i]["question"], add_special_tokens=False)} for i in indices]
    assert max(len(p["prompt_token_ids"]) for p in prompts) + 512 <= 1024
    engine = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
                 dtype="bfloat16", **config["engine"])
    policy = LoRARequest("power_alpha2_seed0_final", 1, str(ADAPTER))
    for name in args.condition:
        output = folder / f"{name}.jsonl"
        completed = load_jsonl(output) if output.exists() else []
        if [r["index"] for r in completed] != indices[:len(completed)]:
            raise ValueError("Saved questions do not match the fixed subset")
        if any(len(r["samples"]) != args.samples for r in completed):
            raise ValueError("Incomplete saved sample group")
        started = time.monotonic()
        previous_seconds = completed[-1]["elapsed_seconds"] if completed else 0
        with output.open("a", buffering=1) as handle:
            for start in range(len(completed), len(indices), args.request_batch):
                batch_indices = indices[start:start + args.request_batch]
                params = [SamplingParams(n=args.samples, temperature=CONDITIONS[name], top_p=1,
                                         top_k=-1, max_tokens=512, seed=args.seed + i,
                                         repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                                         ignore_eos=False) for i in batch_indices]
                outputs = engine.generate(prompts[start:start + len(batch_indices)], params,
                                          lora_request=policy if name == "policy_t1" else None,
                                          use_tqdm=False)
                for i, request in zip(batch_indices, outputs, strict=True):
                    target = ground_truth(rows[i])
                    samples = []
                    assert len(request.outputs) == args.samples
                    for candidate in request.outputs:
                        answer = extract_explicit_answer(candidate.text)
                        correct, boxed = grade(candidate.text, target)
                        numeric_correct = answer is not None and grade("\\boxed{" + answer + "}", target)[0]
                        samples.append({"text": candidate.text, "token_ids": list(candidate.token_ids),
                                        "correct": bool(correct), "answer_correct": bool(numeric_correct),
                                        "boxed": boxed, "answer": answer,
                                        "finish_reason": candidate.finish_reason, "stop_reason": candidate.stop_reason})
                    record = {"index": i, "question": rows[i]["question"], "ground_truth": target,
                              "samples": samples, "n_correct": sum(s["correct"] for s in samples),
                              "n_answer_correct": sum(s["answer_correct"] for s in samples),
                              "elapsed_seconds": previous_seconds + time.monotonic() - started}
                    handle.write(json.dumps(record) + "\n")
                    completed.append(record)
                print(json.dumps({"condition": name, "completed": len(completed), "total": len(indices),
                                  "seconds": completed[-1]["elapsed_seconds"]}), flush=True)
        print(json.dumps({"condition": name, "pass_at_1": sum(r["n_answer_correct"] for r in completed) / len(completed) / args.samples,
                          "pass_at_max": sum(r["n_answer_correct"] > 0 for r in completed) / len(completed)}), flush=True)


if __name__ == "__main__":
    main()
