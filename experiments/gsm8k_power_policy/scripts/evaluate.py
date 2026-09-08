"""Evaluate the published GSM8K baselines with the frozen sampling protocol."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.gsm8k_power_policy.scripts"


from power_sampling.evaluation import scores, pass_at_k as pass_k

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from power_sampling.grading import ground_truth, load_jsonl
from power_sampling.evidence import compact_file

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "results/evaluation"
DATA = ROOT.parents[1] / "data/gsm8k"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def log(message):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    (REPO / "agent").mkdir(exist_ok=True)
    with (REPO / "agent/agent_logging").open("a") as f:
        f.write(f"\n## {stamp}\n\n- {message}\n")
    print(f"{stamp} {message}", flush=True)


def condition(engine, tokenizer, protocol, *, split, label, temperature=1.0, adapter=None,
              alpha=None, block_size=32, mh_steps=2, smoke=False):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from power_sampling.sampler import PowerSampler

    class MeteredSampler(PowerSampler):
        def score(self, sequences, starts):
            self.scoring_tokens += sum(map(len, sequences))
            self.scoring_outputs += len(sequences)
            return super().score(sequences, starts)

    rows = load_jsonl(DATA / ("train.jsonl" if split == "dev" else "test.jsonl"))
    indices = protocol["dev_indices" if split == "dev" else "test_indices"]
    samples = protocol["dev_samples" if split == "dev" else "test_samples"]
    if smoke:
        indices, samples = indices[:2], 2
    folder = OUT / ("smoke" if smoke else split) / label
    folder.mkdir(parents=True, exist_ok=True)
    batch_questions = 8
    seed = 810000 if split == "dev" else 910000
    config = {"experiment": "gsm8k_power_policy", "split": split, "label": label,
              "indices": indices, "samples": samples, "temperature": temperature,
              "adapter": adapter, "alpha": alpha, "block_size": block_size if alpha else None,
              "mh_steps": mh_steps if alpha else None, "suffix_policy": "full" if alpha else None,
              "max_tokens": 512, "model": MODEL, "revision": REVISION,
              "prompt_template": protocol["prompt_template"], "engine": protocol["engine"],
              "batch_questions": batch_questions, "seed": seed,
              "seed_policy": "standard: seed+1009*original_index, n independent draws; MCMC: seed+100000*batch_offset, chain j uses seed+j*10007",
              "protocol_sha256": digest(OUT / "protocol.json")}
    path = folder / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config, "Resume configuration mismatch"
    else:
        write_json(path, config)
    output = OUT / "raw" / ("smoke" if smoke else split) / f"{label}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_jsonl(output) if output.exists() else []
    assert [r["index"] for r in completed] == indices[:len(completed)]
    assert all(len(r["samples"]) == samples for r in completed)
    assert len(completed) % batch_questions == 0 or len(completed) == len(indices)
    if len(completed) == len(indices) and (folder / "completion.json").exists():
        return completed
    lora = LoRARequest(label, 1 if "alpha2" in label else 2, str(Path(adapter).resolve())) if adapter else None
    sampler = MeteredSampler(engine, tokenizer) if alpha else None
    started = time.monotonic()
    log(f"Matched evaluation start {split}/{label}: {len(indices)} questions x {samples}; smoke={smoke}.")
    with output.open("a", buffering=1) as handle:
        for offset in range(len(completed), len(indices), batch_questions):
            tick = time.monotonic()
            batch = indices[offset:offset + batch_questions]
            prompts = [tokenizer.encode(protocol["prompt_template"].replace("{question}", rows[i]["question"]),
                                        add_special_tokens=False) for i in batch]
            assert max(map(len, prompts)) + 513 <= protocol["engine"]["max_model_len"]
            if alpha:
                sampler.scoring_tokens = sampler.scoring_outputs = 0
                chains = sampler.power([p for p in prompts for _ in range(samples)], alpha=alpha,
                    proposal_temperature=1 / alpha, mcmc_steps=mh_steps, block_size=block_size,
                    max_tokens=512, seed=seed + 100000 * offset, suffix_policy="full")
            else:
                params = [SamplingParams(n=samples, temperature=temperature, top_p=1, top_k=-1,
                    max_tokens=512, seed=seed + 1009 * i, repetition_penalty=1,
                    presence_penalty=0, frequency_penalty=0, ignore_eos=False) for i in batch]
                outputs = engine.generate([{"prompt_token_ids": p} for p in prompts], params,
                                          lora_request=lora, use_tqdm=False)
            records = []
            for j, i in enumerate(batch):
                target = ground_truth(rows[i])
                saved = []
                candidates = chains[j * samples:(j + 1) * samples] if alpha else outputs[j].outputs
                assert len(candidates) == samples
                for draw, candidate in enumerate(candidates):
                    ids = list(candidate.tokens if alpha else candidate.token_ids)
                    assert 0 < len(ids) <= 512
                    if tokenizer.eos_token_id in ids:
                        assert ids.index(tokenizer.eos_token_id) == len(ids) - 1
                    # Decode the saved IDs identically for every condition.
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    item = {"draw": draw, "text": text, "token_ids": ids,
                            "generated_tokens": candidate.sampled_tokens if alpha else len(ids),
                            "eos": tokenizer.eos_token_id in ids, **scores(text, target)}
                    if alpha:
                        assert all(math.isfinite(x) for x in candidate.log_ratios)
                        item.update(attempts=candidate.attempts, accepts=candidate.accepts,
                                    log_ratios=candidate.log_ratios, base_logprobs=candidate.base_logprobs,
                                    proposal_logprobs=candidate.proposal_logprobs)
                    saved.append(item)
                records.append({"index": i, "question": rows[i]["question"], "ground_truth": target,
                                "prompt_tokens": len(prompts[j]), "samples": saved})
            handle.write("".join(json.dumps(row) + "\n" for row in records))
            completed.extend(records)
            timing = {"offset": offset, "questions": len(batch), "seconds": time.monotonic() - tick,
                      "generated_tokens": sum(s["generated_tokens"] for r in records for s in r["samples"]),
                      "scoring_input_tokens": sampler.scoring_tokens if alpha else 0,
                      "scoring_discarded_output_tokens": sampler.scoring_outputs if alpha else 0}
            with (folder / "batches.jsonl").open("a") as f:
                f.write(json.dumps(timing) + "\n")
            write_json(OUT / "status.json", {"phase": split, "condition": label, "smoke": smoke,
                "completed_questions": len(completed), "total_questions": len(indices),
                "session_seconds": time.monotonic() - started,
                "updated_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")})
            print(json.dumps({"condition": label, "completed": len(completed), "total": len(indices),
                              "seconds": time.monotonic() - started}), flush=True)
    write_json(folder / "completion.json", {"questions": len(indices), "samples": samples,
        "session_seconds": time.monotonic() - started, "output_sha256": digest(output),
        "raw_path": str(output.relative_to(REPO))})
    log(f"Matched evaluation completed {split}/{label} in {time.monotonic() - started:.1f}s; smoke={smoke}.")
    return completed


def select_temperature(protocol):
    values = []
    for t in protocol["temperature_grid"]:
        records = load_jsonl(OUT / "raw/dev" / f"base_t{t:g}.jsonl")
        assert [r["index"] for r in records] == protocol["dev_indices"]
        estimates = {str(k): sum(pass_k(len(r["samples"]), sum(s["correct"]["answer"] for s in r["samples"]), k)
                                for r in records) / len(records) for k in [1, 2, 4, 8]}
        values.append({"temperature": t, "pass_at_k": estimates, "score": sum(estimates.values()) / 4})
    best = min((row for row in values if row["temperature"] < 1), key=lambda r: (-r["score"], r["temperature"]))
    selection = {"selected_temperature": best["temperature"], "grid": values,
                 "selected_before_test": True, "metric": "mean answer pass@1/2/4/8; lower-T tie break"}
    path = OUT / "temperature_selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == selection
    else:
        write_json(path, selection)
    log(f"Development-only low temperature frozen at T={best['temperature']} before test generation.")
    return best["temperature"]



def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True,
                        choices=['base_t1', 'base_t0.25', 'policy_alpha2', 'mcmc_alpha2', 'select_temperature'])
    parser.add_argument('--output', type=Path, required=True,
                        help='A separate result directory; do not overwrite published evidence.')
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        parser.error('Expose only physical GPU 1.')
    if args.output.resolve() == OUT.resolve():
        parser.error('Use a separate output directory to preserve the published evaluation.')
    if args.condition == 'policy_alpha2' and not args.adapter:
        parser.error('The fixed-length policy requires --adapter PATH.')
    source_protocol = json.loads((OUT/'protocol.json').read_text())
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == source_protocol
    else:
        write_json(path, source_protocol)
    for split in ['train', 'test']:
        expected = next(value for key, value in source_protocol['input_sha256'].items()
                        if key.endswith(f'data/gsm8k/{split}.jsonl'))
        assert digest(DATA/f'{split}.jsonl') == expected
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    engine = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION, **source_protocol['engine'])
    write_json(OUT/'environment.json', {'versions': {n: version(n) for n in ['vllm','torch','transformers','peft']},
                                       'physical_gpu': 1, 'source_sha256': digest(__file__)})
    if args.condition == 'select_temperature':
        for t in source_protocol['temperature_grid']:
            condition(engine, tok, source_protocol, split='dev', label=f'base_t{t:g}', temperature=t)
        select_temperature(source_protocol)
        return
    options = {
        'base_t1': {'temperature': 1.0},
        'base_t0.25': {'temperature': 0.25},
        'policy_alpha2': {'adapter': str(args.adapter) if args.adapter else None},
        'mcmc_alpha2': {'alpha': 2, 'block_size': 32, 'mh_steps': 2},
    }
    condition(engine, tok, source_protocol, split='test', label=args.condition, **options[args.condition])
    compact_file(OUT/'raw/test'/f'{args.condition}.jsonl', OUT/'questions'/f'{args.condition}.jsonl')


if __name__ == '__main__':
    main()
