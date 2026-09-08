"""HF generation and exact sequence-level RLOO for a frozen-base LoRA policy."""

from contextlib import nullcontext
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def load_policy(adapter=None, trainable=False, model_id=MODEL, revision=REVISION):
    from peft import LoraConfig, PeftModel, get_peft_model

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Expose only physical GPU 1: CUDA_VISIBLE_DEVICES=1")
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, target_modules="all-linear",
            lora_dropout=0.0, bias="none",
        ))
    return model.eval(), tokenizer


class Deadline(StoppingCriteria):
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        return time.monotonic() >= self.deadline


def trim_response(tokens, eos, tokenizer=None, stop_string=None):
    """Retain the first genuine EOS; discard only tokens after it."""
    tokens = tokens[:tokens.index(eos) + 1] if eos in tokens else tokens
    if stop_string and stop_string in tokenizer.decode(tokens, skip_special_tokens=True):
        # Retain the generated token completing the delimiter, excluding later EOS padding.
        low, high = 1, len(tokens)
        while low < high:
            middle = (low + high) // 2
            if stop_string in tokenizer.decode(tokens[:middle], skip_special_tokens=True):
                high = middle
            else:
                low = middle + 1
        tokens = tokens[:low]
    return tokens


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_tokens=512, temperature=1.0, batch_size=4, deadline=None,
             stop_string=None):
    model.eval()
    config = GenerationConfig(
        do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
        typical_p=1.0, repetition_penalty=1.0, num_beams=1, use_cache=True,
        max_new_tokens=max_tokens, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        stop_strings=[stop_string] if stop_string else None,
    )
    responses = []
    for start in range(0, len(prompts), batch_size):
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Rollout deadline reached; discard unfinished batch")
        inputs = tokenizer.pad({"input_ids": prompts[start:start + batch_size]},
                               padding=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, generation_config=config,
                                 **({"tokenizer": tokenizer} if stop_string else {}),
                                 stopping_criteria=[Deadline(deadline)] if deadline else None)
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Rollout deadline reached; discard unfinished batch")
        width = inputs.input_ids.shape[1]
        responses.extend(trim_response(row[width:].tolist(), tokenizer.eos_token_id, tokenizer, stop_string)
                         for row in outputs)
    return responses


def response_batch(prompts, responses, pad_id, device):
    """Right-padded teacher forcing, with shifted response masks including EOS."""
    sequences = [q + x for q, x in zip(prompts, responses, strict=True)]
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (q, x, seq) in enumerate(zip(prompts, responses, sequences, strict=True)):
        if not q or not x:
            raise ValueError("Prompts and responses must contain at least one token")
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        attention[i, :len(seq)] = 1
        mask[i, len(q):len(seq)] = True
    return ids[:, :-1], attention[:, :-1], ids[:, 1:], mask[:, 1:]


def sequence_logprobs(model, prompts, responses, pad_id):
    ids, attention, labels, mask = response_batch(prompts, responses, pad_id, model.device)
    # Prompt-only positions do not contribute to the objective or need vocabulary logits.
    positions = mask.any(dim=0).nonzero(as_tuple=True)[0]
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False,
                   logits_to_keep=positions).logits
    labels, mask = labels[:, positions], mask[:, positions]
    logprobs = -F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none")
    return (logprobs * mask).sum(dim=1)


@torch.no_grad()
def score(model, prompts, responses, pad_id, batch_size=2, reference=False, token_budget=None):
    with model.disable_adapter() if reference else nullcontext():
        return torch.cat([
            sequence_logprobs(model, prompts[i:j], responses[i:j], pad_id)
            for i, j in scoring_ranges(prompts, responses, batch_size, token_budget)
        ])


def scoring_ranges(prompts, responses, batch_size, token_budget=None):
    """Keep short examples batched while bounding padded tokens for long prompts."""
    start = 0
    while start < len(prompts):
        end, width = start, 0
        while end < min(start+batch_size, len(prompts)):
            next_width = max(width, len(prompts[end])+len(responses[end]))
            if end > start and token_budget and next_width*(end-start+1) > token_budget:
                break
            end, width = end+1, next_width
        yield start, end
        start = end


def leave_one_out(scores, group_size):
    if group_size < 2 or scores.numel() % group_size:
        raise ValueError("Need complete groups of at least two samples")
    grouped = scores.reshape(-1, group_size)
    return (grouped - (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)).flatten()


def update(model, optimizer, prompts, responses, pad_id, group_size=4, alpha=2.0, microbatch=2,
           token_budget=None):
    """One update on one fresh batch; sequence sums, no advantage normalization."""
    model.eval()  # Keep dropout disabled even in the differentiable forward.
    old = score(model, prompts, responses, pad_id, microbatch, token_budget=token_budget)
    base = score(model, prompts, responses, pad_id, microbatch, reference=True, token_budget=token_budget)
    values = alpha * base - old
    advantage = leave_one_out(values, group_size)
    optimizer.zero_grad(set_to_none=True)
    surrogate = 0.0
    for i, j in scoring_ranges(prompts, responses, microbatch, token_budget):
        current = sequence_logprobs(model, prompts[i:j], responses[i:j], pad_id)
        loss = -(advantage[i:j] * current).sum() / len(prompts)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite policy-gradient loss")
        surrogate += loss.detach().item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True,
    )
    optimizer.step()
    return {
        "objective": values.mean().item(), "true_loss_without_log_z": -values.mean().item(),
        "surrogate_loss": surrogate, "base_logprob": base.mean().item(),
        "policy_logprob": old.mean().item(), "kl_estimate": (old - base).mean().item(),
        "sequence_entropy": -old.mean().item(), "score_std": values.std().item(),
        "advantage_std": advantage.std().item(), "gradient_norm": norm.item(),
        "gradient_clipped": norm.item() > 1.0,
        "old_logprobs": old.tolist(), "base_logprobs": base.tolist(),
        "advantages": advantage.tolist(),
    }
