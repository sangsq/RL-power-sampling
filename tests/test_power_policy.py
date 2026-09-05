"""Probability, masking, and frozen-reference checks without CUDA or downloads."""

import copy
import itertools

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from power_sampling.hf_policy import leave_one_out, response_batch, score, sequence_logprobs, trim_response, update


def tiny_policy():
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        attention_dropout=0.0, eos_token_id=0, pad_token_id=0)
    base = Qwen2ForCausalLM(config).eval()
    clean = copy.deepcopy(base)
    policy = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                                             lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    return policy.eval(), clean


def test_exact_tree_rloo_gradient():
    # Three leaves: EOS, continuation+EOS, continuation+token at horizon H=2.
    base = torch.tensor([.2, .8 * .3, .8 * .7], dtype=torch.double)
    logits = torch.tensor([.3, -.2, .7], dtype=torch.double, requires_grad=True)
    lp = logits.log_softmax(0)
    p = lp.exp()
    objective = (p * (2 * base.log() - lp)).sum()
    exact = torch.autograd.grad(objective, logits, retain_graph=True)[0]
    estimate = 0
    for i, j in itertools.product(range(3), repeat=2):
        scores = 2 * base[[i, j]].log() - lp[[i, j]].detach()
        advantages = leave_one_out(scores, 2)
        estimate = estimate + (p[i] * p[j]).detach() * (advantages * lp[[i, j]]).mean()
    actual = torch.autograd.grad(estimate, logits)[0]
    torch.testing.assert_close(actual, exact)
    optimum = (base.square() / base.square().sum()).log().requires_grad_()
    target_lp = optimum.log_softmax(0)
    optimum_objective = (target_lp.exp() * (2 * base.log() - target_lp)).sum()
    torch.testing.assert_close(torch.autograd.grad(optimum_objective, optimum)[0], torch.zeros_like(optimum))


def test_eos_and_response_mask():
    assert trim_response([2, 0, 0, 0], 0) == [2, 0]
    assert trim_response([2, 3], 0) == [2, 3]
    ids, attention, labels, mask = response_batch([[4, 5], [6]], [[7, 0], [8]], 0, "cpu")
    assert mask.tolist() == [[False, True, True], [True, False, False]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert labels[0][mask[0]].tolist() == [7, 0]
    with pytest.raises(ValueError):
        leave_one_out(torch.ones(3), 2)


def test_batch_scores_and_microbatch_gradients():
    model, _ = tiny_policy()
    qs, xs = [[1, 2], [3], [4, 5, 6], [7]], [[8, 0], [9, 10, 0], [11, 12], [0]]
    batched = score(model, qs, xs, 0, batch_size=4)
    single = score(model, qs, xs, 0, batch_size=1)
    torch.testing.assert_close(batched, single, atol=2e-6, rtol=2e-6)
    weights = torch.tensor([1.2, -.5, .7, -1.4])
    parameters = [p for p in model.parameters() if p.requires_grad]
    def grads(microbatch):
        model.zero_grad(set_to_none=True)
        for i in range(0, 4, microbatch):
            lp = sequence_logprobs(model, qs[i:i+microbatch], xs[i:i+microbatch], 0)
            (-(weights[i:i+microbatch] * lp).sum() / 4).backward()
        return torch.cat([p.grad.flatten() for p in parameters])
    expected = grads(4)
    torch.testing.assert_close(grads(1), expected, atol=2e-6, rtol=2e-5)


def test_identity_reference_and_checkpoint(tmp_path):
    model, clean = tiny_policy()
    qs, xs = [[1, 2]] * 4, [[3, 0], [4, 5, 0], [6, 7], [0]]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    reference = score(model, qs, xs, 0, reference=True).clone()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = update(model, optimizer, qs, xs, 0, alpha=1)
    assert result["gradient_norm"] == 0
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    update(model, optimizer, qs, xs, 0, alpha=2)
    torch.testing.assert_close(score(model, qs, xs, 0, reference=True), reference, atol=0, rtol=0)
    for k, v in model.state_dict().items():
        if "lora_" not in k:
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if "lora_" in k)
    model.save_pretrained(tmp_path / "adapter")
    reloaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", is_trainable=True).eval()
    torch.testing.assert_close(score(reloaded, qs, xs, 0), score(model, qs, xs, 0))

    torch.save({"optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, tmp_path / "state.pt")
    expected = torch.rand(4)
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    torch.set_rng_state(state["rng"])
    torch.testing.assert_close(torch.rand(4), expected)
    restored_optimizer = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad], lr=1e-3)
    restored_optimizer.load_state_dict(state["optimizer"])
    update(model, optimizer, qs, xs, 0)
    update(reloaded, restored_optimizer, qs, xs, 0)
    torch.testing.assert_close(score(reloaded, qs, xs, 0), score(model, qs, xs, 0))


def test_explicit_answer_sensitivity():
    from experiments.gsm8k_power_policy.analyze import extract_explicit_answer, relaxed_correct
    assert extract_explicit_answer("Work 12 + 60 = 72") is None
    assert extract_explicit_answer("#### 72\nThe answer is: 72") == "72"
    assert extract_explicit_answer("The answer is: $1,234.5 dollars.") == "1,234.5"
    assert extract_explicit_answer("The final answer = -1/2") == "-1/2"
    assert extract_explicit_answer("Answer: unknown, though 72 was mentioned") is None
    assert extract_explicit_answer(r"\boxed{7}; #### 72") == "7"
    assert relaxed_correct({"responses": ["#### 72", "answer is 7", "calculated 72"], "ground_truth": "72"}) == [True, False, False]


def test_training_prompt_and_epoch_order():
    from experiments.gsm8k_power_policy.train import epoch_indices, format_prompt
    template = r"Solve the following question step by step and place final result in \box{}: {question}"
    assert format_prompt(template, "What is {2 + 3}?") == (
        r"Solve the following question step by step and place final result in \box{}: What is {2 + 3}?"
    )
    with pytest.raises(ValueError):
        format_prompt("{question} {question}", "Q")
    indices = list(range(1024))
    assert epoch_indices(indices, 0, 0) == indices
    order = epoch_indices(indices, 0, 1)
    assert sorted(order) == indices and order != indices
    assert order == epoch_indices(indices, 0, 1)
    assert indices == list(range(1024))
    assert all(len(set(order[i:i+16])) == 16 for i in range(0, 1024, 16))
