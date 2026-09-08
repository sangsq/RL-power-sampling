"""Check sequence-power slopes and the RLOO gradient against exact enumeration."""

from importlib import import_module

import numpy as np
import torch

diagnostics = import_module('power_sampling.diagnostics')


def test_question_intercepts_and_length_confounding():
    base, policy, lengths = [], [], []
    for q in range(8):
        for j in range(4):
            length = 3+j
            x = -2*length-(j % 2)-q
            base.append(x)
            lengths.append(length)
            policy.append(2.5*x+q*10)
    result = diagnostics.effective_alpha(base, policy, lengths, 4)
    assert np.isclose(result['alpha'], 2.5)
    assert np.isclose(result['question_slope_std'], 0)
    assert np.isclose(result['alpha_length_adjusted'], 2.5)
    adjusted = diagnostics.effective_alpha(base, np.asarray(policy)+3*np.asarray(lengths), lengths, 4)
    assert np.isclose(adjusted['alpha_length_adjusted'], 2.5)
    assert not np.isclose(adjusted['alpha'], 2.5)


def test_rloo_expected_gradient_matches_true_reverse_kl():
    from power_sampling.hf_policy import leave_one_out
    theta = torch.tensor([-.4, .2, .6], dtype=torch.float64, requires_grad=True)
    log_policy = theta.log_softmax(0)
    log_base = torch.tensor([.2, .3, .5], dtype=torch.float64).log()
    alpha = 1.5
    target = (alpha*log_base).log_softmax(0)
    exact = (log_policy.exp()*(log_policy-target)).sum()
    exact_gradient = torch.autograd.grad(exact, theta, retain_graph=True)[0]
    expected_surrogate = theta.sum()*0
    for a in range(3):
        for b in range(3):
            indices = torch.tensor([a,b])
            reward = (alpha*log_base[indices]-log_policy[indices]).detach()
            advantage = leave_one_out(reward, 2)
            probability = (log_policy[a]+log_policy[b]).exp().detach()
            expected_surrogate = expected_surrogate-probability*(advantage*log_policy[indices]).mean()
    actual = torch.autograd.grad(expected_surrogate, theta)[0]
    torch.testing.assert_close(actual, exact_gradient)
