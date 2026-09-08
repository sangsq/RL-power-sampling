"""Check that gradient accumulation preserves the full-batch RLOO update."""

import copy
import torch

from power_sampling.hf_policy import update
from test_power_policy import tiny_policy


def test_rloo_update_is_independent_of_microbatch_size():
    full, _ = tiny_policy()
    split = copy.deepcopy(full)
    prompts = [[1, 2]]*4+[[3, 4, 5]]*4
    responses = [[6, 0], [7, 8, 0], [9, 0], [10, 11, 12, 0]]*2
    optimizers = [torch.optim.SGD(m.parameters(), lr=.01) for m in [full, split]]
    a = update(full, optimizers[0], prompts, responses, 0, microbatch=8)
    b = update(split, optimizers[1], prompts, responses, 0, microbatch=2, token_budget=11)
    assert a['backward_microbatches'] == 1
    assert b['backward_microbatches'] > 4
    assert a['effective_batch_responses'] == b['effective_batch_responses'] == 8
    torch.testing.assert_close(torch.tensor(a['advantages']), torch.tensor(b['advantages']))
    for x, y in zip(full.parameters(), split.parameters()):
        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)
