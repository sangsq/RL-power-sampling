"""Verify the shared response scorer and LoRA update on the OLMo architecture."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import OlmoConfig, OlmoForCausalLM

from power_sampling.hf_policy import response_batch, score, sequence_logprobs, update


def test_olmo_response_mask_and_frozen_reference():
    torch.manual_seed(13)
    config=OlmoConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=64, eos_token_id=0, pad_token_id=0)
    model=get_peft_model(OlmoForCausalLM(config),LoraConfig(
        r=8,lora_alpha=16,target_modules='all-linear',lora_dropout=0,bias='none',task_type='CAUSAL_LM')).eval()
    prompts=[[1,2]]*4
    responses=[[3,0],[4,5,0],[6,7,8,0],[9,0]]
    ids,attention,labels,mask=response_batch(prompts,responses,0,'cpu')
    logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits
    expected=(-torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,reduction='none')*mask).sum(1)
    torch.testing.assert_close(sequence_logprobs(model,prompts,responses,0),expected)
    base=score(model,prompts,responses,0,reference=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0)
    stats=update(model,optimizer,prompts,responses,0,microbatch=2)
    assert stats['effective_batch_responses']==4 and stats['backward_microbatches']==2
    torch.testing.assert_close(score(model,prompts,responses,0,reference=True),base,atol=0,rtol=0)
    assert not torch.equal(score(model,prompts,responses,0),base)
