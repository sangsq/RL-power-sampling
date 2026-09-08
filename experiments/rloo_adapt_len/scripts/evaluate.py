"""Evaluate the final adapter on the historical 500-question GSM8K subset."""

import argparse
import json
import os
import time

from power_sampling.evidence import compact_file

from common import HERE, ROOT, append, evaluation_path, log, rows, run_name, sha, spec, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha',type=float,choices=[2,4],default=2)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    s=spec(args.alpha)
    name=run_name(args.alpha)
    checkpoint=HERE/'checkpoints'/name/'final'
    assert json.loads((HERE/'results'/name/'completion.json').read_text())['complete']
    out=evaluation_path(args.alpha)
    out.mkdir(exist_ok=True)
    config={'settings':s,'adapter_sha256':sha(checkpoint/'adapter_model.safetensors'),
            'dataset_sha256':sha(ROOT/s['test_dataset']),'seed':910000,
            'seed_policy':'910000 + 1009 * original question index; 32 draws per question',
            'batch_questions':8,'max_new_tokens':512,'temperature':1.0,'source_sha256':sha(__file__)}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config
    else:
        write(out/'config.json',config)
    data=rows(ROOT/s['test_dataset'])
    done=rows(out/'samples.jsonl')
    assert [r['index'] for r in done]==s['test_indices'][:len(done)]
    if len(done)==len(s['test_indices']):
        compact_file(out/'samples.jsonl', out/'questions.jsonl')
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from power_sampling.evaluation import scores
    from power_sampling.grading import ground_truth
    tok=AutoTokenizer.from_pretrained(s['model'],revision=s['revision'],local_files_only=True)
    engine=LLM(model=s['model'],revision=s['revision'],tokenizer_revision=s['revision'],**s['evaluation_engine'])
    adapter=LoRARequest(name,1,str(checkpoint))
    start=time.monotonic()
    for offset in range(len(done),len(s['test_indices']),8):
        indices=s['test_indices'][offset:offset+8]
        prompts=[tok.encode(s['prompt_template'].replace('{question}',data[i]['question']),add_special_tokens=False) for i in indices]
        params=[SamplingParams(n=32,temperature=1,top_p=1,top_k=-1,max_tokens=512,seed=910000+1009*i,
                repetition_penalty=1,presence_penalty=0,frequency_penalty=0,ignore_eos=False) for i in indices]
        outputs=engine.generate([{'prompt_token_ids':p} for p in prompts],params,lora_request=adapter,use_tqdm=False)
        for i,output in zip(indices,outputs):
            samples=[]
            for draw,response in enumerate(output.outputs):
                tokens=list(response.token_ids)
                text=tok.decode(tokens,skip_special_tokens=True)
                samples.append({'draw':draw,'text':text,'token_ids':tokens,'eos':tok.eos_token_id in tokens,
                                **scores(text,ground_truth(data[i]))})
            record={'index':i,'question':data[i]['question'],'ground_truth':ground_truth(data[i]),'samples':samples}
            append(out/'samples.jsonl',record)
            done.append(record)
        write(out/'status.json',{'questions':len(done),'total':len(s['test_indices']),'complete':len(done)==len(s['test_indices'])})
    write(out/'completion.json',{'complete':True,'questions':len(done),'samples_per_question':32,
                                 'wall_seconds':time.monotonic()-start,'samples_sha256':sha(out/'samples.jsonl')})
    compact_file(out/'samples.jsonl', out/'questions.jsonl')
    log(f'Completed final GSM8K evaluation: {len(done)} questions x 32 draws.')


if __name__=='__main__':
    main()
