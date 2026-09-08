"""Calibrate or train equal-rollout stages with increasing response horizons."""

import argparse
from pathlib import Path
from importlib.metadata import version
import os
import random
import threading
import time

import numpy as np
import torch

from common import HERE, ROOT, append, log, rows, run_name, settings_path, sha, spec, write
from power_sampling.diagnostics import effective_alpha, true_loss
from power_sampling.evaluation import answer_scores
from power_sampling.grading import ground_truth
from power_sampling.hf_policy import generate, load_policy, score, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--alpha', type=float, choices=[2,4], default=2)
    parser.add_argument('--name', required=True, help='A fresh run name; saved runs cannot be overwritten.')
    parser.add_argument('--max-seconds', type=int, default=1680)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1740 or Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('Require a simple run name and 0 < max-seconds <= 1740.')
    s = spec(args.alpha)
    assert not args.calibrate or args.alpha == 2, 'Reuse the frozen alpha-2 calibration quota.'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    name = args.name
    out = HERE/'results'/name
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started+min(args.max_seconds, 550 if args.calibrate else args.max_seconds)
    watchdog = threading.Timer(deadline-started+20, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    random.seed(s['seed'])
    torch.manual_seed(s['seed'])
    model, tok = load_policy(trainable=True, model_id=s['model'], revision=s['revision'])
    data = rows(ROOT/s['train_dataset'])
    prompts = {i: tok.encode(s['prompt_template'].replace('{question}', data[i]['question']), add_special_tokens=False)
               for i in set(s['train_indices']+s['diagnostic_indices'])}
    assert set(s['train_indices']).isdisjoint(s['diagnostic_indices'])
    config = {**s, 'internal_deadline_seconds': args.max_seconds, 'external_timeout_seconds': 1790, 'calibration': args.calibrate, 'settings_sha256': sha(settings_path(args.alpha)),
              'data_sha256': sha(ROOT/s['train_dataset']),
              'versions': {n: version(n) for n in ['torch','transformers','peft']}}
    write(out/'config.json', config)
    source = out/'source'
    source.mkdir()
    files = [HERE/'scripts/train.py', HERE/'scripts/common.py', ROOT/'src/power_sampling/hf_policy.py',
             ROOT/'src/power_sampling/diagnostics.py', ROOT/'src/power_sampling/evaluation.py']
    for p in files:
        (source/p.name).write_bytes(p.read_bytes())
    write(out/'source_hashes.json', {str(p.relative_to(ROOT)):sha(p) for p in files})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=s['lr'], weight_decay=0)
    group, micro = s['group_size'], s['microbatch']
    step = 0
    fixed = None
    monitor_times = []

    def save(label):
        path = HERE/'checkpoints'/name/label
        model.save_pretrained(path)
        tok.save_pretrained(path)
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config':config}, path/'state.pt')

    def monitor(stage):
        nonlocal fixed
        tick = time.monotonic()
        qp = [prompts[i] for i in s['diagnostic_indices'] for _ in range(s['diagnostic_samples'])]
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20260907+step)
            responses = generate(model, tok, qp, s['diagnostic_max_tokens'],
                                 batch_size=s['generation_batch'], deadline=deadline-20)
        old = score(model, qp, responses, tok.pad_token_id, micro).tolist()
        base = score(model, qp, responses, tok.pad_token_id, micro, reference=True).tolist()
        lengths = list(map(len,responses))
        if fixed is None:
            fixed = {'prompts':qp,'responses':responses,'base_logprobs':base}
            write(out/'fixed_probe.json',fixed)
            fixed_old = old
        else:
            fixed_old = score(model, fixed['prompts'], fixed['responses'], tok.pad_token_id, micro).tolist()
        grades = [answer_scores(tok.decode(tokens,skip_special_tokens=True),
                   ground_truth(data[s['diagnostic_indices'][j//s['diagnostic_samples']]]))
                  for j,tokens in enumerate(responses)]
        record = {'step':step,'completed_stage':stage,'probe_max_tokens':512,'elapsed_seconds':time.monotonic()-started,
                  'true_loss':true_loss(base,old,s['alpha'],s['diagnostic_samples']),
                  'on_policy':effective_alpha(base,old,lengths,s['diagnostic_samples']),
                  'fixed_responses':effective_alpha(fixed['base_logprobs'],fixed_old,list(map(len,fixed['responses'])),s['diagnostic_samples']),
                  'mean_length':float(np.mean(lengths)),'median_length':float(np.median(lengths)),
                  'cap_rate':float(np.mean([x[-1]!=tok.eos_token_id for x in responses])),
                  'answer_accuracy':float(np.mean([g['answer_correct'] for g in grades])),
                  'boxed_accuracy':float(np.mean([g['boxed_correct'] for g in grades])),
                  'seconds':time.monotonic()-tick}
        append(out/'probe.jsonl',record)
        append(out/'probe_samples.jsonl',{'step':step,'responses':responses,'old_logprobs':old,
                                         'base_logprobs':base,'fixed_policy_logprobs':fixed_old,'grades':grades})
        append(out/'probe_logprobs.jsonl', {'step': step, 'old_logprobs': old,
               'base_logprobs': base, 'fixed_policy_logprobs': fixed_old, 'grades': grades})
        monitor_times.append(record['seconds'])
        print(record,flush=True)

    updates_per_stage = 2 if args.calibrate else s['updates_per_stage']
    assert isinstance(updates_per_stage,int) and updates_per_stage>0
    steps_per_epoch = len(s['train_indices'])//s['prompts_per_step']
    stage_records=[]
    complete=False
    try:
        if not args.calibrate:
            save('initial')
        monitor(0)
        for stage,length in enumerate(s['lengths'],1):
            durations=[]
            stage_tokens=0
            for local_step in range(updates_per_stage):
                if time.monotonic()+max(25,max(durations[-3:],default=0)*1.3)+30 >= deadline:
                    raise TimeoutError('Cannot complete the equal-quota schedule within the external budget.')
                epoch,batch=divmod(step,steps_per_epoch)
                order=s['train_indices'].copy()
                if epoch:
                    random.Random(s['seed']+epoch).shuffle(order)
                indices=order[batch*s['prompts_per_step']:(batch+1)*s['prompts_per_step']]
                qp=[prompts[i] for i in indices for _ in range(group)]
                tick=time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                responses=generate(model,tok,qp,length,batch_size=s['generation_batch'],deadline=deadline-30)
                generation_seconds=time.monotonic()-tick
                stats=update(model,optimizer,qp,responses,tok.pad_token_id,group_size=group,
                             alpha=s['alpha'],microbatch=micro)
                old,base,adv=[stats.pop(k) for k in ['old_logprobs','base_logprobs','advantages']]
                step+=1
                lengths=list(map(len,responses))
                duration=time.monotonic()-tick
                stage_tokens+=sum(lengths)
                record={**stats,'step':step,'scored_policy_step':step-1,'stage':stage,'max_new_tokens':length,
                        'stage_update':local_step+1,'rollouts':len(responses),'sampled_tokens':sum(lengths),
                        'mean_length':float(np.mean(lengths)),'median_length':float(np.median(lengths)),
                        'cap_rate':float(np.mean([x[-1]!=tok.eos_token_id for x in responses])),
                        'true_loss':true_loss(base,old,s['alpha'],group),
                        'effective_alpha':effective_alpha(base,old,lengths,group),
                        'update_seconds':duration,'generation_seconds':generation_seconds,
                        'elapsed_seconds':time.monotonic()-started,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
                append(out/'metrics.jsonl',record)
                append(out/'rollouts.jsonl',{'step':step,'stage':stage,'max_new_tokens':length,'indices':indices,
                       'responses':responses,'old_logprobs':old,'base_logprobs':base,'advantages':adv})
                durations.append(duration)
                print(record,flush=True)
            stage_record={'stage':stage,'max_new_tokens':length,'updates':updates_per_stage,
                          'rollouts':updates_per_stage*s['prompts_per_step']*group,
                          'sampled_tokens':stage_tokens,'update_seconds':sum(durations),
                          'median_update_seconds':float(np.median(durations))}
            stage_records.append(stage_record)
            append(out/'stages.jsonl',stage_record)
            if not args.calibrate:
                save(f'length_{length:03d}')
                monitor(stage)
                log(f'Completed length {length}: {stage_record["rollouts"]} training rollouts, elapsed {(time.monotonic()-started)/60:.1f} min.')
        if args.calibrate:
            monitor(8)
        complete=True
    finally:
        optimizer.zero_grad(set_to_none=True)
        save('final')
        write(out/'completion.json',{'complete':complete,'steps':step,'stages_completed':len(stage_records),
              'wall_seconds':time.monotonic()-started,'diagnostic_seconds':sum(monitor_times),
              'within_requested_time_range':3000 <= time.monotonic()-started <= 4200,
              'equal_stage_rollouts':complete and len({r['rollouts'] for r in stage_records})==1})
        watchdog.cancel()
    if args.calibrate:
        per_cycle=sum(r['median_update_seconds'] for r in stage_records)
        diagnostic_allowance=max(monitor_times)*9+45
        quota=max(1,round((s['target_seconds']-diagnostic_allowance)/per_cycle))
        write(HERE/'results/calibration_plan.json',{'updates_per_stage':quota,
             'rollouts_per_stage':quota*s['prompts_per_step']*group,'total_updates':quota*8,
             'predicted_seconds':quota*per_cycle+diagnostic_allowance,
             'seconds_per_eight_updates':per_cycle,'diagnostic_allowance_seconds':diagnostic_allowance,
             'note':'Prediction from a short separate policy run; stopping lengths and hardware throughput may change.'})
    log(f'{name} completed: {step} updates, {(time.monotonic()-started)/60:.1f} minutes.')


if __name__=='__main__':
    main()
