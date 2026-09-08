"""Plot the fixed-horizon diagnostics and compare saved GSM8K evaluations."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from common import HERE, ROOT, rows, sha, spec, stamp, write
from power_sampling.evidence import evaluate_rows




def paired_comparisons(vectors, policy='adapt_len_alpha2'):
    if policy not in vectors:
        return {}
    n = len(vectors[policy]['answer']['1'])
    bootstrap = np.random.default_rng(20260907).integers(n, size=(10000, n))
    comparisons = {}
    for name in vectors:
        if name == policy:
            continue
        comparisons[name] = {}
        for grading in ['answer', 'boxed']:
            comparisons[name][grading] = {}
            for k in ['1', '4', '32']:
                delta = vectors[policy][grading][k] - vectors[name][grading][k]
                comparisons[name][grading][k] = {
                    'difference': float(delta.mean()),
                    'paired_question_ci95': np.quantile(delta[bootstrap].mean(1), [.025, .975]).tolist()}
    return comparisons


def main():
    s=spec()
    out=HERE/'results/alpha2_adapt_len'
    probes=rows(out/'probe.jsonl')
    stages=rows(out/'stages.jsonl')
    completion=json.loads((out/'completion.json').read_text()) if (out/'completion.json').exists() else None
    legacy=ROOT/'experiments/gsm8k_power_policy/results/evaluation/questions'
    paths={'base_t1':legacy/'base_t1.jsonl','fixed512_alpha2':legacy/'policy_alpha2.jsonl',
           'base_t0.25':legacy/'base_t0.25.jsonl','adapt_len_alpha2':HERE/'results/evaluation/questions.jsonl'}
    alpha4_path=HERE/'results/evaluation_alpha4/questions.jsonl'
    if alpha4_path.exists() and len(rows(alpha4_path))==500:
        paths['adapt_len_alpha4']=alpha4_path
    paths['mcmc_alpha2']=legacy/'mcmc_alpha2.jsonl'
    evaluated={}
    vectors={}
    mcmc_path=legacy/'mcmc_alpha2.jsonl'
    mcmc_config_path=legacy.parent/'test/mcmc_alpha2/config.json'
    mcmc_config=json.loads(mcmc_config_path.read_text())
    assert mcmc_config['model']==s['model'] and mcmc_config['revision']==s['revision']
    assert mcmc_config['prompt_template']==s['prompt_template'] and mcmc_config['max_tokens']==512
    assert (mcmc_config['alpha'],mcmc_config['block_size'],mcmc_config['mh_steps'],mcmc_config['suffix_policy'])==(2,32,2,'full')
    for name,path in paths.items():
        raw=rows(path)
        assert len(raw)==500, f'{name} must complete all 500 questions before report generation.'
        assert [r['index'] for r in raw]==s['test_indices']
        evaluated[name],vectors[name]=evaluate_rows(raw)
    comparisons=paired_comparisons(vectors)
    evaluation_protocol={'question_indices':s['test_indices'],'questions':500,'samples_per_question':32,
             'mcmc_settings':{'alpha':2,'proposal_temperature':0.5,'block_size':32,'mh_steps':2,
                              'max_tokens':512,'suffix_policy':'full'},
             'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [mcmc_config_path,*paths.values()]}}
    out4=HERE/'results/alpha4_adapt_len'
    probes4=rows(out4/'probe.jsonl')
    completion4=json.loads((out4/'completion.json').read_text()) if (out4/'completion.json').exists() else None
    alpha4_summary={'completion':completion4,'stages':rows(out4/'stages.jsonl'),'probes':probes4,
                    'paired_comparisons':paired_comparisons(vectors,'adapt_len_alpha4')}
    write(HERE/'results/summary.json',{'updated_at':stamp(),'completion':completion,'stages':stages,
          'probes':probes,'evaluations':evaluated,'paired_comparisons':comparisons,'alpha4':alpha4_summary,
          'evaluation_protocol':evaluation_protocol})
    if probes4:
        decomposition={}
        loss_fig,loss_axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
        for alpha,records,color in [(2,probes,'C0'),(4,probes4,'C1')]:
            t=[p['elapsed_seconds']/60 for p in records]
            label=f'Target alpha {alpha}'
            sample_path=HERE/f'results/alpha{alpha}_adapt_len/probe_logprobs.jsonl'
            samples=rows(sample_path)
            assert [r['step'] for r in samples]==[r['step'] for r in records]
            points=[]
            for record,sample in zip(records,samples):
                nll=-np.asarray(sample['old_logprobs'],dtype=float)
                ce=-np.asarray(sample['base_logprobs'],dtype=float)
                assert nll.shape==ce.shape==(len(s['diagnostic_indices'])*s['diagnostic_samples'],)
                components={'policy_nll':nll,'reference_ce':ce,'true_loss':alpha*ce-nll,'base_kl':ce-nll}
                point={'step':record['step'],'elapsed_minutes':record['elapsed_seconds']/60}
                for key,values in components.items():
                    question_means=values.reshape(-1,s['diagnostic_samples']).mean(1)
                    point[key]={'mean':float(values.mean()),
                                'question_se':float(question_means.std(ddof=1)/np.sqrt(len(question_means)))}
                assert np.isclose(point['true_loss']['mean'],record['true_loss']['mean'],rtol=0,atol=1e-8)
                assert np.isclose(point['true_loss']['question_se'],record['true_loss']['question_se'],rtol=0,atol=1e-8)
                points.append(point)
            decomposition[str(alpha)]={'points':points,'source':str(sample_path.relative_to(ROOT)),
                                       'source_sha256':sha(sample_path)}
            for ax,key in zip(loss_axes,['policy_nll','reference_ce','true_loss']):
                ax.errorbar(t,[p[key]['mean'] for p in points],
                            yerr=[1.96*p[key]['question_se'] for p in points],
                            marker='o',markersize=4,capsize=2,color=color,label=label)
            axes[0].errorbar(t,[p['on_policy']['alpha'] for p in records],
                             yerr=[p['on_policy']['question_slope_std'] or 0 for p in records],marker='o',color=color,label=label)
            axes[1].plot(t,[100*p['answer_accuracy'] for p in records],marker='o',color=color,label=label+' / answer')
            axes[1].plot(t,[100*p['boxed_accuracy'] for p in records],linestyle='--',marker='s',
                           markerfacecolor='white',markersize=4,color=color,label=label+' / boxed')
            axes[2].plot(t,[p['median_length'] for p in records],marker='o',color=color,label=label)
        for ax,title in zip(loss_axes,[r'Policy NLL: $E_\pi[-\log\pi]$',
                                      r'Reference CE: $E_\pi[-\log p_0]$',
                                      r'True loss: $\alpha\,\mathrm{CE}-\mathrm{NLL}$']):
            ax.set(title=title,xlabel='Elapsed training time (minutes)',ylabel='Nats per response')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        shared_limits=(min(ax.get_ylim()[0] for ax in loss_axes[:2]),
                       max(ax.get_ylim()[1] for ax in loss_axes[:2]))
        for ax in loss_axes[:2]:
            ax.set_ylim(shared_limits)
        loss_fig.savefig(HERE/'figures/loss_decomposition.png',dpi=160)
        plt.close(loss_fig)
        write(HERE/'results/loss_decomposition.json',{
            'definitions':{'policy_nll':'E_pi[-log pi(y|x)] (policy entropy)',
                           'reference_ce':'E_pi[-log p0(y|x)]',
                           'true_loss':'alpha * reference_ce - policy_nll',
                           'base_kl':'reference_ce - policy_nll = E_pi[log pi - log p0]'},
            'units':'Nats per response; no token-length normalization.',
            'uncertainty':'Pointwise 1.96 times question SE; four responses are averaged within each question.',
            'runs':decomposition})
        axes[1].set_ylim(0,100)
        for ax,title in zip(axes,['Effective alpha ± question slope SD',
                                 'Probe response accuracy (%)','Median response length (tokens)']):
            ax.set(title=title,xlabel='Elapsed training time (minutes)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.savefig(HERE/'figures/alpha_comparison.png',dpi=160)
        plt.close(fig)
    labels={'base_t1':'Base T=1','fixed512_alpha2':'Fixed-length RLOO, alpha 2',
            'base_t0.25':'Base T=0.25','adapt_len_alpha2':'Adaptive RLOO, alpha 2',
            'adapt_len_alpha4':'Adaptive RLOO, alpha 4','mcmc_alpha2':'MCMC, alpha 2'}

    def interval(d):
        lo,hi=d['paired_question_ci95']
        return f"{100*d['difference']:+.2f} pp (95% CI [{100*lo:+.2f}, {100*hi:+.2f}])"

    def table(values,names):
        lines=['| Method | Answer pass@1 | Answer pass@4 | Answer pass@32 | Boxed pass@1 | Median tokens |',
               '|---|---:|---:|---:|---:|---:|']
        for name in names:
            r=values[name]
            p=r['pass_at_k']
            lines.append(f"| {labels[name]} | {p['answer']['1']:.2%} | {p['answer']['4']:.2%} | "
                         f"{p['answer']['32']:.2%} | {p['boxed']['1']:.2%} | {r['median_length']:.0f} |")
        return '\n'.join(lines)

    intro=('# RLOO with increasing response length\n\n'
           '**The adaptive-length runs outperform the historical fixed-length policy and preserve boxed answers. '
           'Both target alphas beat low-temperature sampling at pass@1, but neither beats it clearly at answer pass@4. '
           'Effective alpha stays near 1; raising the target to 4 does not resolve this. '
           'All methods, including MCMC, are compared on the same 500 questions below.**')
    method=('## Shared protocol\n\n'
            'Qwen2.5-0.5B on the same 1,024 GSM8K training questions, seed 0, physical GPU 1. '
            'Independent runs start from identical base/LoRA initialization; the settings differ only in target alpha (2 or 4). '
            'The cap follows **64 → 128 → 192 → 256 → 320 → 384 → 448 → 512**, '
            f"with **{s['updates_per_stage']} updates and {s['planned_rollouts_per_stage']:,} rollouts per stage** "
            f"(304 updates, {s['planned_total_rollouts']:,} responses per run). "
            f"Training including diagnostics took **{completion['wall_seconds']/60:.1f} / {completion4['wall_seconds']/60:.1f} minutes** "
            'for alpha 2 / 4. The quota was calibrated once and reused.\n\n'
            'All-linear LoRA: rank 8, scaling alpha 16, dropout 0. AdamW: lr 1e-5, weight decay 0, '
            'gradient clipping 1, no warmup. Each update uses 16 questions × 4 responses, generation batch 64, '
            'scoring microbatch 4, BF16, T=1 and unrestricted token sampling. '
            r'The reward is $r=\alpha\log p_0-\log\pi$, with a leave-one-out baseline and one update per fresh rollout batch. '
            'Answer labels are used only for evaluation. EOS contributes to sequence scores; unfinished responses are censored at the cap.\n\n'
            '```text\n'+s['prompt_template']+'\n```\n\n'
            'The literal double boxed braces match the historical GSM8K experiment. '
            '[Alpha-2 settings](settings.json) · [Alpha-4 settings](settings_alpha4.json).')
    training=('## Training trajectories\n\n'
              'Each point uses the same **16 held-out questions × 4 fresh current-policy responses**, T=1 and a fixed 512-token cap. '
              'Time includes diagnostics; curves connect recorded points without smoothing. Probe accuracy estimates pass@1 '
              'on this small diagnostic set, separately from the final test evaluation.\n\n'
              '![Policy NLL, reference CE and combined true loss](figures/loss_decomposition.png)\n\n'
              r'Policy NLL is $E_\pi[-\log\pi]$, which estimates policy entropy; reference CE is $E_\pi[-\log p_0]$. '
              r'The third subplot combines them as $L=\alpha\,\mathrm{CE}-\mathrm{NLL}=E_\pi[\log\pi-\alpha\log p_0]$. '
              'All three use the same on-policy samples. Values are absolute nats per response, without length normalization; '
              'CE is shown before multiplication by alpha. Error bars are pointwise 1.96 × question SE. '
              'True loss is reverse KL up to a constant at fixed alpha and horizon. Different alphas define different targets, '
              'so their absolute true losses are not a common KL scale. '
              '[Decomposed values and source hashes](results/loss_decomposition.json).\n\n'
              '![Effective alpha, accuracy and response length](figures/alpha_comparison.png)\n\n'
              'Effective-alpha bars show question-slope SD, not confidence intervals or intrinsic temperature noise.\n\n')
    endpoints=[]
    for alpha,records in [(2,probes),(4,probes4)]:
        first,last=records[0],records[-1]
        endpoints.append(f"Target alpha {alpha}: true loss **{first['true_loss']['mean']:.1f} → {last['true_loss']['mean']:.1f}**; "
                         f"final effective alpha **{last['on_policy']['alpha']:.3f}**, question SD **{last['on_policy']['question_slope_std']:.3f}** "
                         f"(fixed-response estimate {last['fixed_responses']['alpha']:.3f}).")
    training+=' '.join(endpoints)+' Both losses fall, but neither diagnostic supports learning the target power exponent.'
    training+=('\n\n'
               r'The similar NLL/CE shapes are not identical values: $\mathrm{CE}=\mathrm{NLL}+D_{KL}(\pi\Vert p_0)$. '
               'Both start at the same value because the policy initially equals the base. Their shared early decrease dominates '
               'the plot scale; the final CE minus NLL estimates are '
               f"**{decomposition['2']['points'][-1]['base_kl']['mean']:.2f} / {decomposition['4']['points'][-1]['base_kl']['mean']:.2f} nats** "
               'for target alpha 2 / 4. The first two panels use identical axis limits. '
               'In the true loss, policy NLL enters with a minus sign.')
    audit_path=HERE/'results/loss_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if audit['passed']:
            worst=max(c['max_absolute_nats'] for r in audit['runs'].values() for c in r['checks'].values())
            training+=(' Independent rescoring of all 128 final probe responses with full logits and a separately loaded base '
                       f"agrees with the saved scores to within {worst:.3f} nat per response. "
                       'No duplicated-array or adapter-switch error was found in these checks. '
                       '[Scoring audit](results/loss_audit.json).')
    c4=alpha4_summary['paired_comparisons']
    full=('## Final evaluation: 500 questions\n\n'
          'Same test questions, 32 independent responses per question and a 512-token cap. Policies use T=1. '
          'Answer grading accepts explicit final answers; boxed grading also requires the requested format. '
          'Low T=0.25 was selected on the historical development set.\n\n'
          +table(evaluated,paths)+'\n\n'
          'Against low T, answer pass@1 gains are '+interval(comparisons['base_t0.25']['answer']['1'])+' for alpha 2 and '
          +interval(c4['base_t0.25']['answer']['1'])+' for alpha 4. Both answer pass@4 difference intervals cross zero. '
          'Alpha 4 versus alpha 2 has no clear pass@1 or pass@4 advantage; its pass@32 difference is '
          +interval(c4['adapt_len_alpha2']['answer']['32'])+'.\n\n'
          'These are 10,000-replicate paired-question bootstrap intervals. One training seed does not measure training variability. '
          'The historical fixed-length run had fewer updates and different diagnostic overhead/scoring kernels; '
          'its comparison does not isolate the causal effect of the curriculum. '
          '[All metrics and intervals](results/summary.json).')
    m2=comparisons['mcmc_alpha2']
    m4=c4['mcmc_alpha2']
    cost=evaluated['mcmc_alpha2']['generated_tokens']
    policy_costs=[evaluated[f'adapt_len_alpha{a}']['generated_tokens'] for a in [2,4]]
    mcmc=('## MCMC interpretation and cost\n\n'
          'The GSM8K baseline MCMC evaluation is complete on the same **500 questions × 32 chains** used in the table above. '
          'MCMC targets alpha 2 with proposal T=0.5, block size 32 and two MH updates per block. '
          'Model revision, prompt, engine settings, seed rule and 512-token cap match the original evaluation. '
          'It uses the local full-suffix acceptance ratio and retains EOS; it is a finite-compute baseline, '
          'not a verified exact power sampler or exact upstream reproduction. No matching-alpha-4 MCMC result is available here.\n\n'
          'Answer pass@4, policy minus MCMC: '+interval(m2['answer']['4'])+' for alpha 2 and '
          +interval(m4['answer']['4'])+' for alpha 4. '
          'Answer pass@1 differences are '+interval(m2['answer']['1'])+' / '+interval(m4['answer']['1'])+' respectively.\n\n'
          f'MCMC generated **{cost:,} tokens**, including rejected proposals, versus **{policy_costs[0]:,} / {policy_costs[1]:,}** '
          f'for the alpha-2 / alpha-4 policies ({cost/policy_costs[0]:.2f}× / {cost/policy_costs[1]:.2f}×). '
          'These counts exclude reference scoring and training; compute is not matched. '
          '[Unified metrics, source hashes and intervals](results/summary.json) · '
          '[Original MCMC completion](../gsm8k_power_policy/results/evaluation/test/mcmc_alpha2/completion.json).')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(t)
                                     for t in [intro,method,training,full,mcmc]])
    nbformat.validate(nb)
    text='\n'.join(c.source for c in nb.cells)
    assert text.count('| Method |')==1 and '336' not in text
    nbformat.write(nb,HERE/'report.ipynb')



if __name__=='__main__':
    main()
