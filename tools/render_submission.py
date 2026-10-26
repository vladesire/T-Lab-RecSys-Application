#!/usr/bin/env python3
"""Generate paper sources, numerical tables and scientific figures from explicit JSON.

Root locked environment. No raw data, model fitting, inference or LLM call.
The article and figures are generated from the explicitly selected evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS={'original':'Original','no_pair':'No pair','manual':'Manual','decay':'Decay',
        'all':'All features','forward':'Automatic','random_search':'Matched search'}
ORDER=list(LABELS)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def tex(s):
    mapping={'_':r'\_','&':r'\&','%':r'\%','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in str(s))
def fmt(x):return f'{x:.6f}'
def interval(x):
    return f"{x['delta_ndcg10']:+.6f} [{x['percentile_95'][0]:+.6f}, {x['percentile_95'][1]:+.6f}]"

def render(root, study_path, final_path, analysis, output, author='Vladislav Neklydov', reading_output=None):
    study,final=read(study_path),read(final_path)
    if study['readiness']['ready'] is not True or final.get('test_scored') is not True:
        raise ValueError('Require completed development and genuine final evaluation')
    for protocol in ['P20','U20']:
        for method in ORDER:
            assert method in study['protocols'][protocol]['methods']
            assert method in final['protocols'][protocol]['methods']
    manifest=read(analysis/'manifest.json')
    for f in manifest['files']:
        p=analysis/f['path']
        if p.stat().st_size!=f['size_bytes'] or sha(p)!=f['sha256']:raise ValueError('Selected analysis inventory mismatch')
    output.mkdir(parents=True,exist_ok=True)
    for p,n in [(study_path,'study.json'),(final_path,'final.json')]:
        if p.resolve()!=(output/n).resolve():shutil.copyfile(p,output/n)
    freeze_path=final_path.parent/'freeze.json'
    if freeze_path.exists():
        if sha(freeze_path)!=final['freeze_sha256']:raise ValueError('Final freeze mismatch')
        shutil.copyfile(freeze_path,output/'freeze.json')
    identities={'study_sha256':sha(study_path),'final_sha256':sha(final_path),
                'final_freeze_sha256':final['freeze_sha256'],'analysis_manifest_sha256':sha(analysis/'manifest.json'),
                'renderer_sha256':sha(Path(__file__)), 'template_sha256':sha(root/'paper/templates/article.tex')}
    mac=root/'evidence/mac-search/analysis.json'
    if mac.is_file():
        owner=read(mac);current=read(analysis/'analysis.json')
        key=lambda x:(x['series_id'],x['metric'],x['subgroup'],x['weighting'])
        cells={key(x):x['value'] for x in current['metrics']}
        differences=[abs(x['value']-cells[key(x)]) for x in owner['metrics'] if key(x) in cells and x['value'] is not None and cells[key(x)] is not None]
        shutil.copyfile(mac,output/'owner-mac-analysis.json')
        identities['owner_mac_analysis_sha256']=sha(mac)
    (output/'evidence.json').write_text(json.dumps(identities,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
    def save(fig,name):
        fig.savefig(output/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(output/(name+'.png'),dpi=160)
        plt.close(fig)
    methods=['manual','decay','all','forward','random_search']
    fig,axes=plt.subplots(1,2,figsize=(7.0,3),sharey=True)
    for ax,protocol in zip(axes,['P20','U20']):
        dev=[study['protocols'][protocol]['methods'][m]['metrics']['V_check']['ndcg10'] for m in methods]
        test=[final['protocols'][protocol]['methods'][m]['metrics']['all']['ndcg10'] for m in methods]
        x=np.arange(len(methods));ax.bar(x-.18,dev,.36,label='Later development',color='#3d729b')
        ax.bar(x+.18,test,.36,label='Temporal test',color='#bf783e')
        ax.set_xticks(x,[LABELS[m] for m in methods],rotation=25,ha='right');ax.set_title(protocol)
        ax.set_ylim(0,1);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Event NDCG@10');axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    fig.tight_layout();save(fig,'quality')

    comparisons=['forward-minus-manual','all-minus-manual','decay-minus-manual']
    fig,ax=plt.subplots(figsize=(7.0,2.8))
    for offset,source,color,label,key in [(-.12,study['protocols']['P20'],'#3d729b','Later development','V_check'),
                                         (.12,final['protocols']['P20'],'#bf783e','Temporal test','all')]:
        records=[source['comparisons'][c+'/'+key] for c in comparisons]
        x=np.array([r['delta_ndcg10'] for r in records]);low=np.array([r['percentile_95'][0] for r in records]);high=np.array([r['percentile_95'][1] for r in records])
        ax.errorbar(x,np.arange(3)+offset,xerr=[x-low,high-x],fmt='o',capsize=3,label=label,color=color)
    ax.axvline(0,color='.4',lw=.8);ax.set_yticks(np.arange(3),['Automatic - manual','All - manual','Decay - manual'])
    ax.set_xlabel('P20 event NDCG@10 difference; paired 95% user-cluster interval')
    ax.invert_yaxis();ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.02));ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'effects')

    forward=study['protocols']['P20']['methods']['forward'];random=study['protocols']['P20']['methods']['random_search']
    ftr=forward['trials'];rtr=random['trials']
    same={tuple(sorted(t['groups'])) for t in ftr}=={tuple(sorted(t['groups'])) for t in rtr}
    fig,ax=plt.subplots(figsize=(7.0,2.6))
    for trials,color,label in [(ftr,'#3d729b','Forward candidates'),(rtr,'#bf783e','Matched candidates')]:
        ax.plot([t['number'] for t in trials],[t['V_search_ndcg10'] for t in trials],'o-',color=color,label=label)
    baseline=study['protocols']['P20']['methods']['manual']['metrics']['V_search']['ndcg10']
    ax.axhline(baseline,color='.3',ls='--',label='Manual baseline');ax.set_xlabel('Candidate fit in its recorded order')
    ax.set_ylabel('Selection NDCG@10');ax.legend(frameon=False,fontsize=8,ncol=3,loc='lower left',bbox_to_anchor=(0,1.02));ax.grid(alpha=.15);fig.tight_layout();save(fig,'search')

    fig,ax=plt.subplots(figsize=(7.0,2.6));x=np.arange(3)
    for i,m in enumerate(['original','no_pair','manual']):
        vals=[final['protocols']['P20']['methods'][m]['metrics'][s]['ndcg10'] for s in ['all','repeat','novel']]
        ax.bar(x+(i-1)*.24,vals,.24,label=LABELS[m],color=['#3d729b','#bf783e','#61947e'][i])
    ax.set_xticks(x,['All targets','Repeat targets','Novel targets']);ax.set_ylim(0,1)
    ax.set_ylabel('P20 test NDCG@10');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'repetition')

    def quality_table(protocol):
        rows=[]
        for method in ORDER:
            d=study['protocols'][protocol]['methods'][method];t=final['protocols'][protocol]['methods'][method]
            rows.append(f"{LABELS[method]} & {t['feature_count']} & {fmt(d['metrics']['V_search']['ndcg10'])} & {fmt(d['metrics']['V_check']['ndcg10'])} & {fmt(t['metrics']['all']['ndcg10'])} & {fmt(t['metrics']['all']['recall10'])} \\\\")
        return r'\begin{center}\small\begin{tabular}{lrrrrr}\toprule Method & Features & Search & Check & Test & Recall\\\midrule'+'\n'+'\n'.join(rows)+r'\bottomrule\end{tabular}\end{center}'
    tables=r'\subsection*{P20: popularity-weighted alternatives}'+quality_table('P20')+r'\subsection*{U20: uniform alternatives}'+quality_table('U20')
    (output/'quality-tables.tex').write_text(tables+'\n')
    rows=[]
    for label,source,slice_ in [('Check',study['protocols']['P20'],'V_check'),('Test',final['protocols']['P20'],'all')]:
        for comp in comparisons:
            r=source['comparisons'][comp+'/'+slice_];lo,hi=r['percentile_95']
            rows.append(f"{label} & {tex(comp.replace('_',' ').replace('-minus-',' - '))} & {r['delta_ndcg10']:+.6f} & {lo:+.6f} & {hi:+.6f} \\\\")
    (output/'effect-table.tex').write_text(r'\begin{center}\small\begin{tabular}{llrrr}\toprule Slice & Difference & Estimate & Lower & Upper\\\midrule'+'\n'+'\n'.join(rows)+r'\bottomrule\end{tabular}\end{center}'+'\n')
    costs=[]
    for m in ['manual','all','forward','random_search']:
        r=study['protocols']['P20']['methods'][m]
        cost=r.get('candidate_fit_seconds') if m in {'forward','random_search'} else r.get('fit_seconds')
        costs.append(f"{LABELS[m]} & {r['job_wall_seconds']:.3f} & {cost:.3f} & {r['sampled_peak_tree_rss_bytes']/1024**3:.3f} \\\\")
    (output/'cost-table.tex').write_text(r'\begin{center}\small\begin{tabular}{lrrr}\toprule Method & Job wall (s) & Fit(s) & Tree RSS (GiB)\\\midrule'+'\n'+'\n'.join(costs)+r'\bottomrule\end{tabular}\end{center}'+'\n')
    dp=study['protocols']['P20'];tp=final['protocols']['P20'];primary=tp['comparisons']['forward-minus-manual/all']
    lo,hi=primary['percentile_95']
    conclusion=('The interval supports a positive effect on this holdout.' if lo>0 else
                'The interval supports a negative effect on this holdout.' if hi<0 else
                'The interval spans zero: these data do not establish a gain or loss in event-average ranking.')
    if lo>-.005:conclusion+=' Its lower endpoint exceeds the chosen -0.005 tolerance, supporting that limited tolerance claim here.'
    if hi<.005:conclusion+=' A gain of 0.005 is outside the reported interval.'
    groups=', '.join(forward['selected_groups']) or 'none (manual baseline retained)'
    b=study['readiness'];matches='Both searches cover the same six one-group additions in different orders. Their agreement is expected under this exhaustive one-round comparison; it does not distinguish optimizer quality.' if same else 'The matched search uses the recorded group-depth schedule and an equal candidate-fit budget.'
    selected='The automatic procedure selected '+tex(groups)+f", giving {len(forward['features'])} features."
    if forward['selected_groups']==['user_item_decay_all']:
        selected+=' It adds four pair-decay counts and four smoothed pair/user decay affinities at 1/7/30/90-day half-lives; it does not learn an unrestricted window length.'
    body=(root/'paper/templates/article.tex').read_text()
    novel=tp['comparisons']['no_pair-minus-original/novel'];all_=tp['comparisons']['no_pair-minus-original/all']
    replacements={'AUTHOR':tex(author),'PRIMARYABSTRACT':f"On the frozen P20 holdout the automatic-minus-manual NDCG difference is {interval(primary)} (paired 95\\% user-cluster interval).",
        'SELECTED':selected,'MATCHES':matches,'PRIMARYRESULT':tex(interval(primary)),'CONCLUSION':conclusion,
        'COUNTSTEXT':f"The main development panel has {dp['queries']:,} queries from {dp['users']} evaluated users. The holdout has {tp['queries']:,} queries from {tp['users']} users, including {tp['novel_queries']:,} novel-target queries. Eligible evaluated-user counts are reported separately for each interval.",
        'REPETITIONTEXT':f"On P20 test data, no-pair-minus-original is {tex(interval(all_))} overall and {tex(interval(novel))} on novel targets. All intervals here use event weighting."}
    for k,v in replacements.items():body=body.replace(k,v)
    (output/'main.tex').write_text(body)
    reading=reading_output or root/'reports/submission'
    reading.mkdir(parents=True,exist_ok=True)
    def link(path):return Path(os.path.relpath(path,reading)).as_posix()
    rows=[]
    for m in ORDER:
        v=dp['methods'][m];t=tp['methods'][m]
        rows.append(f"| {LABELS[m]} | {t['feature_count']} | {v['metrics']['V_check']['ndcg10']:.6f} | {t['metrics']['all']['ndcg10']:.6f} | {t['metrics']['all']['recall10']:.6f} |")
    summary=f'''# Reading the study

Can automatic temporal feature groups compete with a manual gradient-boosting scorer?

Read [the paper](main.pdf), then inspect the [numerical evidence]({link(final_path)}),
[search trace and development slices]({link(study_path)}) and [research decisions]({link(root/'research/decisions')}).


## Methods

One nested 1,000-user cohort with complete retained histories; fixed training/model/candidate rules;
six automatic one-group additions selected on earlier P20 development; later development used for checking;
fitted finalists frozen before one temporal holdout evaluation. U20 is a protocol sensitivity condition.
Earlier baseline and Linux/Mac reproductions reuse data and are not independent replications.

The automatic procedure selected **{len(forward['features'])} features**; added groups: `{groups}`.
Its P20 final automatic-minus-manual event NDCG@10 is **{interval(primary)}**.
{conclusion}

## P20 results

| Method | Features | Later development NDCG@10 | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Development: {dp['queries']:,} queries / {dp['users']} evaluated users. Test: {tp['queries']:,} queries /
{tp['users']} users, including {tp['novel_queries']:,} novel queries. Cohort size and evaluated-user counts differ.
The test is [23,400,000, 26,000,000] in recorded seconds. All reported test models retain their train-only fits.

![Quality by protocol]({link(output/'quality.png')})

![P20 paired effects]({link(output/'effects.png')})

## Interpretation and limitations

- Automatic selection chose an actual temporal group; its quality is judged on check/test, not its selection maximum.
- {matches}
- No-pair-minus-original on P20 test: {interval(all_)} overall; {interval(novel)} on novel targets.
- Fewer columns do not establish lower feature-construction cost: all 47 columns are built upstream.
- One small, development-adaptive study; sampled candidates, training-catalog eligibility and retained-cohort counts.
  This is not a full-catalog leaderboard or an exposure-corrected causal result.

## How to inspect or reproduce

- [Run, analysis and freeze identities](provenance.md).
- [Reproduction commands]({link(root/'docs/reproduction.md')}): figures/paper from saved evidence, or full raw-data computation.
- [Research hypotheses and decisions]({link(root/'research')}).
'''
    (reading/'README.md').write_text(summary)
    provenance=f'''# Selected evidence and provenance

- Development run: `{study['run_id']}`; parent run manifest SHA256 `{study['parent_run_manifest_sha256']}`.
- Native subset export SHA256: `{sha(study_path)}`. [Exact JSON]({link(study_path)}).
- Frozen holdout result SHA256: `{sha(final_path)}`. [Exact JSON]({link(final_path)}).
- Final freeze SHA256: `{final['freeze_sha256']}`. Models/features were frozen before scoring.
- Selected complete generic analysis: `{analysis.name}`; manifest SHA256 `{sha(analysis/'manifest.json')}`.
- Renderer SHA256: `{sha(Path(__file__))}`. [Generation code]({link(root/'tools/render_submission.py')}).
- Standalone final evaluator: [code]({link(root/'tools/evaluate_frozen_final.py')}); six boundary tests in
  [test_final_holdout.py]({link(root/'tests/reproduction/test_final_holdout.py')}).

The native export independently recomputes saved per-query metrics and exact subset means; paired check intervals
use the native user-cluster bootstrap. The final evaluator loads frozen models, generates identical candidates
within protocol, computes strictly earlier history features, checks original columns by direct masks and scores
the reserved temporal interval once. It is a separate scientific stage; M19 remains development-only.

The generic analysis supports whole-development fixed comparisons and includes selection rows.
Its selected-search whole-development scores are not the headline out-of-selection result.
Full analysis per-query records and final models/predictions are included under `runs/published/`; raw data and large
feature matrices stay out of Git. Compact JSON exports are not relabelled as complete native runs.

'''
    (reading/'provenance.md').write_text(provenance)
    return {'paper_source':str(output/'main.tex'),'evidence':identities,'matched_search_enumerates_same_candidates':same}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--study',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('paper/submission'));p.add_argument('--author',default='Vladislav Neklydov');p.add_argument('--reading-output',type=Path)
    a=p.parse_args();r=a.root.resolve();print(json.dumps(render(r,(r/a.study).resolve(),(r/a.final).resolve(),(r/a.analysis).resolve(),(r/a.output).resolve(),a.author,(r/a.reading_output).resolve() if a.reading_output else None),indent=2))
