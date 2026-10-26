#!/usr/bin/env python3
"""Freeze completed development finalists, then evaluate a single temporal holdout.

Run with engines/tlab_warmup/.venv/bin/python. This separate scientific stage
does not change M19, its development bundles, or its final-scoring prohibition.
No estimator is trained or selected here. Recorded strictly earlier events,
including earlier holdout events, are available to online history features.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

METHODS = ('original', 'no_pair', 'manual', 'decay', 'all', 'forward', 'random_search')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        if Path(name).exists():Path(name).unlink()

def read(path):
    return json.loads(Path(path).read_text())

def numeric_environment():
    packages={}
    for name in ['numpy','scikit-learn','scipy','pyarrow','joblib']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='missing'
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':packages}

def setup(root):
    sys.path.insert(0, str(root / 'engines/tlab_warmup/src'))
    from tlab_warmup.experiment_io import code_state
    return code_state(root / 'engines/tlab_warmup')['tree_sha256']

def relocated(path, run):
    token = '/payload/workspace/'
    if token not in str(path):
        raise ValueError('Native dependency is outside the isolated workspace')
    suffix = str(path).split(token, 1)[1]
    if '..' in Path(suffix).parts:
        raise ValueError('Unsafe native path')
    return run / 'payload/workspace' / suffix

def freeze(root, run, output, cap=20000, replicates=1000):
    from tlab_warmup.experiment_io import valid_output
    code = setup(root)
    native = run / 'payload/workspace/runs' / run.name
    if read(run/'manifest.json').get('status')!='succeeded':
        raise ValueError('Parent infrastructure run must have succeeded')
    ledger = read(native / 'ledger.json')
    ready = read(native / 'readiness.json')
    if ready.get('code_hash') != code or not ready.get('ready') or ready.get('test_scored'):
        raise ValueError('Need a completed, compatible, unscored development run')
    if cap < 1 or replicates < 100:
        raise ValueError('Invalid prespecified sample/bootstrap cap')
    refs, models = {}, []
    for key in ['data_main', 'prepare_P20', 'prepare_U20']:
        p = relocated(ledger[key]['output'], run)
        if not valid_output(p):
            raise ValueError(f'Incomplete native dependency: {key}')
        refs[key] = str(p.relative_to(root))
    for protocol in ['P20', 'U20']:
        for method in METHODS:
            key = protocol + '_' + method
            if protocol == 'U20' and method in {'forward', 'random_search'}:
                key += '_frozen'
            job = ledger[key]
            p = relocated(job['output'], run)
            if job['status'] != 'completed' or not valid_output(p):
                raise ValueError(f'Finalists must all complete before test: {key}')
            result = read(p / 'result.json')
            names = result.get('feature_names', result.get('selected_features'))
            if not names or not (p / 'model.joblib').is_file():
                raise ValueError(f'Missing actual fitted finalist: {key}')
            models.append({'protocol':protocol, 'method':method,
                           'model':str((p/'model.joblib').relative_to(root)),
                           'model_sha256':sha(p/'model.joblib'), 'features':names,
                           'selection_result_sha256':sha(p/'result.json')})
    meta=read(root/refs['data_main']/'result.json')
    f = {'kind':'frozen_temporal_holdout_v1', 'created_at':datetime.now(timezone.utc).isoformat(),
         'parent_run':str(run.relative_to(root)), 'parent_run_manifest_sha256':sha(run/'manifest.json'),
         'engine_code_sha256':code, 'evaluator_sha256':sha(Path(__file__)),
         'engine_environment':numeric_environment(),
         'dataset':meta['config']['source'],'boundaries':meta['config']['resolved_boundaries'],
         'dependency_complete_sha256':{k:sha(root/v/'COMPLETE.json') for k,v in refs.items()},
         'dependencies':refs, 'models':models, 'query_cap':cap,
         'bootstrap_replicates':replicates, 'query_sampling_seed_offset':8831,
         'primary_comparison':'P20 forward-minus-manual, event NDCG@10',
         'secondary_comparisons':['random_search-minus-manual','all-minus-manual','decay-minus-manual','no_pair-minus-original'],
         'model_policy':'Load frozen train-only fits; no holdout-driven fitting, selection or refitting',
         'history_policy':'Prequential recorded history s<t; earlier test events become available, target and equal-time events excluded',
         'output':str(output.relative_to(root))}
    output.mkdir(parents=True, exist_ok=False)
    write(output/'freeze.json', f)
    return output/'freeze.json'

def eligible_holdout(events, types, config, start, end):
    """Same training-catalog/user/strict-history exclusions as development."""
    train_end = config['train_end_timestamp']
    if not train_end < start <= end or np.any(events[1:,2] < events[:-1,2]):
        raise ValueError('Invalid chronology')
    past = events[events[:,2] < train_end]
    ids, first_idx = np.unique(past[:,1], return_index=True)
    first = dict(zip(map(int,ids), map(int,past[first_idx,2])))
    first_times = np.sort(np.asarray(list(first.values())))
    user_order = np.argsort(events[:,0], kind='stable')
    users, offsets, sizes = np.unique(events[user_order,0], return_index=True, return_counts=True)
    histories = {int(u):events[user_order[s:s+n],2] for u,s,n in zip(users,offsets,sizes)}
    frozen = {u for u,t in histories.items() if np.searchsorted(t,train_end,'left') >= config['minimum_prior_user_events']}
    pairs, counts = np.unique(events[:,[0,2]], axis=0, return_counts=True)
    ambiguous = {tuple(map(int,p)) for p,n in zip(pairs,counts) if n>1}
    pool, exclusions = [], Counter()
    for row_id in np.flatnonzero((events[:,2]>=start)&(events[:,2]<=end)&(types==config['target_event_type'])):
        u,it,t = map(int,events[row_id])
        reason = None
        if (u,t) in ambiguous: reason='ambiguous_user_timestamp'
        elif np.searchsorted(histories[u],t,'left') < config['minimum_prior_user_events']: reason='insufficient_prior_history'
        elif u not in frozen: reason='training_ineligible_user'
        elif it not in first: reason='outside_training_catalog'
        elif first[it] >= t: reason='target_not_previously_observed'
        elif np.searchsorted(first_times,t,'left') < config['alternatives']+1: reason='insufficient_candidate_pool'
        if reason: exclusions[reason]+=1
        else: pool.append({'query_id':int(row_id),'uid':u,'timestamp':t,'positive':it,'split':'test','anchor':False})
    return pool, dict(exclusions)

def execute(root, freeze_path):
    f = read(freeze_path); output=root/f['output']
    if (output/'STARTED.json').exists():
        raise ValueError('Holdout execution already started; preserve it, do not select another test sample')
    if setup(root)!=f['engine_code_sha256'] or sha(Path(__file__))!=f['evaluator_sha256']:
        raise ValueError('Code changed after final freeze')
    if f.get('engine_environment') and numeric_environment()!=f['engine_environment']:
        raise ValueError('Numerical environment changed after final freeze')
    if sha(root/f['parent_run']/'manifest.json')!=f['parent_run_manifest_sha256']:
        raise ValueError('Parent run manifest changed')
    for m in f['models']:
        if sha(root/m['model'])!=m['model_sha256']:
            raise ValueError('Fitted model changed after final freeze')
        if sha((root/m['model']).parent/'result.json')!=m['selection_result_sha256']:
            raise ValueError('Feature-selection evidence changed')
    from tlab_warmup.experiment_io import load_queries, save_queries
    from tlab_warmup.representations import manifest, build_dictionary
    from tlab_warmup.sampling import candidate_queries
    from tlab_warmup.evaluation import per_query, aggregate, paired_interval
    from tlab_warmup.experiment_jobs import model_scores
    from tlab_warmup.core import brute_features
    import joblib
    import pyarrow.parquet as pq
    from tlab_warmup.experiment_io import valid_output
    for key,digest in f['dependency_complete_sha256'].items():
        dependency=root/f['dependencies'][key]
        if sha(dependency/'COMPLETE.json')!=digest or not valid_output(dependency):
            raise ValueError('Native preparation changed after final freeze')
    # Exclusive creation makes concurrent invocations refuse the second start.
    with (output/'STARTED.json').open('x') as stream:
        json.dump({'started_at':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(freeze_path)},stream)
    started=time.perf_counter()
    data=root/f['dependencies']['data_main'];meta=read(data/'result.json');c=meta['config']
    info=read(data/'source_and_extraction.json')
    run=root/f['parent_run'];work=run/'payload/workspace'
    candidates=list((work/'data'/c['source']['revision']).glob('users-*.parquet'))
    selected=[p for p in candidates if sha(p)==info['selected_sha256']]
    if len(selected)!=1: raise ValueError('Missing exact full-history extraction cache')
    table=pq.read_table(selected[0],columns=['uid','item_id','timestamp','event_type'])
    events=np.column_stack([table[x].to_numpy().astype(np.int64) for x in ['uid','item_id','timestamp']])
    types=np.asarray(table['event_type'].to_pylist())
    order=np.lexsort((events[:,1],events[:,0],events[:,2]));events,types=events[order],types[order]
    if not np.array_equal(events,np.load(data/'events.npy',mmap_mode='r')):
        raise ValueError('Full-history event identity changed')
    start=c['development_end_timestamp'];end=c['resolved_boundaries']['reserved_test_end_inclusive']
    pool, exclusions=eligible_holdout(events,types,c,start,end)
    rng=np.random.default_rng(c['seed']+f['query_sampling_seed_offset'])
    idx=np.sort(rng.choice(len(pool),min(f['query_cap'],len(pool)),replace=False))
    targets=[pool[int(i)] for i in idx]
    if len(targets)<100: raise ValueError('Insufficient final targets')
    results={'kind':'temporal_holdout_result_v1','test_scored':True,'freeze_sha256':sha(freeze_path),
             'interval':[start,end],'queries':len(targets),'users':len({q['uid'] for q in targets}),
             'eligible_pool':len(pool),'excluded':exclusions,'protocols':{},
             'limitations':['One development-adaptive study; finite sampled catalog, no exposure correction',
                 'Training-defined catalog and training-eligible users; complete observed histories',
                 'Online earlier test events are available to features, with strict s<t',
                 'Models trained before train_end, selected on development; no train+development refit',
                 'User-cluster intervals condition on this cohort and candidate design; no multiplicity adjustment']}
    for protocol in ['P20','U20']:
        print('Final protocol:',protocol,flush=True)
        out=output/protocol;out.mkdir()
        queries=candidate_queries(events,targets,c['train_end_timestamp'],c['seed'],protocol)
        save_queries(out/'queries.json.gz',queries)
        spec=build_dictionary(events,queries,out/'features.npy',day_seconds=c['window_seconds']/7)
        features=np.load(out/'features.npy',mmap_mode='r')
        cols=[spec['names'].index(n) for n in spec['original']]
        for j in np.linspace(0,len(queries)-1,5,dtype=int):
            np.testing.assert_allclose(features[j*21:(j+1)*21,cols],brute_features(events,queries[j],c['window_seconds']),rtol=1e-12,atol=1e-12)
        positive=np.array([q['positive'] for q in queries]);items=np.array([q['candidates'] for q in queries])
        repeat=(features[:,spec['names'].index('user_item_count')].reshape(-1,21)[items==positive[:,None]]>0)
        record={'queries':len(queries),'users':len({q['uid'] for q in queries}),
                'repeat_queries':int(repeat.sum()),'novel_queries':int((~repeat).sum()),'methods':{},'comparisons':{}}
        values={}
        for m in [m for m in f['models'] if m['protocol']==protocol]:
            t=time.perf_counter();model=joblib.load(root/m['model'])
            scores=model_scores(model,features,0,m['features'],len(queries),c['threads'])
            v=per_query(queries,scores,c['seed']);values[m['method']]=v
            np.savez_compressed(out/(m['method']+'.npz'),scores=scores,per_query_metrics=v,repeat=repeat,query_ids=np.array([q['query_id'] for q in queries]))
            record['methods'][m['method']]={'feature_count':len(m['features']),'application_and_scoring_seconds':time.perf_counter()-t,
                'metrics':aggregate(queries,v,repeat,start)}
        for left,right in [('forward','manual'),('random_search','manual'),('all','manual'),('decay','manual'),('no_pair','original')]:
            for subgroup,mask in [('all',np.ones(len(queries),bool)),('repeat',repeat),('novel',~repeat)]:
                record['comparisons'][left+'-minus-'+right+'/'+subgroup]=paired_interval(queries,values[left],values[right],mask,c['seed'],f['bootstrap_replicates'])
        results['protocols'][protocol]=record
        write(out/'summary.json',record)
        # The dictionary is recomputable; do not ship a large matrix in evidence.
        del features
    results['wall_seconds']=time.perf_counter()-started
    results['completed_at']=datetime.now(timezone.utc).isoformat()
    write(output/'result.json',results)
    inventory={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(output/'manifest.json',{'kind':'standalone_temporal_holdout_v1','files':inventory})
    return output/'result.json'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--run',type=Path);p.add_argument('--output',type=Path);p.add_argument('--freeze',type=Path)
    a=p.parse_args();root=a.root.resolve();setup(root)
    if a.freeze:
        print(execute(root,(root/a.freeze).resolve()))
    elif a.run and a.output:
        print(freeze(root,(root/a.run).resolve(),(root/a.output).resolve()))
    else:p.error('Use --run and --output to freeze, or --freeze to execute once')

if __name__=='__main__':main()
