#!/usr/bin/env python3
"""Export exact development slices, selection traces and paired V_check results.

Use the engine Python. No selection/fitting happens here. This explicit native
slice exporter complements the generic analyzer, which has no V_check selector.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

from evaluate_frozen_final import setup, relocated, read, sha, write

def collect(root, run, output):
    setup(root)
    from tlab_warmup.experiment_io import load_queries, valid_output
    from tlab_warmup.evaluation import paired_interval, per_query, aggregate
    native=run/'payload/workspace/runs'/run.name
    ledger=read(native/'ledger.json')
    result={'kind':'submission_development_export_v1','run_id':run.name,
            'parent_run_manifest_sha256':sha(run/'manifest.json'),
            'readiness':read(native/'readiness.json'),'protocols':{},
            'selection_policy':'P20 V_search only, manual baseline plus six groups; fixed eight-fit cap and one round',
            'evaluation_policy':'V_check is later development, not final test; main development includes selection rows',
            'source_files':{'ledger.json':sha(native/'ledger.json')}}
    for protocol in ['P20','U20']:
        prep=relocated(ledger['prepare_'+protocol]['output'],run)
        if not valid_output(prep):raise ValueError('Incomplete native preparation')
        meta=read(prep/'result.json');queries=[q for q in load_queries(prep/'queries.json.gz') if q['split']=='development']
        item={'queries':len(queries),'users':len({q['uid'] for q in queries}),'midpoint':meta['midpoint'],
              'methods':{},'comparisons':{},'sampling_seconds':meta['sampling_seconds'],'feature_seconds':meta['feature_seconds']}
        ts=np.array([q['timestamp'] for q in queries]);vals={};rep=None
        for key,job in ledger.items():
            if job['status']!='completed' or job['request']['kind'] not in {'fit','search'}:continue
            if job['request'].get('protocol_result')!=ledger['prepare_'+protocol]['output']:continue
            path=relocated(job['output'],run)
            if not valid_output(path):raise ValueError('Incomplete method output: '+key)
            j=read(path/'result.json');method=j['method']
            with np.load(path/'predictions.npz') as a:
                scores=a['scores'];v=a['per_query_metrics'];repeat=a['repeat']
                if not np.array_equal(a['query_ids'],[q['query_id'] for q in queries]):raise ValueError('Query mismatch')
                recomputed=per_query(queries,scores,j['seed'])
                np.testing.assert_allclose(v,recomputed,rtol=0,atol=1e-14)
            metrics=aggregate(queries,v,repeat,meta['midpoint'])
            if metrics!=j['metrics']:raise ValueError('Stored metric mismatch')
            rep=repeat;vals[method]=v
            item['methods'][method]={'metrics':metrics,'features':j.get('feature_names',j.get('selected_features')),
                'feature_count':len(j.get('feature_names',j.get('selected_features',[]))),
                'selected_groups':j.get('selected_groups'),'candidate_fits':j.get('candidate_fits'),
                'candidate_fit_seconds':j.get('candidate_fit_seconds'),'fit_seconds':j.get('fit_seconds'),
                'application_seconds':j.get('application_seconds'),
                'job_wall_seconds':sum(a.get('wall_seconds',0) for a in job['attempts']),
                'sampled_peak_tree_rss_bytes':job.get('resources',{}).get('peak_tree_rss_bytes'),
                'reused_identical_model_from':j.get('reused_identical_model_from'),
                'result_sha256':sha(path/'result.json'),
                'trials':read(path/'trials.json') if (path/'trials.json').is_file() else None}
        for left,right in [('forward','manual'),('random_search','manual'),('random_search','forward'),('all','manual'),('decay','manual'),('no_pair','original')]:
            if left not in vals or right not in vals:continue
            for name,mask in [('V_check',ts>=meta['midpoint']),('V_search',ts<meta['midpoint']),('all',np.ones(len(queries),bool)),('novel',~rep)]:
                item['comparisons'][left+'-minus-'+right+'/'+name]=paired_interval(queries,vals[left],vals[right],mask,j['seed'],1000)
        result['protocols'][protocol]=item
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Selected export already exists; version it explicitly')
    write(output,result)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();print(collect(root,(root/a.run).resolve(),(root/a.output).resolve()))
