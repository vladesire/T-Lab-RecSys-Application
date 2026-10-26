#!/usr/bin/env python3
"""Verify published run inventories and recompute final metrics from saved scores."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def members(tar):
    found = {}
    for entry in tar.getmembers():
        path = PurePosixPath(entry.name)
        if path.is_absolute() or any(p in {'', '.', '..'} for p in entry.name.split('/')) or '\\' in entry.name or ':' in entry.name:
            raise ValueError('Unsafe archive member: ' + entry.name)
        if entry.name in found or not (entry.isfile() or entry.isdir()):
            raise ValueError('Duplicate or non-regular archive member: ' + entry.name)
        if entry.size > 300_000_000:
            raise ValueError('Oversized archive member')
        found[entry.name] = entry
    return found


def verify_archive(archive, prefix, inventory_name, expected=None):
    with tarfile.open(archive) as tar:
        entries = members(tar)
        inventory_path = prefix + '/' + inventory_name
        document = json.load(tar.extractfile(inventory_path))
        if expected and hashlib.sha256(tar.extractfile(inventory_path).read()).hexdigest() != expected:
            raise ValueError('Original analysis manifest differs')
        records = document['files']
        expected_paths = {prefix + '/' + r['path'] for r in records} | {inventory_path}
        actual_paths = {n for n, r in entries.items() if r.isfile()}
        if actual_paths != expected_paths:
            raise ValueError('Archive inventory differs')
        for record in records:
            item = entries[prefix + '/' + record['path']]
            size = record.get('size_bytes', record.get('bytes'))
            if size is not None and item.size != size:
                raise ValueError('Archive member size differs')
            with tar.extractfile(item) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
                    raise ValueError('Archive member hash differs: ' + item.name)
        return document


def verify(root, extract_final=None):
    published = root / 'runs/published'
    index = json.loads((published / 'INDEX.json').read_text())
    for record in index['files']:
        path = published / record['path']
        if not path.resolve().is_relative_to(published.resolve()) or path.is_symlink():
            raise ValueError('Unsafe run export path')
        if path.stat().st_size != record['size_bytes'] or sha(path) != record['sha256']:
            raise ValueError('Saved export differs: ' + record['path'])
    declaration = json.loads((root / 'evidence/submission/distribution.json').read_text())
    verify_archive(published/'development/analysis.tar.xz', declaration['analysis_path'], 'manifest.json', declaration['manifest_sha256'])
    for record in index.get('additional_analyses', []):
        verify_archive(published/record['archive_path'],record['bundle_id'],'manifest.json',record['manifest_sha256'])
    archive = published / 'final/selected-evidence.tar.xz'
    export = verify_archive(archive, 'final-evidence-selected', 'EXPORT.json')
    sys.path.insert(0, str(root/'engines/tlab_warmup/src'))
    from tlab_warmup.evaluation import per_query, aggregate
    with tarfile.open(archive) as tar:
        prefix = 'final-evidence-selected/'
        def data(name):return tar.extractfile(prefix+name).read()
        frozen_bytes = data('freeze.json')
        freeze = json.loads(frozen_bytes)
        result_bytes = data('result.json')
        result = json.loads(result_bytes)
        if hashlib.sha256(frozen_bytes).hexdigest() != result['freeze_sha256']:
            raise ValueError('Final freeze differs')
        if result_bytes != (published/'final/result.json').read_bytes() or frozen_bytes != (published/'final/freeze.json').read_bytes():
            raise ValueError('Loose final summaries differ from archived originals')
        if hashlib.sha256(data('source-manifest.json')).hexdigest() != export['source_manifest_sha256']:
            raise ValueError('Source manifest differs')
        frozen_models = {m['model']:m['model_sha256'] for m in freeze['models']}
        for record in export['model_mapping']:
            if frozen_models.get(record['source_path']) != record['sha256']:
                raise ValueError('Model export differs from pre-inference freeze')
            if hashlib.sha256(data(record['export_path'])).hexdigest() != record['sha256']:
                raise ValueError('Frozen model differs')
        checked = 0
        for protocol, summary in result['protocols'].items():
            queries = json.loads(gzip.decompress(data(protocol+'/queries.json.gz')))
            for method, details in summary['methods'].items():
                with np.load(io.BytesIO(data(protocol+'/'+method+'.npz')), allow_pickle=False) as arrays:
                    values = per_query(queries, arrays['scores'], json.loads((root/'reproducibility/anchor-config.json').read_text())['seed'])
                    np.testing.assert_allclose(values, arrays['per_query_metrics'], rtol=0, atol=1e-14)
                    np.testing.assert_array_equal(arrays['query_ids'], [q['query_id'] for q in queries])
                    metrics = aggregate(queries, values, arrays['repeat'], freeze['boundaries']['development_end_exclusive'])
                    for subset in ['all', 'repeat', 'novel']:
                        for metric in ['ndcg10', 'recall10', 'ndcg10_user_macro', 'recall10_user_macro']:
                            np.testing.assert_allclose(metrics[subset][metric], details['metrics'][subset][metric], rtol=0, atol=1e-14)
                checked += 1
        if extract_final:
            target = extract_final.resolve()
            if target.exists():raise ValueError('Extraction destination already exists')
            target.mkdir(parents=True)
            for name, entry in members(tar).items():
                if not entry.isfile():continue
                relative = PurePosixPath(name).relative_to('final-evidence-selected')
                path = target / Path(*relative.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as src, path.open('xb') as dst:shutil.copyfileobj(src, dst)
    return {'run_files_verified': len(index['files']), 'development_analysis': 'complete inventory verified',
            'final_export': 'complete selected-export inventory verified', 'score_series_recomputed': checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--extract-final', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), args.extract_final), indent=2))
