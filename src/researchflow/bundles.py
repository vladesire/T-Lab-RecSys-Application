"""Terminal bundles: exact inventories and atomic, exclusive publication.

Callers must stop writing before commit. Verification detects later changes;
this module is not a security boundary against concurrent filesystem writers.
"""
from __future__ import annotations

import copy
import ctypes
import errno
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping

from researchflow import documents
from researchflow.contracts import WorkflowError

_KINDS = frozenset({'run', 'analysis', 'publication', 'compiled', 'context', 'release', 'review'})
_TERMINAL = frozenset({'succeeded', 'failed', 'cancelled'})
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')


def _error(code: str, message: str, **details: Any) -> WorkflowError:
    return WorkflowError(code, message, details)


def _identifier(value: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise _error('IDENTITY', 'Bundle ID must be a safe single component', bundle_id=value)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in range(1, 10)
    }
    if value.endswith('.') or value.split('.')[0].upper() in reserved:
        raise _error('IDENTITY', 'Bundle ID is not portable', bundle_id=value)


def _directory(path: Path) -> None:
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise _error('CONTRACT', 'Bundle root must be a directory, not a symlink', path=str(path))


def _relative(value: str) -> None:
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or '\x00' in value or PurePosixPath(value).is_absolute()
            or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise _error('CONTRACT', 'Inventory path must be a canonical relative path', path=value)
    if value == 'manifest.json':
        raise _error('CONTRACT', 'A manifest cannot inventory itself')


def _manifest(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise _error('CONTRACT', 'Manifest must be an object')
    kind = value.get('bundle_kind')
    if not isinstance(kind, str) or kind not in _KINDS:
        raise _error('CONTRACT', 'Unknown bundle kind', bundle_kind=kind)
    status = value.get('status')
    if not isinstance(status, str) or status not in _TERMINAL:
        raise _error('CONTRACT', 'Only terminal manifests are supported', status=status)
    documents.validate_document(kind + '_manifest', value)
    _identifier(value['bundle_id'])
    seen = set()
    for entry in value['files']:
        relative = entry['path']
        _relative(relative)
        if relative in seen:
            raise _error('CONTRACT', 'Duplicate inventory path', path=relative)
        seen.add(relative)


def _inventory(root: Path) -> list[dict[str, Any]]:
    """Walk without following links; nested manifest.json files are payload."""
    _directory(root)
    files = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                relative = path.relative_to(root).as_posix()
                if relative != 'manifest.json':
                    _relative(relative)
                    documents.contained_path(root, relative)
                    files.append({'path': relative, 'sha256': documents.sha256_file(path),
                                  'size_bytes': path.stat().st_size})
            else:
                raise _error('CONTRACT', 'Symlinks and special files are forbidden', path=str(path))

    visit(root)
    return sorted(files, key=lambda entry: entry['path'])


def _rename_exclusive(source: Path, target: Path) -> None:
    """Native atomic no-replace, including when an existing target is empty."""
    if sys.platform == 'win32':
        os.rename(source, target)  # Windows refuses an existing target.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes, target_bytes = os.fsencode(source), os.fsencode(target)
    if sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, target_bytes, 1)  # RENAME_NOREPLACE
    elif sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, 4)  # RENAME_EXCL
    else:
        raise _error('DEPENDENCY', 'Atomic exclusive directory rename is unavailable')
    if result:
        code = ctypes.get_errno()
        if code in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise _error('DEPENDENCY', 'Filesystem lacks atomic exclusive rename', errno=code)
        raise OSError(code, os.strerror(code), str(target))


def begin_bundle(output_root: Path, bundle_id: str) -> Path:
    """Reserve a fresh stage, rejecting existing stage and final identities."""
    _identifier(bundle_id)
    root = Path(output_root)
    stage = root / '.staging' / bundle_id
    final = root / bundle_id
    try:
        root.mkdir(parents=True, exist_ok=True)
        _directory(root)
        stage.parent.mkdir(exist_ok=True)
        _directory(stage.parent)
        if os.path.lexists(final):
            raise _error('IDENTITY', 'Final bundle identity already exists', path=str(final))
        stage.mkdir()
        if os.path.lexists(final):
            stage.rmdir()
            raise _error('IDENTITY', 'Final bundle identity already exists', path=str(final))
        return stage
    except FileExistsError as exc:
        raise _error('IDENTITY', 'Staging identity already exists', path=str(stage)) from exc
    except OSError as exc:
        raise _error('EXECUTION', 'Cannot create bundle stage', path=str(stage), errno=exc.errno) from exc


def commit_bundle(stage_dir: Path, final_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Validate and seal a stage, then atomically publish it without replacement.

    Failure leaves the stage for diagnosis. Once manifest.json is written, the
    stage is sealed; a failed publication needs a fresh stage to retry.
    """
    stage, final = Path(stage_dir), Path(final_dir)
    try:
        _identifier(final.name)
        _directory(stage)
        if os.path.lexists(final):
            raise _error('IDENTITY', 'Final bundle already exists', path=str(final))
        if stage.resolve() == final.resolve() or stage.resolve() in final.resolve().parents:
            raise _error('CONTRACT', 'Final path cannot be inside the stage')
        if os.path.lexists(stage / 'manifest.json'):
            raise _error('IDENTITY', 'Stage is already sealed by a manifest', path=str(stage))
        if not isinstance(manifest, Mapping):
            raise _error('CONTRACT', 'Manifest must be an object')
        if 'files' in manifest and not isinstance(manifest['files'], list):
            raise _error('CONTRACT', 'Manifest files must be an array')
        for entry in manifest.get('files', []):
            if isinstance(entry, Mapping) and entry.get('path') == 'manifest.json':
                raise _error('CONTRACT', 'A manifest cannot inventory itself')
        complete = copy.deepcopy(dict(manifest))
        complete['files'] = _inventory(stage)
        _manifest(complete)
        final.parent.mkdir(parents=True, exist_ok=True)
        if stage.stat().st_dev != final.parent.stat().st_dev:
            raise _error('EXECUTION', 'Commit requires the same filesystem')
        documents.atomic_write_json(stage / 'manifest.json', complete)
        verify_bundle(stage, complete['bundle_kind'])
        _rename_exclusive(stage, final)
        return final
    except OSError as exc:
        code = 'IDENTITY' if exc.errno in {errno.EEXIST, errno.ENOTEMPTY} else 'EXECUTION'
        raise _error(code, 'Cannot commit bundle', path=str(final), errno=exc.errno) from exc


def verify_bundle(bundle_dir: Path, expected_kind: str | None = None) -> dict[str, Any]:
    """Check kind, schema, exact inventory, hashes and sizes without repair."""
    root = Path(bundle_dir)
    try:
        _directory(root)
        if not stat.S_ISREG((root / 'manifest.json').lstat().st_mode):
            raise _error('CONTRACT', 'Manifest must be a regular file')
        manifest = documents.load_document(root / 'manifest.json')
        _manifest(manifest)
        if expected_kind is not None and manifest['bundle_kind'] != expected_kind:
            raise _error('CONTRACT', 'Unexpected bundle kind', expected=expected_kind,
                         actual=manifest['bundle_kind'])
        expected = {entry['path']: entry for entry in manifest['files']}
        actual = {entry['path']: entry for entry in _inventory(root)}
        if expected.keys() != actual.keys():
            raise _error('CONTRACT', 'Bundle file inventory mismatch',
                         missing=sorted(expected.keys() - actual.keys()),
                         unexpected=sorted(actual.keys() - expected.keys()))
        for relative in expected:
            if expected[relative] != actual[relative]:
                raise _error('CONTRACT', 'Bundle file hash or size mismatch', path=relative)
        return manifest
    except OSError as exc:
        raise _error('CONTRACT', 'Cannot read bundle', path=str(root), errno=exc.errno) from exc


def bundle_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Output identity of the complete terminal manifest, including provenance."""
    _manifest(manifest)
    return documents.digest_document(manifest)
