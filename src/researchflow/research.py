"""Canonical source records; evidence joining belongs to a downstream view."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import os
import re
import tempfile
import uuid

import yaml

from researchflow.contracts import WorkflowError
from researchflow.documents import contained_path, load_document, sha256_file, validate_document


class _HeaderLoader(yaml.SafeLoader):
    """Keep timestamps as declared strings; reject ambiguous duplicate keys."""

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError('Frontmatter keys must be strings')
            if key in result:
                raise ValueError(f"Duplicate frontmatter key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Keep YAML's ordinary type checking, but never coerce a declared timestamp to
# datetime/date (which changes spelling and timezone notation on serialization).
_HeaderLoader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in resolvers
          if tag != 'tag:yaml.org,2002:timestamp']
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _header(text: str, path: Path) -> tuple[dict[str, Any], str]:
    # splitlines(keepends=True) retains CRLF, blank lines and trailing whitespace.
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip('\r\n') != '---':
        raise WorkflowError('CONTRACT', 'Journal requires YAML frontmatter', {'path': str(path)})
    end = next((i for i in range(1, len(lines))
                if lines[i].rstrip('\r\n') == '---'), None)
    if end is None:
        raise WorkflowError('CONTRACT', 'Unterminated journal frontmatter', {'path': str(path)})
    try:
        value = yaml.load(''.join(lines[1:end]), Loader=_HeaderLoader)
        if not isinstance(value, dict):
            raise ValueError('Frontmatter must be a mapping')
    except (yaml.YAMLError, ValueError, TypeError, RecursionError) as exc:
        raise WorkflowError('CONTRACT', 'Invalid journal frontmatter',
                            {'path': str(path), 'error': str(exc)}) from exc
    validate_document('journal_header', value)
    return value, ''.join(lines[end + 1:])


def _sources(root: Path, relative: str, suffixes: set[str]) -> list[Path]:
    directory = contained_path(root, relative)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise WorkflowError('CONTRACT', 'Record directory is not a directory', {'path': relative})
    sources = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in suffixes:
            checked = contained_path(root, path.relative_to(root).as_posix())
            if not checked.is_file():
                raise WorkflowError('CONTRACT', 'Record is not a regular file', {'path': str(path)})
            sources.append(path)
    return sources


def _links(index: dict[str, Any]) -> None:
    ids: dict[str, set[str]] = {}
    for kind in ('hypotheses', 'branches', 'decisions', 'journal'):
        seen: set[str] = set()
        for envelope in index[kind]:
            record = envelope['header'] if kind == 'journal' else envelope['record']
            if record is None:
                continue
            identity = record['id']
            if identity in seen:
                raise WorkflowError('IDENTITY', 'Duplicate research record ID',
                                    {'kind': kind, 'id': identity})
            seen.add(identity)
        ids[kind] = seen

    def require(kind: str, identity: str, source: str, field: str) -> None:
        if identity not in ids[kind]:
            raise WorkflowError('IDENTITY', 'Missing required research reference',
                                {'kind': kind, 'id': identity, 'source': source, 'field': field})

    unresolved: dict[tuple[str, str], dict[str, str]] = {}

    def defer(kind: str, identity: str) -> None:
        unresolved[(kind, identity)] = {
            'kind': kind, 'id': identity,
            'reason': 'External evidence/planning reference; not resolved by the canonical research loader.',
        }

    for kind in ('hypotheses', 'branches', 'decisions', 'journal'):
        for envelope in index[kind]:
            record = envelope['header'] if kind == 'journal' else envelope['record']
            if record is None:
                continue
            source = envelope['source_path']
            for field, target in (('branch_ids', 'branches'), ('hypothesis_ids', 'hypotheses'),
                                  ('decision_ids', 'decisions')):
                for identity in record.get(field, []):
                    require(target, identity, source, field)
            if kind == 'branches':
                for field, target in (('parent_id', 'branches'), ('status_decision_id', 'decisions')):
                    if record[field] is not None:
                        require(target, record[field], source, field)
            for identity in record.get('supersedes', []):
                require(kind, identity, source, 'supersedes')
                if identity == record['id']:
                    raise WorkflowError('IDENTITY', 'Record cannot supersede itself', {'source': source})
            for field, target in (('run_ids', 'run'), ('analysis_ids', 'analysis'),
                                  ('planned_experiment_ids', 'experiment')):
                for identity in record.get(field, []):
                    defer(target, identity)
            for reference in record.get('evidence', []):
                # Evidence strings have no grammar in v1. Recognize typed IDs;
                # retain every other string in full instead of guessing its ID.
                match = re.fullmatch(r'(run|analysis):(.+)', reference)
                defer(match[1], match[2]) if match else defer('evidence', reference)

    # A parent is ancestry, so a cycle cannot denote a valid branch tree.
    parents = {item['record']['id']: item['record']['parent_id'] for item in index['branches']}
    done: set[str] = set()
    for identity in parents:
        trail: set[str] = set()
        current = identity
        while current is not None and current not in done:
            if current in trail:
                raise WorkflowError('IDENTITY', 'Cyclic branch ancestry', {'id': current})
            trail.add(current)
            current = parents[current]
        done.update(trail)
    index['unresolved'] = [unresolved[key] for key in sorted(unresolved)]


def load_research(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    result: dict[str, Any] = {'schema_version': 1, 'hypotheses': [], 'branches': [],
                              'decisions': [], 'journal': [], 'unresolved': []}
    try:
        for plural, kind in (('hypotheses', 'hypothesis'), ('branches', 'branch'),
                             ('decisions', 'decision')):
            for path in _sources(root, f'research/{plural}', {'.json', '.yaml', '.yml'}):
                record = load_document(path)
                validate_document(kind, record)
                result[plural].append({'record': record, 'source_path': path.relative_to(root).as_posix(),
                                       'source_sha256': sha256_file(path)})
        for path in _sources(root, 'research/journal', {'.md'}):
            text = path.read_bytes().decode('utf-8')
            legacy = path.name == 'legacy.md'
            header, body = (None, text) if legacy else _header(text, path)
            result['journal'].append({'header': header, 'body': body, 'legacy': legacy,
                                      'source_path': path.relative_to(root).as_posix(),
                                      'source_sha256': sha256_file(path)})
        _links(result)
        validate_document('research_index', result)
    except (OSError, UnicodeError) as exc:
        raise WorkflowError('CONTRACT', 'Cannot read canonical research sources',
                            {'error': str(exc)}) from exc
    return result


def append_journal(project_root: Path, header: Mapping[str, Any], body: str) -> Path:
    root = project_root.resolve()
    if not isinstance(header, Mapping) or not isinstance(body, str):
        raise WorkflowError('CONTRACT', 'Journal requires a header mapping and text body')
    value = dict(header)
    value.setdefault('schema_version', 1)
    value.setdefault('id', 'J-' + uuid.uuid4().hex)
    if value.get('recorded_at') is None:
        value['recorded_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    value.setdefault('occurred_at', None)
    value.setdefault('model', None)
    validate_document('journal_header', value)
    # A stable ID may already exist under a filename unrelated to that ID.
    # Also validate every required reference before creating any source file.
    index = load_research(root)
    index['journal'].append({'header': value, 'body': body, 'legacy': False,
                              'source_path': f"research/journal/{value['id']}.md"})
    _links(index)
    path = contained_path(root, f"research/journal/{value['id']}.md")
    # legacy.md is the reserved unstructured import, never a structured entry.
    if path.name == 'legacy.md':
        raise WorkflowError('IDENTITY', 'Journal ID uses reserved legacy filename', {'id': value['id']})
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = '---\n' + yaml.safe_dump(value, sort_keys=False, allow_unicode=True) + '---\n' + body
        # Publish a completely written file using an exclusive hard link. A
        # collision cannot overwrite an old file, even with simultaneous writers.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.journal-', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise WorkflowError('IDENTITY', 'Journal filename already exists', {'path': str(path)}) from exc
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise WorkflowError('EXECUTION', 'Cannot append journal entry', {'error': str(exc)}) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
