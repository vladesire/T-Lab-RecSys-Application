"""Strict document I/O, canonical identities, schema validation and safe references.

Public functions intentionally do not assume a process working directory, normalize
scientific content, or mutate the shared schema bundle.
"""
from __future__ import annotations

import hashlib
from datetime import date
from importlib.resources import files
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Any, Mapping

import jsonschema
import yaml
from yaml.nodes import MappingNode, ScalarNode

from researchflow.contracts import WorkflowError


_JSON_INT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_JSON_FLOAT = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)\Z"
)
# Values that YAML 1.1 would implicitly coerce, but JSON-compatible YAML should
# not silently reinterpret. Authors may quote them to mean literal strings.
_AMBIGUOUS_PLAIN = re.compile(
    r"(?:yes|no|on|off|y|n|[+-]?(?:\.inf|\.nan|inf|infinity|nan))\Z",
    re.IGNORECASE,
)
_AMBIGUOUS_NUMBER = re.compile(
    r"[+-]?(?:0[0-9_]+|0[xXoObB][0-9a-fA-F_]+|[0-9][0-9_]*:[0-9:]+|"
    r"[0-9][0-9_]*\.|\.[0-9]+|[0-9][0-9_]*_[0-9_]*|"
    r"[1-9][0-9]*[eE][+-]?$|\+[0-9]+)\Z"
)
_POINTER_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SCHEMA_KIND = re.compile(r"[a-z][a-z0-9_]*\Z")
# jsonschema registers its built-in "date-time" check only when the optional
# rfc3339-validator dependency is installed. The base jsonschema requirement
# does not include that extra, and unregistered formats otherwise pass silently.
_RFC3339_DATETIME = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>0[1-9]|1[0-2])-"
    r"(?P<day>0[1-9]|[12][0-9]|3[01])[Tt]"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])\Z"
)


def _valid_rfc3339_datetime(value: Any) -> bool:
    """RFC 3339 date-time check independent of optional jsonschema extras.

    Non-string instances are not format-checked (JSON Schema's `type` handles
    them). Leap-second notation is rejected, matching rfc3339-validator.
    """
    if not isinstance(value, str):
        return True
    match = _RFC3339_DATETIME.fullmatch(value)
    if match is None:
        return False
    try:
        date(int(match["year"]), int(match["month"]), int(match["day"]))
    except ValueError:
        return False
    return True


def _format_checker() -> jsonschema.FormatChecker:
    # Register on this instance, not the global jsonschema registry, so results
    # are deterministic whether or not rfc3339-validator is installed.
    checker = jsonschema.FormatChecker()
    checker.checks("date-time")(_valid_rfc3339_datetime)
    return checker



class _StrictYAMLLoader(yaml.SafeLoader):
    """Safe YAML without legacy YAML 1.1 scalar inference or duplicate keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, MappingNode):
            raise WorkflowError("CONTRACT", "Expected a YAML mapping")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise WorkflowError("CONTRACT", "YAML merge keys are not supported")
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise WorkflowError("CONTRACT", "YAML mapping keys must be strings", {"key": repr(key)})
            if key in result:
                raise WorkflowError("CONTRACT", "Duplicate YAML object key", {"key": key})
            result[key] = self.construct_object(value_node, deep=deep)
        return result

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        if isinstance(node, ScalarNode) and node.style is None:
            raw = node.value
            if _AMBIGUOUS_PLAIN.fullmatch(raw) or _AMBIGUOUS_NUMBER.fullmatch(raw):
                raise WorkflowError(
                    "CONTRACT", "Ambiguous unquoted YAML scalar; quote it if a string",
                    {"value": raw, "line": node.start_mark.line + 1},
                )
        return super().construct_object(node, deep=deep)


# Make a per-class copy: never mutate PyYAML's process-wide SafeLoader.
_YAML_LEGACY_SCALAR_TAGS = {
    "tag:yaml.org,2002:bool", "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float", "tag:yaml.org,2002:timestamp",
}
_StrictYAMLLoader.yaml_implicit_resolvers = {
    initial: [(tag, pattern) for tag, pattern in candidates if tag not in _YAML_LEGACY_SCALAR_TAGS]
    for initial, candidates in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictYAMLLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"(?:true|false)\Z", re.IGNORECASE),
    list("tTfF"),
)
_StrictYAMLLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", _JSON_INT, list("-0123456789"),
)
_StrictYAMLLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _JSON_FLOAT, list("-0123456789"),
)


def _json_structure(value: Any, *, path: str = "", active: set[int] | None = None) -> None:
    """Reject values that cannot be represented unambiguously as finite JSON."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")  # Escaped, unpaired surrogates are not UTF-8 text.
        except UnicodeError as exc:
            raise WorkflowError("CONTRACT", "Invalid Unicode string", {"path": path}) from exc
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkflowError("CONTRACT", "Non-finite JSON number", {"path": path})
        return
    if active is None:
        active = set()
    if isinstance(value, (Mapping, list, tuple)):
        marker = id(value)
        if marker in active:
            raise WorkflowError("CONTRACT", "Cyclic document is not JSON", {"path": path})
        active.add(marker)
        try:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise WorkflowError("CONTRACT", "Non-string JSON object key", {"path": path})
                    _json_structure(key, path=path, active=active)
                    _json_structure(item, path=path + "/" + _escape_token(key), active=active)
            else:
                for i, item in enumerate(value):
                    _json_structure(item, path=f"{path}/{i}", active=active)
        finally:
            active.remove(marker)
        return
    raise WorkflowError("CONTRACT", "Non-JSON value", {"path": path, "type": type(value).__name__})


def _escape_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _schema_value(value: Any) -> Any:
    """Convert JSON-compatible Mapping/tuple containers for jsonschema type checks."""
    if isinstance(value, Mapping):
        return {key: _schema_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_schema_value(item) for item in value]
    return value


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise WorkflowError("CONTRACT", "Duplicate JSON object key", {"key": key})
        out[key] = value
    return out


def _no_json_constant(value: str) -> Any:
    raise WorkflowError("CONTRACT", "Non-finite JSON constant", {"value": value})


def load_document(path: Path) -> dict[str, Any]:
    """Load one UTF-8 .json/.yaml/.yml mapping, rejecting ambiguous coercions."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                doc = json.load(handle, object_pairs_hook=_no_duplicate_pairs,
                                parse_constant=_no_json_constant)
            elif path.suffix.lower() in (".yaml", ".yml"):
                doc = yaml.load(handle, Loader=_StrictYAMLLoader)
            else:
                raise WorkflowError("CONTRACT", "Unsupported document suffix", {"path": str(path)})
        if not isinstance(doc, dict):
            raise WorkflowError("CONTRACT", "Top-level document must be an object", {"path": str(path)})
        _json_structure(doc)
        return doc
    except WorkflowError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError, ValueError, OverflowError) as exc:
        raise WorkflowError("CONTRACT", "Cannot load document", {"path": str(path), "reason": str(exc)}) from exc


def canonical_bytes(value: Any) -> bytes:
    """The exact, unnormalized canonical JSON byte representation."""
    _json_structure(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise WorkflowError("CONTRACT", "Cannot serialize canonical JSON", {"reason": str(exc)}) from exc


def digest_document(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize before touching disk; replace the destination only after a full write."""
    payload = canonical_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_document(kind: str, value: Mapping[str, Any]) -> None:
    """Validate against the immutable packaged Draft 2020-12 schema of *kind*."""
    if not isinstance(kind, str) or not _SCHEMA_KIND.fullmatch(kind):
        raise WorkflowError("CONTRACT", "Invalid schema kind", {"kind": str(kind)})
    resource = files("researchflow").joinpath("schemas", f"{kind}.json")
    if not resource.is_file():
        raise WorkflowError("CONTRACT", "Unknown schema kind", {"kind": kind})
    if not isinstance(value, Mapping):
        raise WorkflowError("CONTRACT", "Document must be an object", {"kind": kind, "path": ""})
    _json_structure(value)
    schema = json.loads(resource.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=_format_checker())
    issues = list(validator.iter_errors(_schema_value(value)))
    if not issues:
        return
    issues.sort(key=lambda e: (len(e.absolute_path), tuple(map(str, e.absolute_path)), e.message))
    issue = issues[0]
    tokens = [str(part) for part in issue.absolute_path]
    # 'required' errors refer to the parent; report the actual missing member.
    if issue.validator == "required" and isinstance(issue.instance, Mapping):
        missing = next((key for key in issue.validator_value if key not in issue.instance), None)
        if missing is not None:
            tokens.append(str(missing))
    pointer = "".join("/" + _escape_token(part) for part in tokens)
    raise WorkflowError(
        "CONTRACT", f"Invalid {kind} at {pointer or '/'}: {issue.message}",
        {"kind": kind, "path": pointer, "validator": issue.validator,
         "schema_path": "".join("/" + _escape_token(str(p)) for p in issue.absolute_schema_path),
         "reason": issue.message},
    )


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a relative path within root, checking actual symlink targets.

    As with any check-then-use API, concurrent filesystem changes require the
    caller to control the workspace or use descriptor-relative secure I/O.
    """
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise WorkflowError("CONTRACT", "Invalid relative path", {"relative": repr(relative)})
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (posix.is_absolute() or windows.is_absolute() or windows.drive
            or "\\" in relative or ".." in posix.parts):
        raise WorkflowError("CONTRACT", "Absolute or traversing path", {"relative": relative})
    base = Path(root).resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise WorkflowError("CONTRACT", "Path escapes root", {"relative": relative, "root": str(base)})
    return candidate


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer; never treat missing as explicit null."""
    if not isinstance(pointer, str):
        raise WorkflowError("CONTRACT", "JSON Pointer must be a string")
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise WorkflowError("CONTRACT", "JSON Pointer must start with /", {"pointer": pointer})
    current = document
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise WorkflowError("CONTRACT", "Invalid JSON Pointer escape", {"pointer": pointer})
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise WorkflowError("CONTRACT", "Missing JSON Pointer member", {"pointer": pointer, "member": token})
            current = current[token]
        elif isinstance(current, list):
            if (not _POINTER_INDEX.fullmatch(token)
                    or len(token) > len(str(len(current)))
                    or int(token) >= len(current)):
                raise WorkflowError("CONTRACT", "Invalid JSON Pointer array index", {"pointer": pointer, "index": token})
            current = current[int(token)]
        else:
            raise WorkflowError("CONTRACT", "JSON Pointer traverses scalar", {"pointer": pointer, "member": token})
    return current
