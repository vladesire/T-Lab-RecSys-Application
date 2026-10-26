"""M01 isolated tests: independent expectations and the immutable shared fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest
import jsonschema

from researchflow.contracts import WorkflowError
from researchflow import documents

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "src/researchflow/schemas"
EXAMPLES = ROOT / "tests/fixtures/schema_examples"


def contract_error():
    return pytest.raises(WorkflowError, match=".+")


def test_canonical_fixture_bytes_and_hash_without_shared_helper():
    expected = '{"a":{"x":null,"ñ":"⚙"},"z":[1,true,2.5]}'
    original = {"z": [1, True, 2.5], "a": {"x": None, "ñ": "⚙"}}
    reordered = {"a": {"ñ": "⚙", "x": None}, "z": [1, True, 2.5]}
    literal = expected.encode("utf-8")
    assert documents.canonical_bytes(original) == literal
    assert documents.canonical_bytes(reordered) == literal
    assert documents.digest_document(original) == hashlib.sha256(literal).hexdigest()
    assert documents.digest_document(reordered) == hashlib.sha256(literal).hexdigest()


def test_no_implicit_path_or_timestamp_normalization():
    assert documents.digest_document({"p": "./x"}) != documents.digest_document({"p": "x"})
    assert documents.digest_document({"t": "2026-10-25Z"}) != documents.digest_document({"t": "2026-10-25+00:00"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                  {"items": [0, float("inf")]}, {1: "no"},
                                  {"bad": object()}])
def test_canonical_rejects_non_json_or_non_finite(value):
    with contract_error():
        documents.canonical_bytes(value)


def test_json_and_yaml_load_and_scientific_notation(tmp_path: Path):
    j = tmp_path / "experiment.json"
    j.write_text('{"name":"é","parameters":{"x":1e-3,"enabled":false}}', encoding="utf-8")
    assert documents.load_document(j) == {"name": "é", "parameters": {"x": 0.001, "enabled": False}}
    y = tmp_path / "experiment.yaml"
    y.write_text('schema_version: 1\ncreated_at: 2026-10-25T15:30:00Z\nscientific: 1e-3\nenabled: true\nnullable: null\n', encoding="utf-8")
    loaded = documents.load_document(y)
    assert loaded == {"schema_version": 1, "created_at": "2026-10-25T15:30:00Z",
                      "scientific": 0.001, "enabled": True, "nullable": None}
    assert isinstance(loaded["created_at"], str)


@pytest.mark.parametrize("name,text", [
    ("doc.json", '{"a":1,"a":2}'),
    ("doc.json", '{"outer":{"a":1,"a":2}}'),
    ("doc.json", '{"a":NaN}'),
    ("doc.json", '{"a":Infinity}'),
    ("doc.json", '{"a":-Infinity}'),
    ("doc.json", '{"a":1e9999}'),
    ("doc.json", '{"a":"\\ud800"}'),
    ("doc.json", '[]'),
    ("doc.json", '{"a":1} extra'),
    ("doc.yaml", 'a: 1\na: 2\n'),
    ("doc.yaml", 'outer:\n  a: 1\n  a: 2\n'),
    ("doc.yaml", 'a: .nan\n'),
    ("doc.yaml", 'a: .Inf\n'),
    ("doc.yaml", 'a: 1e9999\n'),
    ("doc.yaml", 'a: !!float .nan\n'),
    ("doc.yaml", 'a: on\n'),
    ("doc.yaml", 'a: YES\n'),
    ("doc.yaml", 'schema_version: 012\n'),
    ("doc.yaml", 'a: 0x10\n'),
    ("doc.yaml", 'a: 12:30\n'),
    ("doc.yaml", 'a: !!python/object/apply:os.system ["echo unsafe"]\n'),
    ("doc.yaml", '- not\n- an\n- object\n'),
    ("doc.yaml", 'a: 1\n---\nb: 2\n'),
    ("doc.yaml", '<<: {a: 1}\n'),
    ("doc.yaml", '? [1, 2]\n: bad\n'),
])
def test_invalid_input_documents(tmp_path: Path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with contract_error() as info:
        documents.load_document(path)
    assert info.value.code == "CONTRACT"


def test_quoted_yaml_ambiguities_are_literal_strings(tmp_path: Path):
    path = tmp_path / "quoted.yml"
    path.write_text('one: "on"\ntwo: "012"\nthree: "NaN"\n', encoding="utf-8")
    assert documents.load_document(path) == {"one": "on", "two": "012", "three": "NaN"}


def test_cyclic_yaml_alias_rejected(tmp_path: Path):
    path = tmp_path / "cycle.yaml"
    path.write_text('a: &cycle [*cycle]\n', encoding="utf-8")
    with contract_error():
        documents.load_document(path)


def test_unsupported_suffix_rejected(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text('{"a":1}', encoding="utf-8")
    with contract_error():
        documents.load_document(path)


def test_all_bundled_schemas_validate_their_immutable_fixtures():
    names = sorted(p.stem for p in SCHEMAS.glob("*.json"))
    assert len(names) >= 30
    for name in names:
        fixture = json.loads((EXAMPLES / f"{name}.json").read_text())
        assert documents.validate_document(name, fixture) is None, name


def test_schema_detects_missing_member_with_field_pointer():
    fixture = json.loads((EXAMPLES / "experiment_config.json").read_text())
    del fixture["scientific"]["experiment_id"]
    with contract_error() as err:
        documents.validate_document("experiment_config", fixture)
    assert err.value.code == "CONTRACT"
    assert err.value.details["path"] == "/scientific/experiment_id"
    assert err.value.details["validator"] == "required"


def test_schema_enforces_datetime_format_and_nested_path():
    fixture = json.loads((EXAMPLES / "run_manifest.json").read_text())
    fixture["metadata"]["started_at"] = "this is not a date-time"
    with contract_error() as err:
        documents.validate_document("run_manifest", fixture)
    assert err.value.details["path"] == "/metadata/started_at"
    assert err.value.details["validator"] == "format"


def test_schema_datetime_without_optional_jsonschema_format_dependency(monkeypatch: pytest.MonkeyPatch):
    # Simulate installing only the declared `jsonschema>=4.23` requirement,
    # without the optional rfc3339-validator used by its date-time checker.
    monkeypatch.delitem(jsonschema.FormatChecker.checkers, "date-time", raising=False)
    fixture = json.loads((EXAMPLES / "run_manifest.json").read_text())
    fixture["metadata"]["started_at"] = "this is not a date-time"
    with contract_error() as err:
        documents.validate_document("run_manifest", fixture)
    assert err.value.code == "CONTRACT"
    assert err.value.details["path"] == "/metadata/started_at"
    assert err.value.details["validator"] == "format"


@pytest.mark.parametrize("timestamp", [
    "2026-02-29T12:00:00Z",  # invalid leap day
    "2026-01-01T12:00:00",  # missing timezone
    "2026-01-01 12:00:00Z",  # required T separator
    "2026-01-01T24:00:00Z",  # invalid hour
    "2026-01-01T23:59:00+25:00",  # invalid offset
    "0000-01-01T00:00:00Z",  # invalid year
])
def test_schema_datetime_rejects_malformed_dates(timestamp: str):
    fixture = json.loads((EXAMPLES / "run_manifest.json").read_text())
    fixture["metadata"]["started_at"] = timestamp
    with contract_error() as err:
        documents.validate_document("run_manifest", fixture)
    assert err.value.details["path"] == "/metadata/started_at"
    assert err.value.details["validator"] == "format"


@pytest.mark.parametrize("timestamp", [
    "2024-02-29T12:00:00Z",
    "2026-01-01T12:00:00.123456-03:30",
    "2026-01-01t12:00:00z",
])
def test_schema_datetime_accepts_valid_rfc3339_dates(timestamp: str):
    fixture = json.loads((EXAMPLES / "run_manifest.json").read_text())
    fixture["metadata"]["started_at"] = timestamp
    assert documents.validate_document("run_manifest", fixture) is None


def test_unknown_kind_and_non_object_rejected():
    for kind in ["not_a_schema", "../../passwd", "", "../experiment_config"]:
        with contract_error():
            documents.validate_document(kind, {})
    with contract_error():
        documents.validate_document("experiment_config", [])  # type: ignore[arg-type]


def test_validate_accepts_generic_mapping_including_nested_mappings():
    fixture = json.loads((EXAMPLES / "experiment_config.json").read_text())
    fixture["scientific"] = MappingProxyType(fixture["scientific"])
    assert documents.validate_document("experiment_config", MappingProxyType(fixture)) is None


def test_schema_rejects_non_finite_in_extensible_parameters():
    fixture = json.loads((EXAMPLES / "experiment_config.json").read_text())
    fixture["scientific"]["parameters"]["bad"] = float("nan")
    with contract_error():
        documents.validate_document("experiment_config", fixture)


def test_streaming_sha256_of_file(tmp_path: Path):
    content = b"start" + bytes(range(256)) * 9000 + b"end"
    path = tmp_path / "bytes.bin"
    path.write_bytes(content)
    assert documents.sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_atomic_write_valid_and_previous_preserved_on_serialization_error(tmp_path: Path):
    target = tmp_path / "nested" / "output.json"
    documents.atomic_write_json(target, {"b": 1, "a": "c"})
    assert target.read_bytes() == b'{"a":"c","b":1}'
    before = target.read_bytes()
    with contract_error():
        documents.atomic_write_json(target, {"bad": float("nan")})
    assert target.read_bytes() == before
    assert sorted(target.parent.iterdir()) == [target]


def test_injected_serializer_failure_preserves_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "output.json"
    target.write_bytes(b'{"preserved":true}')

    def fail(*args, **kwargs):
        raise RuntimeError("serialization injected")

    monkeypatch.setattr(documents.json, "dumps", fail)
    with pytest.raises(RuntimeError, match="serialization injected"):
        documents.atomic_write_json(target, {"replace": True})
    assert target.read_bytes() == b'{"preserved":true}'
    assert sorted(tmp_path.iterdir()) == [target]


def test_replace_failure_removes_temp_and_preserves_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "target.json"
    target.write_bytes(b'{"old":true}')

    def fail(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(documents.os, "replace", fail)
    with pytest.raises(OSError, match="injected replace failure"):
        documents.atomic_write_json(target, {"new": True})
    assert target.read_bytes() == b'{"old":true}'
    assert list(tmp_path.iterdir()) == [target]


def test_contained_path_and_symlink_escape(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    inner = root / "inner"
    inner.mkdir()
    assert documents.contained_path(root, "inner/future.json") == inner / "future.json"
    for path in ("../escape", "inner/../../escape", "/tmp/escape", "C:\\escape", "\\\\host\\share", ""):
        with contract_error():
            documents.contained_path(root, path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with contract_error():
        documents.contained_path(root, "link/new.json")
    (root / "internal_link").symlink_to(inner, target_is_directory=True)
    assert documents.contained_path(root, "internal_link/result.json") == inner / "result.json"


def test_pointer_empty_missing_null_escapes_and_array():
    obj = {"a/b": {"c~d": [{"": None}, True]}, "empty": None}
    assert documents.resolve_pointer(obj, "") is obj
    assert documents.resolve_pointer(obj, "/empty") is None
    assert documents.resolve_pointer(obj, "/a~1b/c~0d/0/") is None
    assert documents.resolve_pointer(obj, "/a~1b/c~0d/1") is True
    for pointer in ("/absent", "no-slash", "/a~1b/c~0d/" + "9" * 5000, "/a~2b", "/a~", "/a~1b/c~0d/2", "/a~1b/c~0d/-1", "/a~1b/c~0d/01", "/a~1b/c~0d/0/child"):
        with contract_error() as err:
            documents.resolve_pointer(obj, pointer)
        assert err.value.code == "CONTRACT"
