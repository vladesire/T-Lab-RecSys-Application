from __future__ import annotations
import ast
import json
from pathlib import Path
import jsonschema

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "src/researchflow/schemas"

def test_every_schema_is_valid_and_accepts_its_fixture():
    for p in sorted(SCHEMAS.glob("*.json")):
        schema = json.loads(p.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        fixture = ROOT / "tests/fixtures/schema_examples" / p.name
        assert fixture.is_file(), p.name
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(json.loads(fixture.read_text()))

def test_core_envelopes_reject_unknown_fields():
    for kind in ["experiment_config", "runner_request", "runner_result", "analysis_result", "publication_config"]:
        schema = json.loads((SCHEMAS / (kind + ".json")).read_text())
        value = json.loads((ROOT / "tests/fixtures/schema_examples" / (kind + ".json")).read_text())
        value["invented_field"] = True
        assert not jsonschema.Draft202012Validator(schema).is_valid(value)

def test_public_api_signatures_are_preserved():
    inventory = json.loads((ROOT / "contracts/api_inventory.json").read_text())
    for item in inventory.values():
        tree = ast.parse((ROOT / item["source"]).read_text())
        actual = {n.name:n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for signature in item["signatures"]:
            expected = ast.parse("def " + signature + ": ...").body[0]
            assert expected.name in actual, item["source"]
            got = actual[expected.name]
            assert ast.dump(got.args, include_attributes=False) == ast.dump(expected.args, include_attributes=False), expected.name
            assert ast.dump(got.returns, include_attributes=False) == ast.dump(expected.returns, include_attributes=False), expected.name
