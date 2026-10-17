#!/usr/bin/env python
"""Validate contract/schema.json itself and the committed fixtures against it.

Also runs deliberate negative cases so a schema that accepts everything cannot pass.
Exit code 0 when all positive cases validate and all negative cases are rejected.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "contract" / "schema.json"
FIXTURES = ROOT / "tests" / "fixtures"


def sub_validator(schema: dict, def_name: str) -> Draft202012Validator:
    """Validator for one $defs entry, keeping the full schema as $ref resolution root."""
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{def_name}"}
    return Draft202012Validator(sub)


def expect_valid(v: Draft202012Validator, doc: dict, label: str) -> bool:
    try:
        v.validate(doc)
        print(f"  PASS  {label}")
        return True
    except ValidationError as e:
        print(f"  FAIL  {label}: {e.message}")
        return False


def expect_invalid(v: Draft202012Validator, doc: dict, label: str) -> bool:
    errors = list(v.iter_errors(doc))
    if errors:
        print(f"  PASS  {label} rejected: {errors[0].message}")
        return True
    print(f"  FAIL  {label} was accepted")
    return False


def main() -> int:
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    print(f"schema ok: {SCHEMA.relative_to(ROOT)} (schema_version {schema['schema_version']})")

    # cross-check: the C++-facing channel table and the validator's prefixItems must agree
    table = [c["name"] for c in schema["channels"]]
    prefix = [p["const"] for p in schema["$defs"]["input"]["properties"]["channels"]["prefixItems"]]
    if table != prefix or len(table) != schema["tensor"]["shape"][3]:
        print(f"  FAIL  channel table {table} != prefixItems {prefix} or P mismatch")
        return 1
    print(f"  PASS  channel table matches prefixItems ({len(table)} channels)")

    vin, vout = sub_validator(schema, "input"), sub_validator(schema, "output")
    ok = True

    inp = json.loads((FIXTURES / "sample_input_header.json").read_text())
    out = json.loads((FIXTURES / "sample_output.json").read_text())
    ok &= expect_valid(vin, inp, "fixture sample_input_header.json")
    ok &= expect_valid(vout, out, "fixture sample_output.json")

    bad = copy.deepcopy(inp); bad["channels"] = bad["channels"][:6]; bad["shape"] = [64, 64, 1, 6]
    ok &= expect_invalid(vin, bad, "input with 6 channels")
    bad = copy.deepcopy(inp); bad["channels"][3], bad["channels"][4] = bad["channels"][4], bad["channels"][3]
    ok &= expect_invalid(vin, bad, "input with swapped channel order")
    bad = copy.deepcopy(inp); bad["dtype"] = "float64"
    ok &= expect_invalid(vin, bad, "input with float64")
    bad = copy.deepcopy(out); bad["oob_flag"] = True
    ok &= expect_invalid(vout, bad, "output oob_flag=true with empty oob_channels")
    bad = copy.deepcopy(out); bad["source"] = "surrogate"; bad["model_version"] = ""
    ok &= expect_invalid(vout, bad, "surrogate output without model_version")
    bad = copy.deepcopy(out); bad["tmax_index"] = [64, 0]
    ok &= expect_invalid(vout, bad, "tmax_index out of grid")
    bad = copy.deepcopy(out); bad["status"]["code"] = "Unknown"
    ok &= expect_invalid(vout, bad, "unknown status code")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
