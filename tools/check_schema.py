#!/usr/bin/env python
"""Validate contract/schema.json and the committed fixtures against it.

Why this script exists
----------------------
contract/schema.json plays two roles at once:

1. It is a JSON Schema (draft 2020-12). The ``$defs/input`` and ``$defs/output`` entries
   describe what a legal request and a legal response look like.
2. It is a configuration file. The top-level ``tensor`` and ``channels`` tables are what
   the C++ validator reads at start-up to know the expected shape, the channel order and
   (once filled in by tools/data_loader.py) the training range of every channel.

Both roles must stay consistent with each other and with contract/contract.md. This
script is the cheap guard: it is run locally after every edit and later from CI.

What is checked
---------------
* The schema file is itself a valid draft 2020-12 schema.
* The channel table (role 2) lists the same names, in the same order, as the
  ``prefixItems`` used by the validator (role 1), and the count equals the P dimension.
* Positive cases: the two committed fixtures validate.
* Negative cases: hand-crafted broken documents are rejected. Without negative cases a
  schema that accidentally accepts everything would still "pass".

Exit code 0 only when every check passes; non-zero otherwise so CI can gate on it.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

# Paths are resolved relative to this file so the script works from any working directory.
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "contract" / "schema.json"
FIXTURES = ROOT / "tests" / "fixtures"


def sub_validator(schema: dict, def_name: str) -> Draft202012Validator:
    """Build a validator for a single ``$defs`` entry.

    The top-level schema is not meant to validate documents directly (it carries the
    config tables), so we wrap one ``$defs`` entry in a tiny schema that points at it.
    ``$defs`` is copied along so that ``$ref`` inside the entry (for example
    ``#/$defs/status``) still resolves.
    """
    sub = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{def_name}",
    }
    return Draft202012Validator(sub)


def expect_valid(v: Draft202012Validator, doc: dict, label: str) -> bool:
    """Positive case: ``doc`` must validate. Prints PASS/FAIL and returns the verdict."""
    try:
        v.validate(doc)
        print(f"  PASS  {label}")
        return True
    except ValidationError as e:
        # e.message is the first failing rule in human-readable form.
        print(f"  FAIL  {label}: {e.message}")
        return False


def expect_invalid(v: Draft202012Validator, doc: dict, label: str) -> bool:
    """Negative case: ``doc`` must be rejected. Accepting it is the failure here."""
    errors = list(v.iter_errors(doc))
    if errors:
        print(f"  PASS  {label} rejected: {errors[0].message}")
        return True
    print(f"  FAIL  {label} was accepted")
    return False


def main() -> int:
    schema = json.loads(SCHEMA.read_text())

    # 1. Is the schema file itself well-formed under draft 2020-12?
    Draft202012Validator.check_schema(schema)
    print(f"schema ok: {SCHEMA.relative_to(ROOT)} (schema_version {schema['schema_version']})")

    # 2. Cross-check the two roles of the file.
    #    role 2: top-level channel table read by the C++ library
    table = [c["name"] for c in schema["channels"]]
    #    role 1: channel order enforced on documents by the validator
    prefix = [p["const"] for p in schema["$defs"]["input"]["properties"]["channels"]["prefixItems"]]
    #    P is the last entry of the declared tensor shape (X, Y, Z, P)
    p_dim = schema["tensor"]["shape"][3]
    if table != prefix or len(table) != p_dim:
        print(f"  FAIL  channel table {table} != prefixItems {prefix}, or count != P={p_dim}")
        return 1
    print(f"  PASS  channel table matches prefixItems ({len(table)} channels)")

    vin, vout = sub_validator(schema, "input"), sub_validator(schema, "output")
    ok = True

    # 3. Positive cases: the fixtures we ship must be legal documents.
    inp = json.loads((FIXTURES / "sample_input_header.json").read_text())
    out = json.loads((FIXTURES / "sample_output.json").read_text())
    ok &= expect_valid(vin, inp, "fixture sample_input_header.json")
    ok &= expect_valid(vout, out, "fixture sample_output.json")

    # 4. Negative cases. Each one mutates a copy of a good fixture in exactly one way and
    #    mirrors a rule in contract.md, so a failing negative case points at the rule
    #    whose encoding in the schema is broken.
    bad = copy.deepcopy(inp)                      # contract I2/I3: wrong channel count
    bad["channels"] = bad["channels"][:6]
    bad["shape"] = [64, 64, 1, 6]
    ok &= expect_invalid(vin, bad, "input with 6 channels")

    bad = copy.deepcopy(inp)                      # contract §2: channel order is fixed
    bad["channels"][3], bad["channels"][4] = bad["channels"][4], bad["channels"][3]
    ok &= expect_invalid(vin, bad, "input with swapped channel order")

    bad = copy.deepcopy(inp)                      # contract I1: dtype must be float32
    bad["dtype"] = "float64"
    ok &= expect_invalid(vin, bad, "input with float64")

    bad = copy.deepcopy(out)                      # contract §3: oob_flag and oob_channels agree
    bad["oob_flag"] = True                        # true but oob_channels stays empty
    ok &= expect_invalid(vout, bad, "output oob_flag=true with empty oob_channels")

    bad = copy.deepcopy(out)                      # contract O3: surrogate must report a version
    bad["source"] = "surrogate"
    bad["model_version"] = ""
    ok &= expect_invalid(vout, bad, "surrogate output without model_version")

    bad = copy.deepcopy(out)                      # contract §3: tmax_index inside the 64x64 grid
    bad["tmax_index"] = [64, 0]
    ok &= expect_invalid(vout, bad, "tmax_index out of grid")

    bad = copy.deepcopy(out)                      # contract §6: closed set of status codes
    bad["status"]["code"] = "Unknown"
    ok &= expect_invalid(vout, bad, "unknown status code")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
