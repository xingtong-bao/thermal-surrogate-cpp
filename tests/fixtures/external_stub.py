#!/usr/bin/env python3
"""Stub external reference for tests: implements the ExternalReferenceBackend
protocol without any solver. Behaviour is selected by the EXT_STUB env var:
  ok      write a constant 300 K field
  fail    exit 3
  hang    sleep 30 s (the backend must kill it on timeout)
  short   write a field of the wrong size
Reads the input only to prove the file arrived (numpy-free: parses the header)."""
import os, struct, sys, time

mode = os.environ.get("EXT_STUB", "ok")
inp, outp = sys.argv[1], sys.argv[2]
with open(inp, "rb") as f:
    assert f.read(6) == b"\x93NUMPY", "input is not .npy"
if mode == "fail":
    sys.exit(3)
if mode == "hang":
    time.sleep(30)
n = 64 * 64 if mode != "short" else 10
header = "{'descr': '<f4', 'fortran_order': False, 'shape': (64, 64, 1), }"
pad = 64 - (10 + len(header) + 1) % 64
header = header + " " * pad + "\n"
with open(outp, "wb") as f:
    f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode())
    f.write(struct.pack("<%df" % n, *([300.0] * n)))
