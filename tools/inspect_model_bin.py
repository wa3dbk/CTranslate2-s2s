#!/usr/bin/env python3
"""Read a single variable out of CT2's model.bin and dump stats + raw bytes.

Usage:
  python3 tools/inspect_model_bin.py /path/to/ct2_s2s_model/model.bin \
      --name decoder/projection/bias --out /tmp/disk.bias.bin

If --name is omitted, lists every variable with its shape and dtype.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

DTYPE_IDS = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
DTYPE_SIZES = {"float32": 4, "int8": 1, "int16": 2, "int32": 4, "float16": 2, "bfloat16": 2}


def read_string(f):
    length = struct.unpack("H", f.read(2))[0]
    data = f.read(length)
    # NUL-terminated; drop trailing 0
    return data.rstrip(b"\x00").decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_bin")
    ap.add_argument("--name", help="Variable name to extract (e.g. decoder/projection/bias)")
    ap.add_argument("--out", help="Where to write the raw bytes of --name")
    ap.add_argument("--filter", help="Substring filter when listing")
    args = ap.parse_args()

    with open(args.model_bin, "rb") as f:
        version = struct.unpack("I", f.read(4))[0]
        name = read_string(f)
        revision = struct.unpack("I", f.read(4))[0]
        n_vars = struct.unpack("I", f.read(4))[0]
        print(f"# version={version} name={name} revision={revision} n_vars={n_vars}", file=sys.stderr)

        found = False
        for _ in range(n_vars):
            var_name = read_string(f)
            rank = struct.unpack("B", f.read(1))[0]
            shape = tuple(struct.unpack("I", f.read(4))[0] for _ in range(rank))
            dtype_id = struct.unpack("B", f.read(1))[0]
            n_bytes = struct.unpack("I", f.read(4))[0]
            dtype = DTYPE_IDS[dtype_id] if dtype_id < len(DTYPE_IDS) else f"?{dtype_id}"
            raw = f.read(n_bytes)

            if args.name:
                if var_name == args.name:
                    print(f"{var_name}  shape={shape}  dtype={dtype}  bytes={n_bytes}")
                    if dtype == "float32":
                        arr = np.frombuffer(raw, dtype=np.float32)
                        print(f"  min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}  argmax={int(arr.argmax())}")
                        print(f"  first-10: {arr[:10].tolist()}")
                    if args.out:
                        Path(args.out).write_bytes(raw)
                        print(f"  wrote {n_bytes} bytes to {args.out}")
                    found = True
            else:
                if not args.filter or args.filter in var_name:
                    print(f"{var_name:60s}  shape={str(shape):20s}  dtype={dtype}")

        if args.name and not found:
            print(f"NOT FOUND: {args.name}", file=sys.stderr)
            sys.exit(1)

        # Alias section (binary version >= 3).
        if version >= 3:
            try:
                n_aliases = struct.unpack("I", f.read(4))[0]
                if n_aliases:
                    print(f"\n# {n_aliases} alias(es):", file=sys.stderr)
                for _ in range(n_aliases):
                    alias = read_string(f)
                    target = read_string(f)
                    if not args.name or alias == args.name or target == args.name:
                        print(f"ALIAS: {alias}  ->  {target}")
            except struct.error:
                print("# (no alias section)", file=sys.stderr)


if __name__ == "__main__":
    main()
