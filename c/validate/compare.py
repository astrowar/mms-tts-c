#!/usr/bin/env python3
"""
Compare Python reference (.npy) vs C implementation (.bin) stage outputs.

Usage:
    python compare.py --ref-dir ./ref_out --c-dir ./c_out/

Expected .bin format (written by C dump code):
    [4 bytes] int32 ndim
    [4 bytes * ndim] int32 shape dims
    [4 bytes * prod(shape)] float32 data (row-major)

Reports max absolute difference and relative error per stage.
"""

import argparse
import os
import sys
import struct
import numpy as np


def load_bin(path, dtype):
    """Load a .bin file in our simple format."""
    with open(path, "rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = struct.unpack(f"<{ndim}i", f.read(4 * ndim))
        n = 1
        for s in shape:
            n *= s
        data = np.frombuffer(f.read(4 * n), dtype=dtype)
        return data.reshape(shape)


def compare_stage(ref_path, c_path):
    """Compare one stage. Returns (max_abs_diff, mean_abs_diff, rel_err)."""
    ref = np.load(ref_path)
    dtype = np.int32 if ref.dtype.kind in "iu" else np.float32
    c = load_bin(c_path, dtype)

    ref = ref.astype(np.float64)
    c = c.astype(np.float64)

    if ref.shape != c.shape:
        print(f"    ⚠️  SHAPE MISMATCH: ref={ref.shape} c={c.shape}")
        # Try to compare min common size
        min_size = min(ref.size, c.size)
        ref_flat = ref.flatten()[:min_size]
        c_flat = c.flatten()[:min_size]
    else:
        ref_flat = ref.flatten()
        c_flat = c.flatten()

    diff = np.abs(ref_flat - c_flat)
    max_abs = float(diff.max()) if diff.size > 0 else 0.0
    mean_abs = float(diff.mean()) if diff.size > 0 else 0.0
    ref_norm = float(np.abs(ref_flat).max()) if ref_flat.size > 0 else 1.0
    rel_err = max_abs / max(ref_norm, 1e-8)

    return max_abs, mean_abs, rel_err


def main():
    parser = argparse.ArgumentParser(description="Compare VITS pipeline stages")
    parser.add_argument("--ref-dir", type=str, required=True, help="Python .npy outputs")
    parser.add_argument("--c-dir", type=str, required=True, help="C .bin outputs")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max rel error to pass")
    args = parser.parse_args()

    # Discover stages from ref dir
    ref_files = sorted([f for f in os.listdir(args.ref_dir) if f.endswith(".npy")])

    if not ref_files:
        print(f"No .npy files found in {args.ref_dir}")
        sys.exit(1)

    print(f"Comparing {len(ref_files)} stages")
    print(f"  ref: {args.ref_dir}")
    print(f"  c:   {args.c_dir}")
    print()

    all_pass = True
    first_fail = None

    for ref_file in ref_files:
        name = ref_file.replace(".npy", "")
        c_file = name + ".bin"
        ref_path = os.path.join(args.ref_dir, ref_file)
        c_path = os.path.join(args.c_dir, c_file)

        if not os.path.exists(c_path):
            print(f"  [{name}] ⏭️  SKIP (no C output)")
            continue

        max_abs, mean_abs, rel_err = compare_stage(ref_path, c_path)

        if rel_err < args.tolerance:
            status = "✅ PASS"
        elif rel_err < args.tolerance * 10:
            status = "⚠️  CLOSE"
            all_pass = False
        else:
            status = "❌ FAIL"
            all_pass = False
            if first_fail is None:
                first_fail = name

        print(f"  [{name}] {status}  max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}  rel={rel_err:.6f}")

    print()
    if all_pass:
        print("✅ ALL STAGES PASS")
    else:
        if first_fail:
            print(f"❌ First failure at stage: {first_fail}")
            print(f"   → Check C implementation of this stage against the Python reference")
        else:
            print("⚠️  Some stages are close but not exact (check tolerance)")


if __name__ == "__main__":
    main()
