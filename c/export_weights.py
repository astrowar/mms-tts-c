#!/usr/bin/env python3
"""
Export MMS TTS (mms-tts-por) weights from safetensors to:
  1. A flat binary file (.vtsm) with all model weights
  2. A C header file (.h) describing the binary layout

The .vtsm format (v1):
  Offset  Size  Field
  ------  ----  ----------------------------------------------------------
  0       4     magic bytes "VTSM"
  4       4     version (uint32 LE) = 1
  8       4     vocab_size (uint32 LE)
  12      4     hidden_size (uint32 LE)
  16      8     total_data_size in bytes (uint64 LE)
  24      8     reserved (zero)
  32      ...   sequential float32 tensor data

The .vtsm format (v2, with --int8):
  Offset  Size  Field
  ------  ----  ----------------------------------------------------------
  0       4     magic bytes "VTSM"
  4       4     version (uint32 LE) = 2
  8       4     vocab_size (uint32 LE)
  12      4     hidden_size (uint32 LE)
  16      8     total_data_size in bytes (uint64 LE) [F32 data only]
  24      8     qdata_size (uint64 LE) [int8 element count]
  32      8     n_scales (uint64 LE) [float32 scale count]
  40      8     reserved (zero)
  48      ...   sequential float32 tensor data
  ...     ...   int8 HiFi-GAN weight data (qdata_size bytes)
  ...     ...   float32 per-channel scales (n_scales * 4 bytes)

The generated .h defines:
  - Size constants for each tensor
  - Offset constants (byte position in file)
  - A WtsTensor descriptor struct and a layout table
  - Section groupings

Usage:
    python export_weights.py --input model.safetensors --output model.vtsm
    python export_weights.py --input model.safetensors --output model.vtsm --header model_weights.h
"""

import argparse
import struct
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

MAGIC = b"VTSM"
VERSION = 1
HEADER_SIZE = 32  # bytes before tensor data


# ============================================================
# Model dimension constants
# ============================================================
VOCAB_SIZE = 43
HIDDEN = 192
NUM_HEADS = 2
HEAD_DIM = 96
NUM_LAYERS = 6
WINDOW = 4
REL_SIZE = 2 * WINDOW + 1  # 9
FFN_DIM = 768
FFN_KERNEL = 3
FLOW_SIZE = 192
HALF_FLOW = 96
WAVE_KERNEL = 5
NUM_WAVE = 4
NUM_PRIOR_FLOWS = 4
DP_BINS = 10
DP_KERNEL = 3
DP_NUM_FLOWS = 4
DP_CHANNELS = 2
DP_DDS_LAYERS = 3
HIFI_INIT_CH = 512
NUM_UP = 4
RF_DILS = 3
DP_PROJ_OUT = DP_BINS * 3 - 1  # 29
UP_IN = [512, 256, 128, 64]
UP_OUT = [256, 128, 64, 32]
UP_K = [16, 16, 4, 4]
RB_CH = [256, 128, 64, 32]
RB_K = [3, 7, 11]


# ============================================================
# Tensor record for tracking the layout
# ============================================================
@dataclass
class TensorRec:
    name: str           # e.g. "embed_tokens.weight"
    stft_name: str      # original safetensors key
    shape: Tuple[int, ...]
    ndim: int
    n: int              # number of float32 elements
    offset: int = 0     # byte offset in .vtsm file (set during export)
    transform: str = "" # "none", "weight_norm", "transpose_convT"


@dataclass
class SectionRec:
    name: str           # e.g. "encoder.layer.0"
    tensors: List[TensorRec] = field(default_factory=list)
    offset: int = 0     # byte offset of first tensor
    n_total: int = 0    # total float32 elements in section


# ============================================================
# Safetensors reader
# ============================================================
def read_safetensors(path: str) -> dict:
    """Read safetensors file, return dict of name -> numpy array."""
    with open(path, "rb") as f:
        raw = f.read()

    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len])
    data_start = 8 + header_len

    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        shape = meta["shape"]
        dtype = meta["dtype"]
        if dtype != "F32":
            raise ValueError(f"Unexpected dtype {dtype} for {name}")
        n = 1
        for d in shape:
            n *= d
        arr = np.frombuffer(raw, dtype=np.float32, count=n, offset=data_start + s)
        tensors[name] = np.array(arr).reshape(shape)

    return tensors


# ============================================================
# Transform helpers
# ============================================================
def apply_weight_norm(tensors: dict, base: str) -> np.ndarray:
    """Fuse weight_g * weight_v / ||weight_v|| for a conv layer.
    g: [out,1,1], v: [out,in,k] -> result [out,in,k]
    """
    g = tensors[base + ".weight_g"].reshape(-1, 1, 1)  # [out,1,1]
    v = tensors[base + ".weight_v"]                    # [out,in,k]
    norm = np.linalg.norm(v, axis=(1, 2), keepdims=True)  # [out,1,1]
    norm = np.where(norm < 1e-9, 1.0, norm)
    return g * v / norm


def transpose_convT(tensors: dict, base: str) -> np.ndarray:
    """[in,out,k] -> [out,in,k]"""
    w = tensors[base + ".weight"]
    return np.transpose(w, (1, 0, 2))


# ============================================================
# Int8 quantization for HiFi-GAN (per-output-channel symmetric)
# ============================================================
QMAX = 127


def quantize_conv_w(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a [out_ch, in_ch, k] weight tensor.
    Matches C hifigan_quantize exactly: scale = max/127, v = w*(1/s), lroundf."""
    w = w.astype(np.float32)
    out_ch = w.shape[0]
    scales = np.zeros(out_ch, dtype=np.float32)
    q = np.zeros_like(w, dtype=np.int8)
    for o in range(out_ch):
        max_val = np.max(np.abs(w[o]))
        s = np.float32(max_val / QMAX) if max_val > 1e-10 else np.float32(1.0)
        scales[o] = s
        inv_scale = np.float32(1.0) / s
        v = w[o] * inv_scale
        # lroundf: round half away from zero
        v_abs = np.abs(v)
        rounded = np.sign(v) * np.floor(v_abs + np.float32(0.5)).astype(np.int32)
        np.clip(rounded, -QMAX - 1, QMAX, out=rounded)
        q[o] = rounded.astype(np.int8)
    return q, scales


def quantize_hifigan(tensors: dict) -> Tuple[bytes, bytes]:
    """Quantize all HiFi-GAN decoder weight tensors to int8.
    Returns (qdata_bytes, scales_bytes) in the same order as hifigan_quantize() in C."""
    dec = "decoder."
    q_parts = []
    s_parts = []

    def add_q(w_np: np.ndarray):
        """w_np is [out_ch, in_ch, k]"""
        q, s = quantize_conv_w(w_np)
        q_parts.append(q.tobytes())
        s_parts.append(s.tobytes())

    # conv_pre: [512, 192, 7]
    add_q(tensors[dec + "conv_pre.weight"])

    # upsamplers: stored as [in,out,k] in safetensors → transpose to [out,in,k]
    for i in range(NUM_UP):
        w = tensors[f"{dec}upsampler.{i}.weight"]
        add_q(np.transpose(w, (1, 0, 2)))

    # resblocks: 4 stages × 3 MRF, interleaved per dilation (c1[d], c2[d])
    for s in range(4):
        for kb in range(3):
            rb = s * 3 + kb
            base = f"{dec}resblocks.{rb}."
            for d in range(RF_DILS):
                add_q(tensors[f"{base}convs1.{d}.weight"])
                add_q(tensors[f"{base}convs2.{d}.weight"])

    # conv_post: [1, 32, 7]
    add_q(tensors[dec + "conv_post.weight"])

    return b"".join(q_parts), b"".join(s_parts)


# ============================================================
# Build the export plan (ordered list of tensors)
# ============================================================
def build_export_plan(tensors: dict) -> List[TensorRec]:
    """Build ordered list of all tensors to export, with computed shapes."""
    W: List[TensorRec] = []

    def add(name, stft_key, shape, transform="none"):
        n = 1
        for d in shape:
            n *= d
        W.append(TensorRec(name=name, stft_name=stft_key, shape=tuple(shape),
                           ndim=len(shape), n=n, transform=transform))

    # ── EMBEDDING ──
    add("embed_tokens.weight",
        "text_encoder.embed_tokens.weight", [VOCAB_SIZE, HIDDEN])

    # ── ENCODER LAYERS (6x) ──
    for l in range(NUM_LAYERS):
        p = f"text_encoder.encoder.layers.{l}."
        add(f"enc.{l}.q_proj.weight", p + "attention.q_proj.weight", [HIDDEN, HIDDEN])
        add(f"enc.{l}.q_proj.bias", p + "attention.q_proj.bias", [HIDDEN])
        add(f"enc.{l}.k_proj.weight", p + "attention.k_proj.weight", [HIDDEN, HIDDEN])
        add(f"enc.{l}.k_proj.bias", p + "attention.k_proj.bias", [HIDDEN])
        add(f"enc.{l}.v_proj.weight", p + "attention.v_proj.weight", [HIDDEN, HIDDEN])
        add(f"enc.{l}.v_proj.bias", p + "attention.v_proj.bias", [HIDDEN])
        add(f"enc.{l}.out_proj.weight", p + "attention.out_proj.weight", [HIDDEN, HIDDEN])
        add(f"enc.{l}.out_proj.bias", p + "attention.out_proj.bias", [HIDDEN])
        add(f"enc.{l}.emb_rel_k", p + "attention.emb_rel_k", [REL_SIZE, HEAD_DIM])
        add(f"enc.{l}.emb_rel_v", p + "attention.emb_rel_v", [REL_SIZE, HEAD_DIM])
        add(f"enc.{l}.ffn1.weight", p + "feed_forward.conv_1.weight", [FFN_DIM, HIDDEN, FFN_KERNEL])
        add(f"enc.{l}.ffn1.bias", p + "feed_forward.conv_1.bias", [FFN_DIM])
        add(f"enc.{l}.ffn2.weight", p + "feed_forward.conv_2.weight", [HIDDEN, FFN_DIM, FFN_KERNEL])
        add(f"enc.{l}.ffn2.bias", p + "feed_forward.conv_2.bias", [HIDDEN])
        add(f"enc.{l}.ln1.weight", p + "layer_norm.weight", [HIDDEN])
        add(f"enc.{l}.ln1.bias", p + "layer_norm.bias", [HIDDEN])
        add(f"enc.{l}.ln2.weight", p + "final_layer_norm.weight", [HIDDEN])
        add(f"enc.{l}.ln2.bias", p + "final_layer_norm.bias", [HIDDEN])

    # ── ENCODER PROJECTION ──
    add("enc.proj.weight", "text_encoder.project.weight", [2 * HIDDEN, HIDDEN, 1])
    add("enc.proj.bias", "text_encoder.project.bias", [2 * HIDDEN])

    # ── DURATION PREDICTOR ──
    dp = "duration_predictor."
    add("dp.conv_pre.weight", dp + "conv_pre.weight", [HIDDEN, HIDDEN, 1])
    add("dp.conv_pre.bias", dp + "conv_pre.bias", [HIDDEN])

    # DDS (shared structure)
    dds_base = dp + "conv_dds"
    for i in range(DP_DDS_LAYERS):
        add(f"dp.dds.dw.{i}.weight", f"{dds_base}.convs_dilated.{i}.weight", [HIDDEN, 1, DP_KERNEL])
        add(f"dp.dds.dw.{i}.bias", f"{dds_base}.convs_dilated.{i}.bias", [HIDDEN])
        add(f"dp.dds.pw.{i}.weight", f"{dds_base}.convs_pointwise.{i}.weight", [HIDDEN, HIDDEN, 1])
        add(f"dp.dds.pw.{i}.bias", f"{dds_base}.convs_pointwise.{i}.bias", [HIDDEN])
        add(f"dp.dds.n1.{i}.weight", f"{dds_base}.norms_1.{i}.weight", [HIDDEN])
        add(f"dp.dds.n1.{i}.bias", f"{dds_base}.norms_1.{i}.bias", [HIDDEN])
        add(f"dp.dds.n2.{i}.weight", f"{dds_base}.norms_2.{i}.weight", [HIDDEN])
        add(f"dp.dds.n2.{i}.bias", f"{dds_base}.norms_2.{i}.bias", [HIDDEN])

    add("dp.conv_proj.weight", dp + "conv_proj.weight", [HIDDEN, HIDDEN, 1])
    add("dp.conv_proj.bias", dp + "conv_proj.bias", [HIDDEN])

    # Flow 0: ElementwiseAffine
    add("dp.flow0.translate", dp + "flows.0.translate", [DP_CHANNELS])
    add("dp.flow0.log_scale", dp + "flows.0.log_scale", [DP_CHANNELS])

    # Flows 1-4: ConvFlow
    for f in range(1, DP_NUM_FLOWS + 1):
        base = f"{dp}flows.{f}."
        add(f"dp.flow{f}.conv_pre.weight", base + "conv_pre.weight", [HIDDEN, 1, 1])
        add(f"dp.flow{f}.conv_pre.bias", base + "conv_pre.bias", [HIDDEN])
        cb = f"{base}conv_dds"
        for i in range(DP_DDS_LAYERS):
            add(f"dp.flow{f}.dds.dw.{i}.weight", f"{cb}.convs_dilated.{i}.weight", [HIDDEN, 1, DP_KERNEL])
            add(f"dp.flow{f}.dds.dw.{i}.bias", f"{cb}.convs_dilated.{i}.bias", [HIDDEN])
            add(f"dp.flow{f}.dds.pw.{i}.weight", f"{cb}.convs_pointwise.{i}.weight", [HIDDEN, HIDDEN, 1])
            add(f"dp.flow{f}.dds.pw.{i}.bias", f"{cb}.convs_pointwise.{i}.bias", [HIDDEN])
            add(f"dp.flow{f}.dds.n1.{i}.weight", f"{cb}.norms_1.{i}.weight", [HIDDEN])
            add(f"dp.flow{f}.dds.n1.{i}.bias", f"{cb}.norms_1.{i}.bias", [HIDDEN])
            add(f"dp.flow{f}.dds.n2.{i}.weight", f"{cb}.norms_2.{i}.weight", [HIDDEN])
            add(f"dp.flow{f}.dds.n2.{i}.bias", f"{cb}.norms_2.{i}.bias", [HIDDEN])
        add(f"dp.flow{f}.conv_proj.weight", base + "conv_proj.weight", [DP_PROJ_OUT, HIDDEN, 1])
        add(f"dp.flow{f}.conv_proj.bias", base + "conv_proj.bias", [DP_PROJ_OUT])

    # ── FLOW (4 coupling layers) ──
    for f in range(NUM_PRIOR_FLOWS):
        base = f"flow.flows.{f}."
        add(f"flow.{f}.conv_pre.weight", base + "conv_pre.weight", [HIDDEN, HALF_FLOW, 1])
        add(f"flow.{f}.conv_pre.bias", base + "conv_pre.bias", [HIDDEN])

        wn = f"{base}wavenet."
        for i in range(NUM_WAVE):
            rs_out = 2 * HIDDEN if i < NUM_WAVE - 1 else HIDDEN
            add(f"flow.{f}.wn.in.{i}.weight", wn + f"in_layers.{i}",
                [2 * HIDDEN, HIDDEN, WAVE_KERNEL], transform="weight_norm")
            add(f"flow.{f}.wn.in.{i}.bias", wn + f"in_layers.{i}.bias", [2 * HIDDEN])
            add(f"flow.{f}.wn.rs.{i}.weight", wn + f"res_skip_layers.{i}",
                [rs_out, HIDDEN, 1], transform="weight_norm")
            add(f"flow.{f}.wn.rs.{i}.bias", wn + f"res_skip_layers.{i}.bias", [rs_out])

        add(f"flow.{f}.conv_post.weight", base + "conv_post.weight", [HALF_FLOW, HIDDEN, 1])
        add(f"flow.{f}.conv_post.bias", base + "conv_post.bias", [HALF_FLOW])

    # ── HIFI-GAN DECODER ──
    dec = "decoder."
    add("dec.conv_pre.weight", dec + "conv_pre.weight", [HIFI_INIT_CH, HIDDEN, 7])
    add("dec.conv_pre.bias", dec + "conv_pre.bias", [HIFI_INIT_CH])

    # Upsamplers: stored as [in,out,k] in safetensors, export as [out,in,k]
    for i in range(NUM_UP):
        add(f"dec.up.{i}.weight", f"{dec}upsampler.{i}",
            [UP_OUT[i], UP_IN[i], UP_K[i]], transform="transpose_convT")
        add(f"dec.up.{i}.bias", f"{dec}upsampler.{i}.bias", [UP_OUT[i]])

    # ResBlocks: 4 stages × 3 kernels
    for s in range(4):
        for kb in range(3):
            rb = s * 3 + kb
            ch = RB_CH[s]
            k = RB_K[kb]
            base = f"{dec}resblocks.{rb}."
            for d in range(RF_DILS):
                add(f"dec.rb.{rb}.c1.{d}.weight", f"{base}convs1.{d}.weight", [ch, ch, k])
                add(f"dec.rb.{rb}.c1.{d}.bias", f"{base}convs1.{d}.bias", [ch])
                add(f"dec.rb.{rb}.c2.{d}.weight", f"{base}convs2.{d}.weight", [ch, ch, k])
                add(f"dec.rb.{rb}.c2.{d}.bias", f"{base}convs2.{d}.bias", [ch])

    # conv_post: [1, 32, 7], no bias
    add("dec.conv_post.weight", dec + "conv_post.weight", [1, 32, 7])

    return W


# ============================================================
# Group tensors into sections for the header
# ============================================================
def group_sections(tensors: List[TensorRec], int8: bool = False) -> List[SectionRec]:
    """Group flat tensor list into logical sections."""
    sections: List[SectionRec] = []

    # Embedding
    sections.append(SectionRec(name="embed", tensors=tensors[0:1]))

    # Encoder layers
    idx = 1
    enc_layer_count = 18  # tensors per encoder layer
    for l in range(NUM_LAYERS):
        s = SectionRec(name=f"encoder.layer.{l}", tensors=tensors[idx:idx + enc_layer_count])
        sections.append(s)
        idx += enc_layer_count

    # Encoder projection
    sections.append(SectionRec(name="encoder.project", tensors=tensors[idx:idx + 2]))
    idx += 2

    # Duration predictor: conv_pre (2) + dds (24) + conv_proj (2) + flow0 (2)
    dp_main = 2 + DP_DDS_LAYERS * 8 + 2 + 2  # 30 tensors
    sections.append(SectionRec(name="dp.main", tensors=tensors[idx:idx + dp_main]))
    idx += dp_main

    # DP flows 1-4
    cf_count = 2 + DP_DDS_LAYERS * 8 + 2  # 30 tensors per ConvFlow
    for f in range(1, DP_NUM_FLOWS + 1):
        sections.append(SectionRec(name=f"dp.flow.{f}", tensors=tensors[idx:idx + cf_count]))
        idx += cf_count

    # Flow coupling layers
    # conv_pre(2) + wavenet(4*4=16) + conv_post(2) = 20 tensors
    cl_count = 2 + NUM_WAVE * 4 + 2
    for f in range(NUM_PRIOR_FLOWS):
        sections.append(SectionRec(name=f"flow.{f}", tensors=tensors[idx:idx + cl_count]))
        idx += cl_count

    # HiFi-GAN decoder
    if int8:
        # v2: only biases (weights replaced by int8 section)
        dec_count = 1 + NUM_UP + 4 * 3 * RF_DILS * 2  # conv_pre.b + ups.b + rb biases
    else:
        dec_count = (2 + NUM_UP * 2 + 4 * 3 * RF_DILS * 4 + 1)
    sections.append(SectionRec(name="decoder", tensors=tensors[idx:idx + dec_count]))
    idx += dec_count

    # Compute offsets and totals
    offset = 48 if int8 else HEADER_SIZE
    for s in sections:
        s.offset = offset
        s.n_total = sum(t.n for t in s.tensors)
        offset += s.n_total * 4

        # Also set individual tensor offsets
        toff = offset - s.n_total * 4
        for t in s.tensors:
            t.offset = toff
            toff += t.n * 4

    return sections


# ============================================================
# Export: write binary + header
# ============================================================
def export(tensors: dict, out_bin: str, out_header: str = None, int8: bool = False):
    plan = build_export_plan(tensors)

    # v2: skip F32 HiFi-GAN weight tensors (replaced by int8 section)
    if int8:
        plan = [t for t in plan
                if not (t.name.startswith("dec.") and t.name.endswith(".weight"))]

    sections = group_sections(plan, int8=int8)

    total_floats = sum(t.n for t in plan)
    total_bytes = total_floats * 4
    print(f"  tensors:     {len(plan)}" + (" (v2: F32 hifi weights skipped)" if int8 else ""))
    print(f"  sections:    {len(sections)}")
    print(f"  total floats: {total_floats:,}")
    print(f"  total bytes:  {total_bytes:,} ({total_bytes / 1024 / 1024:.1f} MB)")

    # ── Compute int8 quantized HiFi-GAN weights ──
    qdata_bytes = b""
    scales_bytes = b""
    qdata_size = 0
    n_scales = 0

    if int8:
        print(f"  quantizing HiFi-GAN to int8...")
        qdata_bytes, scales_bytes = quantize_hifigan(tensors)
        qdata_size = len(qdata_bytes)
        n_scales = len(scales_bytes) // 4
        print(f"  int8 qdata:  {qdata_size:,} bytes ({qdata_size / 1024 / 1024:.1f} MB)")
        print(f"  scales:      {n_scales} (float32)")

    ver = 2 if int8 else VERSION

    # ── Write binary ──
    with open(out_bin, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", ver))
        f.write(struct.pack("<I", VOCAB_SIZE))
        f.write(struct.pack("<I", HIDDEN))
        f.write(struct.pack("<Q", total_bytes))
        if int8:
            f.write(struct.pack("<Q", qdata_size))
            f.write(struct.pack("<Q", n_scales))
            f.write(struct.pack("<Q", 0))  # reserved
        else:
            f.write(struct.pack("<Q", 0))  # reserved

        for t in plan:
            if t.transform == "none":
                data = tensors[t.stft_name].ravel()
            elif t.transform == "weight_norm":
                data = apply_weight_norm(tensors, t.stft_name).ravel()
            elif t.transform == "transpose_convT":
                data = transpose_convT(tensors, t.stft_name).ravel()
            else:
                raise ValueError(f"Unknown transform: {t.transform}")

            if data.size != t.n:
                print(f"  [WARN] {t.name}: expected {t.n} floats, got {data.size}")
            f.write(data.astype(np.float32).tobytes())

        # ── Append int8 quantized HiFi-GAN weights ──
        if int8:
            f.write(qdata_bytes)
            f.write(scales_bytes)

    size_mb = os.path.getsize(out_bin) / 1024 / 1024
    print(f"  written:     {out_bin} ({size_mb:.1f} MB)")

    # ── Write header ──
    if out_header:
        gen_header(plan, sections, out_header, int8=int8,
                   qdata_size=qdata_size, n_scales=n_scales)
        print(f"  header:      {out_header}")


# ============================================================
# Generate C header
# ============================================================
def gen_header(plan: List[TensorRec], sections: List[SectionRec], path: str,
               int8: bool = False, qdata_size: int = 0, n_scales: int = 0):
    lines: List[str] = []
    L = lines.append

    ver = 2 if int8 else VERSION
    hdr_size = 48 if int8 else HEADER_SIZE

    L(f"/* ============================================================")
    L(f" * {os.path.basename(path)} — Auto-generated by export_weights.py")
    L(f" * Model: facebook/mms-tts-por (VITS, 36.3M params)")
    L(f" * DO NOT EDIT MANUALLY — regenerate with:")
    L(f" *   python export_weights.py --input model.safetensors \\")
    L(f" *       --output model.vtsm --header {os.path.basename(path)}"
        + (" --int8" if int8 else ""))
    L(f" * ============================================================ */")
    L(f"")
    L(f"#ifndef MODEL_WEIGHTS_H")
    L(f"#define MODEL_WEIGHTS_H")
    L(f"")
    L(f"#include <stdint.h>")
    L(f"#include <stddef.h>")
    L(f"")

    # ── File header constants ──
    L(f"/* ── File header ── */")
    L(f"#define WTS_MAGIC            0x4D535456u  /* 'V','T','S','M' LE */")
    L(f"#define WTS_VERSION          {ver}")
    L(f"#define WTS_VOCAB_SIZE       {VOCAB_SIZE}")
    L(f"#define WTS_HIDDEN_SIZE      {HIDDEN}")
    L(f"#define WTS_FILE_HEADER_SIZE {hdr_size}")
    L(f"#define WTS_TOTAL_DATA_BYTES {total_bytes(plan)}")
    if int8:
        L(f"#define WTS_QDATA_SIZE         {qdata_size}  /* int8 weight elements */")
        L(f"#define WTS_N_SCALES           {n_scales}  /* per-channel scales (float32) */")
        L(f"/* int8 qdata offset: WTS_FILE_HEADER_SIZE + WTS_TOTAL_DATA_BYTES */")
        L(f"/* scales offset:     WTS_FILE_HEADER_SIZE + WTS_TOTAL_DATA_BYTES + WTS_QDATA_SIZE */")
    L(f"")

    # ── Model dimension constants ──
    L(f"/* ── Model dimensions ── */")
    L(f"#define WTS_NUM_HEADS        {NUM_HEADS}")
    L(f"#define WTS_HEAD_DIM         {HEAD_DIM}")
    L(f"#define WTS_NUM_LAYERS       {NUM_LAYERS}")
    L(f"#define WTS_REL_SIZE         {REL_SIZE}")
    L(f"#define WTS_FFN_DIM          {FFN_DIM}")
    L(f"#define WTS_FFN_KERNEL       {FFN_KERNEL}")
    L(f"#define WTS_HALF_FLOW        {HALF_FLOW}")
    L(f"#define WTS_WAVE_KERNEL      {WAVE_KERNEL}")
    L(f"#define WTS_NUM_WAVE         {NUM_WAVE}")
    L(f"#define WTS_NUM_PRIOR_FLOWS  {NUM_PRIOR_FLOWS}")
    L(f"#define WTS_DP_BINS          {DP_BINS}")
    L(f"#define WTS_DP_KERNEL        {DP_KERNEL}")
    L(f"#define WTS_DP_NUM_FLOWS     {DP_NUM_FLOWS}")
    L(f"#define WTS_DP_CHANNELS      {DP_CHANNELS}")
    L(f"#define WTS_DP_DDS_LAYERS    {DP_DDS_LAYERS}")
    L(f"#define WTS_DP_PROJ_OUT      {DP_PROJ_OUT}")
    L(f"#define WTS_HIFI_INIT_CH     {HIFI_INIT_CH}")
    L(f"#define WTS_NUM_UP           {NUM_UP}")
    L(f"#define WTS_RF_DILS          {RF_DILS}")
    L(f"#define WTS_NUM_RESBLOCKS    {NUM_UP * 3}")
    L(f"")

    # ── Size constants ──
    L(f"/* ── Size constants (float32 element counts) ── */")
    _gen_size_constants(L)
    L(f"")

    # ── WtsTensor descriptor struct ──
    L(f"/* ── Tensor descriptor ── */")
    L(f"typedef struct {{")
    L(f"    const char *name;       /* tensor name */")
    L(f"    uint32_t     ndim;       /* number of dimensions */")
    L(f"    uint32_t     shape[4];   /* dimensions (0 if ndim < 4) */")
    L(f"    uint32_t     n_elements; /* total float32 count */")
    L(f"    uint64_t     offset;     /* byte offset in .vtsm file */")
    L(f"    uint64_t     size_bytes; /* n_elements * 4 */")
    L(f"}} WtsTensor;")
    L(f"")

    # ── WtsSection struct ──
    L(f"/* ── Section descriptor ── */")
    L(f"typedef struct {{")
    L(f"    const char *name;             /* section name */")
    L(f"    const WtsTensor *tensors;     /* array of tensor descriptors */")
    L(f"    uint32_t     n_tensors;       /* number of tensors */")
    L(f"    uint32_t     n_elements;      /* total floats in section */")
    L(f"    uint64_t     offset;          /* byte offset of first tensor */")
    L(f"}} WtsSection;")
    L(f"")

    # ── Layout table: all tensors ──
    L(f"/* ── Flat tensor layout table ({len(plan)} entries) ── */")
    L("static const WtsTensor wts_tensors[] = {")
    for t in plan:
        shape_str = ", ".join(str(d) for d in t.shape)
        if t.ndim < 4:
            shape_str += ", " + ", ".join("0" for _ in range(4 - t.ndim))
        L(f"    {{ \"{t.name}\", {t.ndim}, {{{shape_str}}}, {t.n}, {t.offset}, {t.n * 4} }},")
    L("};")
    L(f"")
    L(f"#define WTS_NUM_TENSORS  {len(plan)}")
    L(f"")

    # ── Section tables ──
    L(f"/* ── Section layout ── */")
    L("static const WtsSection wts_sections[] = {")
    for s in sections:
        first_idx = _find_tensor_index(plan, s.tensors[0])
        L(f"    {{ \"{s.name}\", &wts_tensors[{first_idx}], {len(s.tensors)}, {s.n_total}, {s.offset} }},")
    L("};")
    L(f"")
    L(f"#define WTS_NUM_SECTIONS  {len(sections)}")
    L(f"")

    # ── Section index constants ──
    L(f"/* ── Section indices ── */")
    for i, s in enumerate(sections):
        L(f"#define WTS_SEC_{_c_ident(s.name).upper()}  {i}")
    L(f"")

    # ── Convenience: get pointer to tensor data from mmap'd file ──
    L(f"/* ── Helper: get float pointer for a tensor ── */")
    L(f"static inline const float *wts_data(const unsigned char *base, const WtsTensor *t) {{")
    L(f"    return (const float *)(base + t->offset);")
    L(f"}}")
    L(f"")
    L(f"/* ── Helper: get float pointer by index ── */")
    L(f"static inline const float *wts_data_at(const unsigned char *base, uint32_t idx) {{")
    L(f"    return (const float *)(base + wts_tensors[idx].offset);")
    L(f"}}")
    L(f"")

    L(f"#endif /* MODEL_WEIGHTS_H */")
    L(f"")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def total_bytes(plan: List[TensorRec]) -> int:
    return sum(t.n for t in plan) * 4


def _c_ident(name: str) -> str:
    """Convert a section name to a valid C identifier fragment."""
    return name.replace(".", "_").replace("-", "_")


def _find_tensor_index(plan: List[TensorRec], target: TensorRec) -> int:
    for i, t in enumerate(plan):
        if t is target:
            return i
    raise ValueError(f"Tensor not found in plan: {target.name}")


def _gen_size_constants(L):
    """Generate size constants for key tensors."""
    # Embedding
    L(f"#define WTS_EMBED_W_SIZE             {VOCAB_SIZE * HIDDEN}")
    L(f"")
    # Per encoder layer
    L(f"#define WTS_ENC_ATTN_W_SIZE          {HIDDEN * HIDDEN}")
    L(f"#define WTS_ENC_ATTN_B_SIZE          {HIDDEN}")
    L(f"#define WTS_ENC_REL_SIZE             {REL_SIZE * HEAD_DIM}")
    L(f"#define WTS_ENC_FFN1_W_SIZE          {FFN_DIM * HIDDEN * FFN_KERNEL}")
    L(f"#define WTS_ENC_FFN1_B_SIZE          {FFN_DIM}")
    L(f"#define WTS_ENC_FFN2_W_SIZE          {HIDDEN * FFN_DIM * FFN_KERNEL}")
    L(f"#define WTS_ENC_FFN2_B_SIZE          {HIDDEN}")
    L(f"#define WTS_ENC_LN_SIZE              {HIDDEN}")
    enc_layer_total = (4 * HIDDEN * HIDDEN + 4 * HIDDEN +
                       2 * REL_SIZE * HEAD_DIM +
                       FFN_DIM * HIDDEN * FFN_KERNEL + FFN_DIM +
                       HIDDEN * FFN_DIM * FFN_KERNEL + HIDDEN +
                       4 * HIDDEN)
    L(f"#define WTS_ENC_LAYER_SIZE           {enc_layer_total}  /* per layer, all tensors */")
    L(f"")
    # Encoder projection
    L(f"#define WTS_ENC_PROJ_W_SIZE          {2 * HIDDEN * HIDDEN}")
    L(f"#define WTS_ENC_PROJ_B_SIZE          {2 * HIDDEN}")
    L(f"")
    # Duration predictor
    L(f"#define WTS_DP_CONV_PRE_W_SIZE       {HIDDEN * HIDDEN}")
    L(f"#define WTS_DP_CONV_PRE_B_SIZE       {HIDDEN}")
    L(f"#define WTS_DP_DDS_DW_W_SIZE         {HIDDEN * DP_KERNEL}")
    L(f"#define WTS_DP_DDS_DW_B_SIZE         {HIDDEN}")
    L(f"#define WTS_DP_DDS_PW_W_SIZE         {HIDDEN * HIDDEN}")
    L(f"#define WTS_DP_DDS_PW_B_SIZE         {HIDDEN}")
    L(f"#define WTS_DP_DDS_NORM_SIZE         {HIDDEN}")
    dds_layer = (HIDDEN * DP_KERNEL + HIDDEN + HIDDEN * HIDDEN + HIDDEN +
                 4 * HIDDEN)
    L(f"#define WTS_DP_DDS_LAYER_SIZE        {dds_layer}  /* per DDS layer */")
    L(f"#define WTS_DP_DDS_SIZE              {dds_layer * DP_DDS_LAYERS}  /* full DDS block */")
    L(f"#define WTS_DP_CONV_PROJ_W_SIZE      {HIDDEN * HIDDEN}")
    L(f"#define WTS_DP_CONV_PROJ_B_SIZE      {HIDDEN}")
    L(f"#define WTS_DP_EA_SIZE               {DP_CHANNELS}  /* translate or log_scale */")
    L(f"#define WTS_DP_CF_PRE_W_SIZE         {HIDDEN}")
    L(f"#define WTS_DP_CF_PRE_B_SIZE         {HIDDEN}")
    L(f"#define WTS_DP_CF_PROJ_W_SIZE        {DP_PROJ_OUT * HIDDEN}")
    L(f"#define WTS_DP_CF_PROJ_B_SIZE        {DP_PROJ_OUT}")
    L(f"")
    # Flow (coupling layers)
    L(f"#define WTS_FLOW_PRE_W_SIZE          {HIDDEN * HALF_FLOW}")
    L(f"#define WTS_FLOW_PRE_B_SIZE          {HIDDEN}")
    L(f"#define WTS_FLOW_WN_IN_W_SIZE        {2 * HIDDEN * HIDDEN * WAVE_KERNEL}")
    L(f"#define WTS_FLOW_WN_IN_B_SIZE        {2 * HIDDEN}")
    L(f"#define WTS_FLOW_WN_RS_W_SIZE        {2 * HIDDEN * HIDDEN}  /* layers 0-2 */")
    L(f"#define WTS_FLOW_WN_RS_W_SIZE_LAST   {HIDDEN * HIDDEN}  /* layer 3 */")
    L(f"#define WTS_FLOW_WN_RS_B_SIZE        {2 * HIDDEN}")
    L(f"#define WTS_FLOW_WN_RS_B_SIZE_LAST   {HIDDEN}")
    L(f"#define WTS_FLOW_POST_W_SIZE         {HALF_FLOW * HIDDEN}")
    L(f"#define WTS_FLOW_POST_B_SIZE         {HALF_FLOW}")
    L(f"")
    # HiFi-GAN
    L(f"#define WTS_DEC_PRE_W_SIZE           {HIFI_INIT_CH * HIDDEN * 7}")
    L(f"#define WTS_DEC_PRE_B_SIZE           {HIFI_INIT_CH}")
    for i in range(NUM_UP):
        L(f"#define WTS_DEC_UP{i}_W_SIZE           {UP_OUT[i] * UP_IN[i] * UP_K[i]}")
        L(f"#define WTS_DEC_UP{i}_B_SIZE           {UP_OUT[i]}")
    for s in range(4):
        for kb in range(3):
            ch = RB_CH[s]
            k = RB_K[kb]
            L(f"#define WTS_DEC_RB{s}_{kb}_W_SIZE        {ch * ch * k}")
            L(f"#define WTS_DEC_RB{s}_{kb}_B_SIZE        {ch}")
    L(f"#define WTS_DEC_POST_W_SIZE          {32 * 7}")
    L(f"")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Export VITS weights to binary .vtsm + C layout header"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to model.safetensors")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .vtsm binary path")
    parser.add_argument("--header", type=str, default=None,
                        help="Output .h header path (default: <output>.h)")
    args = parser.parse_args()

    if args.header is None:
        base = os.path.splitext(args.output)[0]
        args.header = base + ".h"

    print(f"Reading: {args.input}")
    tensors = read_safetensors(args.input)
    print(f"  {len(tensors)} tensors in safetensors")
    print()

    export(tensors, args.output, args.header, int8=True)
    print("Done.")


if __name__ == "__main__":
    main()
