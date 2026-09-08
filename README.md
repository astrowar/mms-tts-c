# mms-tts-por — Pure C

⚡ **Zero-dependency, on-device Portuguese TTS.** A pure C implementation of [facebook/mms-tts-por](https://huggingface.co/facebook/mms-tts-por) — Meta MMS (Massively Multilingual Speech) VITS model for Brazilian Portuguese. No Python, no PyTorch, no external libraries. Just `libc` + `libm`.

## ✨ Key Highlights

- **🪶 36.3M Parameters** — compact VITS model (VAE + normalizing flow + HiFi-GAN)
- **📱 Edge-Device Ready** — runs on Raspberry Pi 4/5 with ~133 MB peak memory
- **🔊 16 kHz 16-bit WAV** — direct output, no external upsample required
- **🚫 Zero Dependencies** — single `mmap` for weight loading; F32 weights live entirely in the mapped file
- **⚡ 67.4 MB flat binary** — v2 `.vtsm`, single file, no parsing, no copy — pointer arithmetic only
- **🔢 Quantized HiFi-GAN** — Q16 fixed-point (default, INT16 activations) or INT8+FP32 (`--int8`); no float in the decoder hot path
- **🧮 Auto-vectorized** — CMake detects AVX2/NEON and picks optimized kernels
- **✅ 100% Validated** — every F32 stage matches the HuggingFace Python reference (< 3e-3 rel. error)

## 🎧 Samples

> Generated with the Q16 fixed-point HiFi-GAN (default) and `--seed 42`
> for reproducibility. 16 kHz mono, 16-bit PCM.

---

**hello** — "Olá, mundo!" · 1.3 s → [▶ play](https://raw.githubusercontent.com/astrowar/mms-tts-c/int8/samples/hello.wav)

**bom_dia** — "Bom dia, como você está hoje?" · 2.5 s → [▶ play](https://raw.githubusercontent.com/astrowar/mms-tts-c/int8/samples/bom_dia.wav)

**tech** — "A tecnologia de fala avança mais rápido do que imaginamos." · 4.9 s → [▶ play](https://raw.githubusercontent.com/astrowar/mms-tts-c/int8/samples/tech.wav)

**embedded** — "Este modelo roda em C puro, sem dependências externas, direto no hardware…" · 13.0 s → [▶ play](https://raw.githubusercontent.com/astrowar/mms-tts-c/int8/samples/embedded.wav)

## 🚀 Quick Start

```bash
cd c/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # OpenMP is on by default
make -j$(nproc)

# From project root (needs model.vtsm)
./build/mms-tts --text "Olá, mundo!" --seed 42 --output output.wav
```

One-time setup — export weights from HuggingFace safetensors:

```bash
python3 c/export_weights.py \
    --input ~/.cache/huggingface/hub/models--facebook--mms-tts-por/snapshots/*/model.safetensors \
    --output c/model.vtsm \
    --header c/model_weights.h
```

### Usage

```bash
# Reproducible output
./mms-tts --text "Olá, mundo!" --seed 42 --output hello.wav

# Random seed
./mms-tts --text "A tecnologia de fala avança rápido." --seed -1

# Multiple sample texts
./mms-tts --multi

# Custom model path
./mms-tts --model /path/to/model.vtsm --text "Teste"

# Fixed-point (INT16) HiFi-GAN — no float in the decoder (default)
./mms-tts --text "Olá, mundo!" --output hello_q16.wav

# Legacy INT8 + FP32 activations
./mms-tts --text "Olá, mundo!" --int8 --output hello_int8.wav
```

## 📊 Performance

Benchmarked on **Raspberry Pi 4 Model B (8 GB)** — Cortex-A72 @ 1.5 GHz × 4, NEON, GCC 14.2.

| Build | Text | Audio | Wall time | Peak RSS | RTF |
|-------|------|-------|-----------|----------|-----|
| NEON (1 thread) | 3 × ~5.2 s sentences | 15.7 s | 335 s | ~133 MB | 21× |
| NEON + OpenMP × 4 | 3 × ~5.2 s sentences | 15.7 s | 97 s | ~133 MB | 5.8× |
| NEON + OpenMP × 4 | "raspberry pi funcionando e falando" | 2.4 s | 15 s (incl. model load) | ~133 MB | ~5× |
| NEON + OpenMP × 4 | "Olá, mundo!" | 1.3 s | 7 s (incl. model load) | ~133 MB | ~2× |

**x86_64 (AVX2/FMA + OpenMP)** is much faster than the Pi. A 2 s utterance
("Olá, mundo!") runs the full pipeline in **~1.0 s** on a 28-core box with
`OMP_NUM_THREADS=14` (the HiFi-GAN stage alone is ~0.9 s). The INT8 and Q16
decoders scale near-linearly with core count — 1→14 threads gave ~8.4× on the
HiFi-GAN stage for an 11.5 s utterance, with bit-identical output.

**Memory strategy:** weights live entirely in the `mmap`'d file — file-backed, reclaimable by the kernel under pressure (re-faulted from disk on access, no swap needed). The `VitsModel` struct is only **6.4 KB** of pointers + metadata; **zero heap allocation** for the F32 weights (the optional Q16 decoder keeps its small INT8/INT16 params on the heap).

**Suitability:**

| Use case | Verdict |
|----------|---------|
| Batch/offline TTS | ✅ Fully viable |
| Short phrases (< 2 s) + OpenMP | ✅ ~3 s latency |
| Long sentences (> 5 s) | ⚠️ ~30 s per sentence (4 threads) |
| Real-time streaming | ❌ Pi 5 halves the latency |

## 🏗️ Pipeline

```
text → tokenizer → encoder (6L transformer) → stochastic DP (RQS)
     → prior sampling → normalizing flow (WaveNet) → HiFi-GAN → WAV
```

All 7 stages validated against the HuggingFace `transformers` Python reference (relative error < 3e-3 at every point). See [`c/STATUS.md`](c/STATUS.md).

## 🔬 Validation

Deterministic stage-by-stage comparison using latent injection:

```bash
cd c/build
cmake .. -DENABLE_DUMP=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Generate Python reference
python3 ../validate/stages_ref.py --text "ola, mundo, tudo bem ?" --seed 42 --out-dir ../ref_out

# Run C with injection + dumps
../mms-tts --text "ola, mundo, tudo bem ?" --seed 42 --inject-dir ../ref_out --dump-dir ../c_out

# Compare
python3 ../validate/compare.py --ref-dir ../ref_out --c-dir ../c_out
```

## 📁 Structure

```
├── c/
│   ├── CMakeLists.txt      # Auto-detects arch (x86_64/AVX2, ARM64/NEON, ARMv7/NEON)
│   ├── STATUS.md           # Full implementation & validation documentation
│   ├── export_weights.py   # safetensors → .vtsm + model_weights.h
│   ├── model_weights.h     # Generated: tensor offsets & layout
│   ├── src/
│   │   ├── vits.h          # Structs, constants, declarations
│   │   ├── main.c          # CLI entry point
│   │   ├── ops_base.c      # F32 scalar ops: conv1d, conv_transpose, layernorm, activations
│   │   ├── ops_neon.c      # F32 NEON ops (aarch64 / ARMv7)
│   │   ├── ops_avx.c       # F32 AVX2/FMA ops (x86_64)
│   │   ├── ops_int8.c      # INT8 conv kernels (portable fallback)
│   │   ├── ops_int8_avx2.c # INT8 conv kernels, AVX2/FMA (x86_64)
│   │   ├── ops_int8_neon.c # INT8 conv kernels, NEON (aarch64 / ARMv7)
│   │   ├── ops_q16.c       # Q16 fixed-point kernels (portable fallback)
│   │   ├── ops_q16_avx2.c  # Q16 fixed-point kernels, AVX2 (x86_64)
│   │   ├── tokenizer.c     # UTF-8 text → token IDs
│   │   ├── encoder.c       # 6-layer transformer (rel. position attention)
│   │   ├── duration.c      # Stochastic DP (DDS + RQS + ConvFlows)
│   │   ├── flow.c          # WaveNet + residual coupling flow
│   │   ├── hifigan_q.c     # INT8 HiFi-GAN vocoder (--int8; 4× upsample, MRF)
│   │   ├── hifigan_q16.c   # Q16 fixed-point HiFi-GAN vocoder (default)
│   │   ├── hifigan.c       # F32 HiFi-GAN vocoder (legacy; not built by default)
│   │   ├── vtsm.c          # Weight loader (v2: mmap F32 + int8, zero-copy)
│   │   ├── model.c         # Pipeline orchestration
│   │   ├── npy_reader.c    # Minimal .npy loader (for latent injection)
│   │   └── wav.c           # WAV writer (16-bit PCM)
│   └── validate/
│       ├── stages_ref.py   # Python reference: per-stage .npy dumps
│       ├── compare.py      # .npy vs .bin comparison
│       └── dump.c/h        # C-side dump interface
├── samples/                # Generated audio samples
├── tts.py                  # Python TTS (reference)
└── requirements.txt        # Python deps (validation only)
```

## 📐 Architecture

| Property | Value |
|----------|-------|
| Architecture | VITS (VAE + normalizing flow + HiFi-GAN) |
| Parameters | 36.3M (encoder/flow F32; HiFi-GAN Q16 or INT8) |
| Sampling rate | 16 kHz |
| Vocab | 43 characters (PT-BR) |
| Weight format | `.vtsm` v2 (flat binary, **67.4 MB**) |
| Weight loading | `mmap` + zero-copy for F32 weights (no memcpy); INT8/Q16 decoder params computed in-memory |
| Optimizations | AVX2/FMA (x86_64), NEON (aarch64/ARMv7); INT8 + Q16 fixed-point HiFi-GAN |
| Threading | OpenMP (**on by default**; encoder, flow, HiFi-GAN) |
| License | CC-BY-NC 4.0 (non-commercial) |

## 📄 Notes

- Non-deterministic by default (stochastic duration predictor). Use `--seed` for reproducible output.
- `model.vtsm` (v2, 67.4 MB) is not committed. Generate with `export_weights.py` or restore from backup.
- F32 weight loading is a single `mmap` call — no parsing, no memcpy, no heap allocation. All tensor pointers reference the mapped file region directly, and `free_model()` is a single `munmap`. (The optional Q16 decoder keeps its small INT8/INT16 params on the heap and frees them separately.)
