# mms-tts-por — Pure C

⚡ **Zero-dependency, on-device Portuguese TTS.** A pure C implementation of [facebook/mms-tts-por](https://huggingface.co/facebook/mms-tts-por) — Meta MMS (Massively Multilingual Speech) VITS model for Brazilian Portuguese. No Python, no PyTorch, no external libraries. Just `libc` + `libm`.

## ✨ Key Highlights

- **🪶 36.3M Parameters** — compact VITS model (VAE + normalizing flow + HiFi-GAN)
- **📱 Edge-Device Ready** — runs on Raspberry Pi 4/5 with ~133 MB peak memory
- **🔊 16 kHz 16-bit WAV** — direct output, no external upsample required
- **🚫 Zero Dependencies** — single `mmap` for weight loading; no heap allocation for model
- **⚡ 108 MB flat binary** — single file, no parsing, no copy — pointer arithmetic only
- **🧮 Auto-vectorized** — CMake detects AVX2/NEON and picks optimized kernels
- **✅ 100% Validated** — every stage matches the HuggingFace Python reference (< 3e-3 rel. error)

## 🎧 Samples

> Generated with `--seed` for reproducibility. 16 kHz mono, 16-bit PCM.

| Sample | Text | Duration |
|--------|------|----------|
| [hello.wav](samples/hello.wav) | "Olá, mundo!" | 1.3 s |
| [bom_dia.wav](samples/bom_dia.wav) | "Bom dia, como você está hoje?" | 2.8 s |
| [tech.wav](samples/tech.wav) | "A tecnologia de fala avança mais rápido do que imaginamos." | 5.4 s |
| [embedded.wav](samples/embedded.wav) | "Este modelo roda em C puro, sem dependências externas, direto no hardware…" | 9.0 s |

<audio src="samples/hello.wav" controls preload="none"></audio>
<audio src="samples/bom_dia.wav" controls preload="none"></audio>
<audio src="samples/tech.wav" controls preload="none"></audio>
<audio src="samples/embedded.wav" controls preload="none"></audio>

## 🚀 Quick Start

```bash
cd c/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OMP=ON   # OpenMP multi-thread
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
```

## 📊 Performance

Benchmarked on **Raspberry Pi 4 Model B (8 GB)** — Cortex-A72 @ 1.5 GHz × 4, NEON, GCC 14.2.

| Build | Text | Audio | Wall time | Peak RSS | RTF |
|-------|------|-------|-----------|----------|-----|
| NEON (1 thread) | 3 × ~5.2 s sentences | 15.7 s | 335 s | ~133 MB | 21× |
| NEON + OpenMP × 4 | 3 × ~5.2 s sentences | 15.7 s | 97 s | ~133 MB | 5.8× |
| NEON + OpenMP × 4 | "raspberry pi funcionando e falando" | 2.4 s | 15 s (incl. model load) | ~133 MB | ~5× |
| NEON + OpenMP × 4 | "Olá, mundo!" | 1.3 s | 7 s (incl. model load) | ~133 MB | ~2× |

**Memory strategy:** weights live entirely in the `mmap`'d file — file-backed, reclaimable by the kernel under pressure (re-faulted from disk on access, no swap needed). The `VitsModel` struct is only **6.4 KB** of pointers + metadata; **zero heap allocation** for model weights.

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
│   │   ├── ops_base.c      # Scalar ops: conv1d, conv_transpose, layernorm, activations
│   │   ├── ops_neon.c      # NEON-optimized ops (aarch64 / ARMv7)
│   │   ├── ops_avx.c       # AVX2/FMA-optimized ops (x86_64)
│   │   ├── tokenizer.c     # UTF-8 text → token IDs
│   │   ├── encoder.c       # 6-layer transformer (rel. position attention)
│   │   ├── duration.c      # Stochastic DP (DDS + RQS + ConvFlows)
│   │   ├── flow.c          # WaveNet + residual coupling flow
│   │   ├── hifigan.c       # HiFi-GAN vocoder (4× upsample, MRF)
│   │   ├── vtsm.c          # Weight loader (mmap, 100% zero-copy)
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
| Parameters | 36.3M (float32) |
| Sampling rate | 16 kHz |
| Vocab | 43 characters (PT-BR) |
| Weight format | `.vtsm` (flat binary, ~108 MB) |
| Weight loading | `mmap` + zero-copy (no memcpy, no heap for weights) |
| Optimizations | AVX2/FMA (x86_64), NEON (aarch64/ARMv7) |
| Threading | Optional OpenMP (encoder + flow) |
| License | CC-BY-NC 4.0 (non-commercial) |

## 📄 Notes

- Non-deterministic by default (stochastic duration predictor). Use `--seed` for reproducible output.
- `model.vtsm` is not committed (67.4 MB). Generate with `export_weights.py` or restore from backup.
- Weight loading is a single `mmap` call — no parsing, no memcpy, no heap allocation for weights. All 410 tensor pointers reference the mapped file region directly. `free_model()` is a single `munmap`.
