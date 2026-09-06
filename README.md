# mms-tts-por — Pure C

Pure C implementation of [facebook/mms-tts-por](https://huggingface.co/facebook/mms-tts-por) — Meta MMS (Massively Multilingual Speech) VITS model for Portuguese Text-to-Speech.

No runtime dependencies beyond `libc` and `libm`. No Python, no PyTorch, no external libraries.

| Property | Value |
|----------|-------|
| Architecture | VITS (VAE + normalizing flow + HiFi-GAN) |
| Parameters | 36.3M (float32) |
| Sampling rate | 16 kHz |
| Vocab | 43 characters (PT-BR) |
| Weight format | `.vtsm` (flat binary, ~108 MB) |
| Weight loading | `mmap` + zero-copy (no memcpy, no heap for weights) |
| License | CC-BY-NC 4.0 (non-commercial) |

## Build

```bash
cd c/

# One-time: export weights from HuggingFace safetensors to flat binary
python3 export_weights.py \
    --input ~/.cache/huggingface/hub/models--facebook--mms-tts-por/snapshots/*/model.safetensors \
    --output model.vtsm \
    --header model_weights.h

# Compile (CMake auto-detects architecture: x86_64/AVX2, ARM64/NEON, ARMv7/NEON)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release          # single-thread
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OMP=ON   # multi-thread (OpenMP)
make -j$(nproc)

# Run (from project root, so it finds model.vtsm)
cd ..
./build/mms-tts --text "Olá, mundo!"
```

## Usage

```bash
# Default (random seed, output.wav)
./mms-tts --text "Bom dia, como você está?"

# Fixed seed (reproducible)
./mms-tts --text "Olá, mundo!" --seed 42 --output hello.wav

# Random seed
./mms-tts --text "A tecnologia de fala avança rápido." --seed -1

# Multiple sample texts
./mms-tts --multi

# Custom model path
./mms-tts --model /path/to/other.vtsm --text "Teste"
```

## Pipeline

```
text → tokenizer → encoder (6L transformer) → stochastic DP (RQS)
     → prior sampling → normalizing flow (WaveNet) → HiFi-GAN → WAV
```

All 7 stages validated against the HuggingFace `transformers` Python reference
(relative error < 3e-3 at every point). See [`c/STATUS.md`](c/STATUS.md).

## Validation

Deterministic stage-by-stage comparison using latent injection:

```bash
cd c/

# Build with dump support
mkdir -p build && cd build
cmake .. -DENABLE_DUMP=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Generate Python reference
python3 ../validate/stages_ref.py --text "ola, mundo, tudo bem ?" --seed 42 --out-dir ../ref_out

# Run C with injection + dumps
../mms-tts --text "ola, mundo, tudo bem ?" --seed 42 --inject-dir ../ref_out --dump-dir ../c_out

# Compare
python3 ../validate/compare.py --ref-dir ../ref_out --c-dir ../c_out
```

## Structure

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
├── tts.py                  # Python TTS (reference)
└── requirements.txt        # Python deps (validation only)
```

## Performance

Benchmarked on **Raspberry Pi 4 Model B (8 GB)** — Cortex-A72 @ 1.5 GHz × 4, NEON, GCC 14.2.

| Build | Text | Audio | Wall time | Peak RSS | RTF |
|-------|------|-------|-----------|----------|-----|
| NEON (1 thread) | 3 × ~5.2 s sentences | 15.7 s | 335 s | ~133 MB | 21× |
| NEON + OpenMP × 4 | 3 × ~5.2 s sentences | 15.7 s | 97 s | ~133 MB | 5.8× |
| NEON + OpenMP × 4 | "raspberry pi funcionando e falando" | 2.4 s | 15 s (incl. model load) | ~133 MB | ~5× |
| NEON + OpenMP × 4 | "Olá, mundo!" | 1.3 s | 7 s (incl. model load) | ~133 MB | ~2× |

**Memory:** peak RSS is **~133 MB** — weights live entirely in the `mmap`'d file
(file-backed, reclaimable by the kernel). The `VitsModel` struct is only **6.4 KB**
of pointers + metadata; there is **zero heap allocation** for model weights.
The ~126 MB of RSS beyond the struct is temporary activation/vocoder buffers
plus file-backed pages faulted in during inference. Under memory pressure, the
kernel can evict weight pages without swap — they are re-read from disk on access.
Comfortably fits in a Pi 4 (8 GB) or Pi 5 (4/8 GB).

**Speedup:** OpenMP 4-thread delivers ~3.6× over single-thread (near-linear; vocoder is the
dominant bottleneck, encoder/flow parallelize well).

**Suitability:**
- ✅ Batch/offline TTS: fully viable on Pi 4
- ✅ Short phrases (< 2 s audio) with OpenMP: ~3 s latency — acceptable for interactive use
- ⚠️ Long sentences (> 5 s): ~30 s per sentence with 4 threads
- ❌ Real-time streaming: not achievable on Pi 4; Pi 5 (A76 × 4 @ 2.4 GHz) halves the latency

## Notes

- Non-deterministic by default (stochastic duration predictor). Use `--seed` for reproducible output.
- `model.vtsm` is not committed (108 MB). Generate with `export_weights.py` or restore from backup.
- Weight loading is a single `mmap` call — no parsing, no memcpy, no heap allocation for weights.
  All 488 tensor pointers reference the mapped file region directly. `free_model()` is a
  single `munmap`.
