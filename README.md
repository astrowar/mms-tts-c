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
| License | CC-BY-NC 4.0 (non-commercial) |

## Build

```bash
cd c/

# One-time: export weights from HuggingFace safetensors to flat binary
python3 export_weights.py \
    --input ~/.cache/huggingface/hub/models--facebook--mms-tts-por/snapshots/*/model.safetensors \
    --output model.vtsm \
    --header model_weights.h

# Compile
make

# Run
./mms-tts --text "Olá, mundo!"
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
make clean && make DUMP=1

# Generate Python reference
python3 validate/stages_ref.py --text "ola, mundo, tudo bem ?" --seed 42 --out-dir ref_out

# Run C with injection + dumps
./mms-tts --text "ola, mundo, tudo bem ?" --seed 42 --inject-dir ref_out --dump-dir c_out

# Compare
python3 validate/compare.py --ref-dir ref_out --c-dir c_out
```

## Structure

```
├── c/
│   ├── Makefile
│   ├── STATUS.md           # Full implementation & validation documentation
│   ├── export_weights.py   # safetensors → .vtsm + model_weights.h
│   ├── model_weights.h     # Generated: tensor offsets & layout
│   ├── src/
│   │   ├── vits.h          # Structs, constants, declarations
│   │   ├── main.c          # CLI entry point
│   │   ├── ops.c           # conv1d, conv_transpose, layernorm, activations
│   │   ├── tokenizer.c     # UTF-8 text → token IDs
│   │   ├── encoder.c       # 6-layer transformer (rel. position attention)
│   │   ├── duration.c      # Stochastic DP (DDS + RQS + ConvFlows)
│   │   ├── flow.c          # WaveNet + residual coupling flow
│   │   ├── hifigan.c       # HiFi-GAN vocoder (4× upsample, MRF)
│   │   ├── vtsm.c          # Weight loader (1 fread + N memcpys)
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

~50x realtime on a single core (no SIMD/OpenMP). A 10s audio takes ~8 minutes to synthesize.
For production embedded deployment, SIMD optimization (NEON/AVX2) is the next step.

## Notes

- Non-deterministic by default (stochastic duration predictor). Use `--seed` for reproducible output.
- `model.vtsm` is not committed (108 MB). Generate with `export_weights.py` or restore from backup.
- Weight loading is a single `fread` + N `memcpy` calls — no parsing, no runtime transforms.
