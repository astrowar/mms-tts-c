# mms-tts-por

Portuguese Text-to-Speech using [facebook/mms-tts-por](https://huggingface.co/facebook/mms-tts-por) (Meta MMS — Massively Multilingual Speech).

## Model

| Property | Value |
|----------|-------|
| Architecture | VITS (VAE + Flow-based acoustic model + HiFi-GAN decoder) |
| Parameters | 36.3M |
| Sampling rate | 16 kHz |
| Format | Safetensors (F32) |
| License | CC-BY-NC 4.0 (non-commercial) |

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default sample
python tts.py

# Custom text
python tts.py --text "Olá, mundo!" --output hello.wav

# Multiple sample texts
python tts.py --multi

# Random seed (non-deterministic)
python tts.py --text "Bom dia" --seed -1
```

## Pure C Implementation

A standalone C implementation (no dependencies beyond libc/libm) lives in [`c/`](c/).
It supports two weight formats:

| Format | File | Size | Notes |
|--------|------|------|-------|
| VTSM (binary) | `model.vtsm` | ~108 MB | Pre-processed, fastest load |
| Safetensors | `model.safetensors` | ~140 MB | HuggingFace native |

```bash
cd c/

# One-time: generate binary weights from safetensors
python3 export_weights.py \
    --input ~/.cache/huggingface/hub/models--facebook--mms-tts-por/snapshots/*/model.safetensors \
    --output model.vtsm \
    --header model_weights.h

# Build and run
make
./mms-tts --text "Olá, mundo!"
```

The C binary auto-detects the format by magic bytes. If `--model` is not given,
it looks for `./model.vtsm` first, then falls back to the HuggingFace cache.

Current validation status: the C pipeline is validated through Stage 5 (speech flow)
and the remaining mismatch is limited to the final HiFi-GAN waveform output.
The decoder issue is currently isolated to the final stage, not the acoustic model.

See [`c/STATUS.md`](c/STATUS.md) for full implementation status, architecture,
and validation details.

## Notes

- The model is **non-deterministic** due to the stochastic duration predictor.
  Use a fixed `--seed` (default: 42) for reproducible output.
- First run downloads the model (~140 MB) and caches it in `~/.cache/huggingface/`.
- Output is 16-bit PCM WAV at 16 kHz.
