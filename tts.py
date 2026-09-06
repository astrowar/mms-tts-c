#!/usr/bin/env python3
"""
MMS TTS Portuguese — Text-to-Speech using Hugging Face facebook/mms-tts-por.

Architecture: VITS (Variational Inference with adversarial learning for end-to-end TTS)
- 36.3M parameters, F32, Safetensors
- Sampling rate: 16 kHz
- Non-deterministic (stochastic duration predictor) — use --seed for reproducibility
- License: CC-BY-NC 4.0

Usage:
    python tts.py
    python tts.py --text "Olá, mundo!"
    python tts.py --text "Olá, mundo!" --output hello.wav
    python tts.py --multi
"""

import argparse
import os
import tempfile
import torch

import soundfile as sf
from transformers import VitsModel, AutoTokenizer

MODEL_ID = "facebook/mms-tts-por"

SAMPLE_TEXTS = [
    "Olá, mundo! Este é um exemplo de síntese de voz em português.",
    "A tecnologia de fala tem avançado rapidamente nos últimos anos.",
    "O modelo MMS foi desenvolvido pela Meta para suportar mais de mil línguas.",
    "É desgracioso, desengonçado, torto. Hércules-Quasímodo, reflete no aspecto "
    "a fealdade típica dos fracos. O andar sem firmeza, sem aprumo, quase "
    "gingante e sinuoso, aparenta a translação de membros desarticulados.",
]


def synthesize(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    text: str,
    seed: int | None = 42,
) -> tuple[torch.Tensor, int]:
    """Generate waveform from text.

    Returns:
        (waveform, sampling_rate)
    """
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    return output, model.config.sampling_rate


def save_wav(waveform: torch.Tensor, sampling_rate: int, path: str) -> None:
    """Save waveform tensor as a 16-bit PCM WAV file."""
    audio = waveform.squeeze().cpu().numpy().astype("float32")
    peak = max(abs(audio).max(), 1e-8)
    audio = audio / peak
    audio_int16 = (audio * 32767).astype("int16")
    sf.write(path, audio_int16, sampling_rate, subtype="PCM_16")
    duration = len(audio_int16) / sampling_rate
    print(f"  -> {path} ({os.path.getsize(path) / 1024:.1f} KB, {duration:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="MMS TTS Portuguese (facebook/mms-tts-por)"
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42, use -1 for random)")
    parser.add_argument("--multi", action="store_true", help="Synthesize all sample texts")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None

    print(f"Loading model: {MODEL_ID} ...")
    model = VitsModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  sampling rate: {model.config.sampling_rate} Hz")

    if args.text:
        texts = [args.text]
        outputs = [args.output]
    elif args.multi:
        texts = SAMPLE_TEXTS
        outputs = [
            os.path.join(tempfile.gettempdir(), f"mms_por_{i+1}.wav")
            for i in range(len(texts))
        ]
    else:
        texts = [SAMPLE_TEXTS[0]]
        outputs = [args.output]

    for i, (text, out_path) in enumerate(zip(texts, outputs), 1):
        print(f"[{i}/{len(texts)}] \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        waveform, sr = synthesize(model, tokenizer, text, seed=seed)
        save_wav(waveform, sr, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
