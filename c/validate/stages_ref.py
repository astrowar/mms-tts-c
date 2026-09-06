#!/usr/bin/env python3
"""
Stage-by-stage reference outputs for VITS (mms-tts-por).

Runs each pipeline stage independently and saves intermediates as .npy files.
Use `compare.py` to diff against C-side dumps.

Usage:
    python stages_ref.py --text "Olá, mundo!" --seed 42 --out-dir ./ref_out/
"""

import argparse
import math
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import VitsModel, AutoTokenizer

MODEL_ID = "facebook/mms-tts-por"


def get_model():
    model = VitsModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


def save(name, arr, out_dir):
    path = os.path.join(out_dir, f"{name}.npy")
    arr = np.asarray(arr, dtype=np.float32) if not isinstance(arr, np.ndarray) else arr
    np.save(path, arr)
    shape = arr.shape
    stats = f"shape={shape} mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}"
    print(f"  [{name}] {stats}")
    return path


def run_pipeline(model, tokenizer, text, seed, out_dir):
    """Run the full VITS inference pipeline, dumping each stage."""
    os.makedirs(out_dir, exist_ok=True)

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)

    print(f"Text: {text!r}")
    print(f"Seed: {seed}")
    print()

    # ── Stage 1: Tokenize ──────────────────────────────────────────────
    print("Stage 1: Tokenize")
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]          # [1, T]
    attention_mask = inputs["attention_mask"]  # [1, T]
    T = input_ids.shape[1]
    save("01_token_ids", input_ids[0].numpy().astype(np.int32), out_dir)
    save("01_attention_mask", attention_mask[0].numpy().astype(np.float32), out_dir)
    print(f"  token count: {T}")
    print()

    # ── Stage 2: Text Encoder ──────────────────────────────────────────
    print("Stage 2: Text Encoder")
    with torch.no_grad():
        encoder = model.text_encoder
        # Embedding: embed × sqrt(hidden_size)
        embed_out = encoder.embed_tokens(input_ids) * math.sqrt(model.config.hidden_size)
        save("02_embedding", embed_out[0].numpy(), out_dir)  # [T, 192]

        # Run transformer layers one by one
        hidden = embed_out
        padding_mask = attention_mask  # [1, T]

        for i, layer in enumerate(encoder.encoder.layers):
            # Attention (uses 4D mask, but for all-ones 2D mask → no masking needed)
            # We pass attention_mask=None since there's no padding
            attn_out = layer.attention(hidden, attention_mask=None)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            save(f"02_attn_L{i}", attn_out[0].numpy(), out_dir)  # [T, 192]

            # Residual + LayerNorm
            hidden = layer.layer_norm(hidden + attn_out)
            save(f"02_after_ln1_L{i}", hidden[0].numpy(), out_dir)  # [T, 192]

            # FFN (takes padding_mask as [1, T, 1])
            ffn_out = layer.feed_forward(hidden, padding_mask.unsqueeze(-1))
            save(f"02_ffn_L{i}", ffn_out[0].numpy(), out_dir)  # [T, 192]

            # Residual + LayerNorm
            hidden = layer.final_layer_norm(hidden + ffn_out)
            save(f"02_hidden_L{i}", hidden[0].numpy(), out_dir)  # [T, 192]

        # Final: apply padding_mask (no-op here), then project
        hidden = hidden * attention_mask.unsqueeze(-1)  # [1, T, 192]

        # Project to prior params: Conv1d(192, 384, 1)
        prior = encoder.project(hidden.permute(0, 2, 1))  # [1, 384, T]
        prior = prior.permute(0, 2, 1)                     # [1, T, 384]
        prior = prior * attention_mask.unsqueeze(-1)
        prior_means = prior[:, :, :192]      # [1, T, 192]
        prior_log_vars = prior[:, :, 192:]   # [1, T, 192]

    # Save channel-first for C comparison
    save("02_hidden_cf", hidden[0].permute(1, 0).numpy(), out_dir)       # [192, T]
    save("02_prior_means", prior_means[0].permute(1, 0).numpy(), out_dir)  # [192, T]
    save("02_prior_log_vars", prior_log_vars[0].permute(1, 0).numpy(), out_dir)  # [192, T]
    print()

    # ── Stage 3: Duration Predictor ───────────────────────────────────
    print("Stage 3: Duration Predictor")
    with torch.no_grad():
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)  # re-seed for DP randomness

        dp = model.duration_predictor

        # Conditioning: conv_pre → DDS → conv_proj
        cond_in = hidden.permute(0, 2, 1)  # [1, 192, T]
        cond = dp.conv_pre(cond_in)
        save("03_dp_conv_pre", cond[0].numpy(), out_dir)  # [192, T]

        cond = dp.conv_dds(cond, attention_mask)
        save("03_dp_dds_out", cond[0].numpy(), out_dir)  # [192, T]

        cond = dp.conv_proj(cond) * attention_mask
        save("03_dp_condition", cond[0].numpy(), out_dir)  # [192, T]

        # Sample latents [1, 2, T]
        latents = torch.randn(1, 2, T) * model.config.noise_scale_duration
        save("03_dp_latents_init", latents[0].numpy(), out_dir)  # [2, T]

        # Run flows in reverse: [CF4, CF3, CF2, EA] (skip CF1)
        flows = list(reversed(dp.flows))
        flows = flows[:-2] + [flows[-1]]  # remove CF1, keep EA
        flow_names = [type(f).__name__ for f in flows]
        print(f"  reverse flow order: {flow_names}")

        for i, flow in enumerate(flows):
            latents = torch.flip(latents, dims=[1])
            latents, _ = flow(latents, attention_mask,
                              global_conditioning=cond, reverse=True)
            save(f"03_dp_flow_{i}_out", latents[0].numpy(), out_dir)  # [2, T]

        log_duration = latents[:, 0, :]  # [1, T]
        duration = torch.exp(log_duration) * attention_mask  # [1, T]
        duration = torch.ceil(duration)
        duration = torch.where(attention_mask == 1, duration,
                               torch.zeros_like(duration))
        duration = torch.clamp_min(duration, 0)

    save("03_log_duration", log_duration[0].numpy(), out_dir)  # [T]
    save("03_duration", duration[0].numpy(), out_dir)  # [T]
    total_mel = int(duration.sum().item())
    print(f"  total mel frames: {total_mel}")
    print()

    # ── Stage 4: Prior Expansion + Sampling ───────────────────────────
    print("Stage 4: Prior Expansion & Sampling")
    with torch.no_grad():
        mel_T = total_mel
        if mel_T == 0:
            print("  ⚠️  mel_T=0, skipping stage 4-7")
            return

        # Build expansion matrix (one-hot across T for each mel frame)
        dur = duration[0]  # [T]
        cum_dur = torch.cumsum(dur, dim=0).long()
        indices = torch.arange(mel_T)
        valid = (indices.unsqueeze(0) < cum_dur.unsqueeze(1)).float()  # [T, mel]
        padded = valid - torch.nn.functional.pad(valid, (0, 0, 1, 0), value=0)[:-1, :]
        attn = padded.unsqueeze(0)  # [1, T, mel]

        # Expand prior: for each mel frame, sum over text positions
        attn_t = attn.transpose(1, 2)  # [1, mel, T]
        pm = torch.matmul(attn_t, prior_means[0])      # [1, mel, 192]
        plv = torch.matmul(attn_t, prior_log_vars[0])  # [1, mel, 192]

        # Sample from prior (generate noise channel-first to match E2E RNG order)
        noise_cf = torch.randn(1, model.config.flow_size, mel_T)
        noise = noise_cf.permute(0, 2, 1)  # [1, mel, 192]
        latents = pm + noise * torch.exp(plv) * model.config.noise_scale
        save("04_prior_means_expanded", pm[0].permute(1, 0).numpy(), out_dir)  # [192, mel]
        save("04_prior_log_vars_expanded", plv[0].permute(1, 0).numpy(), out_dir)
        save("04_latents_sampled", latents[0].permute(1, 0).numpy(), out_dir)  # [192, mel]

    print(f"  mel length: {mel_T}")
    print()

    # ── Stage 5: Flow (reverse) ───────────────────────────────────────
    print("Stage 5: Flow (Prior → Spectrogram)")
    with torch.no_grad():
        latents_cf = latents.permute(0, 2, 1)  # [1, 192, mel]
        output_mask = torch.ones(1, 1, mel_T)
        flow_out = model.flow(latents_cf, output_mask, reverse=True)
        save("05_flow_output", flow_out[0].numpy(), out_dir)  # [192, mel]

    spectrogram = flow_out[0]  # keep as tensor
    print(f"  spectrogram: {spectrogram.shape}")
    print()

    # ── Stage 6: HiFi-GAN Decoder ─────────────────────────────────────
    print("Stage 6: HiFi-GAN")
    with torch.no_grad():
        decoder = model.decoder
        x = spectrogram.unsqueeze(0)  # [1, 192, mel_T]

        x = decoder.conv_pre(x)  # [1, 512, mel_T]
        save("06_hifi_conv_pre", x[0].numpy(), out_dir)

        num_up = len(decoder.upsampler)
        num_k = len(decoder.config.resblock_kernel_sizes)
        for i in range(num_up):
            x = F.leaky_relu(x, decoder.config.leaky_relu_slope)
            x = decoder.upsampler[i](x)

            res_state = decoder.resblocks[i * num_k](x)
            for j in range(1, num_k):
                res_state = res_state + decoder.resblocks[i * num_k + j](x)
            x = res_state / num_k
            save(f"06_hifi_stage_{i}", x[0].numpy(), out_dir)

        x = F.leaky_relu(x)  # NOTE: default slope 0.01, NOT config's 0.1
        x = decoder.conv_post(x)  # [1, 1, mel*256]
        save("06_hifi_pre_tanh", x[0, 0].numpy(), out_dir)

        waveform = torch.tanh(x).squeeze()  # [mel*256]

    save("06_waveform", waveform.numpy(), out_dir)
    print(f"  waveform: {waveform.shape[0]} samples, {waveform.shape[0]/16000:.2f}s")
    print()

    # ── Stage 7: Normalize + Int16 ────────────────────────────────────
    print("Stage 7: WAV Output")
    audio = waveform.numpy().astype(np.float32)
    peak = max(np.abs(audio).max(), 1e-8)
    audio_norm = audio / peak
    audio_int16 = (audio_norm * 32767).astype(np.int16)
    save("07_audio_normalized", audio_norm, out_dir)
    save("07_audio_int16", audio_int16.astype(np.float32), out_dir)
    wav_path = os.path.join(out_dir, "07_output.wav")
    sf.write(wav_path, audio_int16, model.config.sampling_rate, subtype="PCM_16")
    print(f"  int16 range: [{audio_int16.min()}, {audio_int16.max()}]")
    print(f"  -> {wav_path} ({os.path.getsize(wav_path) / 1024:.1f} KB, {len(audio_int16) / model.config.sampling_rate:.2f}s)")
    print()

    print("Done. Reference outputs saved to:", out_dir)


def main():
    parser = argparse.ArgumentParser(description="VITS stage-by-stage reference")
    parser.add_argument("--text", type=str, default="Olá, mundo!")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./ref_out")
    args = parser.parse_args()

    model, tokenizer = get_model()
    print(f"Model loaded. Sampling rate: {model.config.sampling_rate} Hz")
    print()
    run_pipeline(model, tokenizer, args.text, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
