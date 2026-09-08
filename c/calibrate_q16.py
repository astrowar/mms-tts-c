#!/usr/bin/env python3
"""
Calibrate Q16 fixed-point exponents for HiFi-GAN.

Runs the FP32 HiFi-GAN on calibration sentences, records the max absolute
value at each tensor boundary, and computes optimal INT16 exponents such
that the quantized values use the full INT16 range without overflowing.

Usage:
    python calibrate_q16.py
    python calibrate_q16.py --texts "Olá, mundo!" "Teste rápido."
    python calibrate_q16.py --margin 0.9 --output exponents.txt
"""

import argparse
import math
import sys
import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer

DEFAULT_TEXTS = [
    "Olá, mundo!",
    "A tecnologia de fala tem avançado rapidamente nos últimos anos.",
    "O modelo MMS foi desenvolvido pela Meta para suportar mais de mil línguas.",
    "É desgracioso, desengonçado, torto. Hércules-Quasímodo, reflete no aspecto a fealdade típica dos fracos.",
    "Ele disse que não entenderia nada do que estava acontecendo ali.",
    "A reunião foi cancelada por causa da chuva forte que caiu a tarde.",
    "Quero comprar um café e um jornal na padaria da esquina.",
    "A resposta do modelo foi surpreendentemente precisa para a pergunta.",
    "Ela caminhava devagar pela praia, pensando no dia que teria pela frente.",
    "Os números mostram um crescimento de vinte por cento no último trimestre.",
    "Não sei onde deixei as chaves do carro ontem à noite.",
    "O professor explicou a teoria das relatividades de forma muito simples.",
    "Vamos marcar um almoço para sábado, que tal o restaurante novo?",
    "A criança brincava no parque enquanto a mãe observava de longe.",
    "Preciso terminar este relatório antes das cinco da tarde.",
    "O avião aterrissou com delay de duas horas por causa do temporal.",
    "Ela cantarolava uma música enquanto preparava o jantar.",
    "O sistema de reconhecimento de voz funciona muito bem em português.",
    "Fiz uma caminhada de trinta minutos pela manhã antes do trabalho.",
    "A pesquisa indica que a maioria dos participantes aprovou o novo método.",
]


def optimal_exp(max_val: float, margin: float = 0.85) -> int:
    """Compute optimal INT16 exponent for a given max absolute value.

    We want: max_val * 2^(-E) <= 32767 * margin
    => 2^(-E) <= 32767 * margin / max_val
    => -E <= log2(32767 * margin / max_val)
    => E >= log2(max_val) - log2(32767 * margin)

    We pick E = ceil(log2(max_val / (32767 * margin)))
    """
    if max_val <= 0:
        return -14
    e = math.ceil(math.log2(max_val / (32767.0 * margin)))
    # Clamp to reasonable range
    e = max(e, -24)
    e = min(e, 0)
    return e


def main():
    parser = argparse.ArgumentParser(description="Calibrate Q16 exponents for HiFi-GAN")
    parser.add_argument("--texts", nargs="*", default=None,
                        help="Calibration texts (default: 20 built-in PT-BR sentences)")
    parser.add_argument("--margin", type=float, default=0.85,
                        help="Headroom fraction (0.85 = use 85%% of INT16 range, default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Write exponents to file (C #define format)")
    parser.add_argument("--model", type=str, default="facebook/mms-tts-por")
    args = parser.parse_args()

    texts = args.texts if args.texts else DEFAULT_TEXTS
    print(f"Loading model: {args.model} ...")
    model = VitsModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    hifigan = model.decoder
    cfg = model.config

    print(f"HiFi-GAN: conv_pre({cfg.flow_size}→{cfg.upsample_initial_channel}, k=7), "
          f"ups=[{cfg.upsample_rates}], RB k={cfg.resblock_kernel_sizes} dil={cfg.resblock_dilation_sizes}")
    print(f"LeakyReLU slope: {cfg.leaky_relu_slope}")
    print(f"Calibrating on {len(texts)} sentences, margin={args.margin} ...")

    # Storage for max absolute values at each point
    stats = {
        "mel": 0.0,
        "conv_pre": 0.0,
        "lr0": 0.0,        # leaky_relu after conv_pre
        "up0": 0.0,        # after upsampler[0]
        "mrf0": 0.0,       # after MRF group 0
        "lr1": 0.0,
        "up1": 0.0,
        "mrf1": 0.0,
        "lr2": 0.0,
        "up2": 0.0,
        "mrf2": 0.0,
        "lr3": 0.0,
        "up3": 0.0,
        "mrf3": 0.0,
        "lr_final": 0.0,   # final leaky_relu before conv_post
        "conv_post": 0.0,  # before tanh
    }

    # We'll manually trace through HiFi-GAN to get all intermediate values
    def run_hifigan_with_stats(spectrogram: torch.Tensor):
        """Run HiFi-GAN and record max |values| at each stage."""
        h = hifigan.conv_pre(spectrogram)
        stats["conv_pre"] = max(stats["conv_pre"], h.abs().max().item())

        for i in range(hifigan.num_upsamples):
            # LeakyReLU
            h_lr = torch.nn.functional.leaky_relu(h, cfg.leaky_relu_slope)
            key_lr = f"lr{i}" if i < 4 else "lr_final"
            stats[key_lr] = max(stats[key_lr], h_lr.abs().max().item())

            # Upsample
            h_up = hifigan.upsampler[i](h_lr)
            stats[f"up{i}"] = max(stats[f"up{i}"], h_up.abs().max().item())

            # MRF: sum of 3 residual blocks / 3
            res = hifigan.resblocks[i * hifigan.num_kernels](h_up)
            for j in range(1, hifigan.num_kernels):
                res = res + hifigan.resblocks[i * hifigan.num_kernels + j](h_up)
            h = res / hifigan.num_kernels
            stats[f"mrf{i}"] = max(stats[f"mrf{i}"], h.abs().max().item())

        # Final LeakyReLU (default slope = 0.01 in torch)
        h = torch.nn.functional.leaky_relu(h)
        stats["lr_final"] = max(stats["lr_final"], h.abs().max().item())

        # conv_post
        h = hifigan.conv_post(h)
        stats["conv_post"] = max(stats["conv_post"], h.abs().max().item())

        return torch.tanh(h)

    # Run calibration
    with torch.no_grad():
        for idx, text in enumerate(texts):
            torch.manual_seed(args.seed + idx)
            inputs = tokenizer(text, return_tensors="pt")
            # Get mel spectrogram by running encoder + flow
            # We need the latent that feeds into HiFi-GAN
            # The VitsModel forward computes: encoder → duration → sampling → flow → decoder
            # We'll hook into the decoder input

            # Alternative: run the full model and capture the spectrogram input to decoder
            mel_buf = []

            def capture_mel(module, inputs):
                mel_buf.append(inputs[0].detach())

            hook = hifigan.register_forward_pre_hook(capture_mel)

            _ = model(**inputs)
            hook.remove()

            if mel_buf:
                mel = mel_buf[0]
                stats["mel"] = max(stats["mel"], mel.abs().max().item())
                run_hifigan_with_stats(mel)

            if (idx + 1) % 5 == 0:
                print(f"  {idx+1}/{len(texts)} done")

    # Compute exponents
    print(f"\n{'='*70}")
    print(f"Calibration results (margin={args.margin}, max over {len(texts)} sentences)")
    print(f"{'='*70}")
    print(f"\n{'Tensor':<15} {'Max |val|':>10} {'Exp':>5} {'Step':>10} {'Eff. bits':>10}")
    print(f"{'-'*55}")

    exps = {}
    for name, max_val in stats.items():
        e = optimal_exp(max_val, args.margin)
        exps[name] = e
        step = 2.0 ** e
        # Effective bits = log2(range/step) = log2(32767*margin/step) ≈ log2(32767*margin) - e
        eff_bits = math.log2(32767 * args.margin) - e if max_val > 0 else 0
        print(f"{name:<15} {max_val:>10.4f} {e:>+5d} {step:>10.6f} {eff_bits:>10.1f}")

    # Map to C code naming
    print(f"\n{'='*70}")
    print("C #define mapping:")
    print(f"{'='*70}")
    print(f"#define Q16_MEL_EXP       {exps['mel']:d}")
    print(f"#define Q16_STAGE0_EXP    {exps['mrf0']:d}   // conv_pre output domain (512ch)")
    print(f"#define Q16_STAGE1_EXP    {exps['mrf1']:d}   // after up[0]+MRF (256ch)")
    print(f"#define Q16_STAGE2_EXP    {exps['mrf2']:d}   // after up[1]+MRF (128ch)")
    print(f"#define Q16_STAGE3_EXP    {exps['mrf3']:d}   // after up[2]+MRF (64ch)")
    print(f"#define Q16_STAGE4_EXP    {exps['mrf3']:d}   // after up[3]+MRF (32ch)")
    print(f"#define Q16_POST_EXP      {exps['conv_post']:d}   // conv_post output (tanh input)")

    # Verify: for each conv layer, check that the alpha computation makes sense
    print(f"\n{'Layer':<20} {'in_exp':>7} {'out_exp':>8} {'alpha scale':>14}")
    print(f"{'-'*52}")
    layers = [
        ("conv_pre",     exps["mel"],    exps["mrf0"]),
        ("up[0]",        exps["mrf0"],   exps["mrf1"]),
        ("up[1]",        exps["mrf1"],   exps["mrf2"]),
        ("up[2]",        exps["mrf2"],   exps["mrf3"]),
        ("up[3]",        exps["mrf3"],   exps["mrf3"]),
        ("conv_post",    exps["mrf3"],   exps["conv_post"]),
    ]
    ALPHA_BITS = 14
    for name, in_e, out_e in layers:
        scale = 2.0 ** (in_e - out_e + ALPHA_BITS)
        print(f"{name:<20} {in_e:>+7d} {out_e:>+8d} {scale:>14.2f}")

    # Write output file
    if args.output:
        with open(args.output, "w") as f:
            f.write("/* Auto-generated by calibrate_q16.py */\n")
            f.write(f"/* Margin: {args.margin}, {len(texts)} calibration sentences */\n\n")
            f.write(f"#define Q16_MEL_EXP       {exps['mel']:d}\n")
            f.write(f"#define Q16_STAGE0_EXP    {exps['mrf0']:d}\n")
            f.write(f"#define Q16_STAGE1_EXP    {exps['mrf1']:d}\n")
            f.write(f"#define Q16_STAGE2_EXP    {exps['mrf2']:d}\n")
            f.write(f"#define Q16_STAGE3_EXP    {exps['mrf3']:d}\n")
            f.write(f"#define Q16_STAGE4_EXP    {exps['mrf3']:d}\n")
            f.write(f"#define Q16_POST_EXP      {exps['conv_post']:d}\n")
        print(f"\nWrote exponents to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
