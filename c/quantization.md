# Quantização do HiFi-GAN para Inteiro (int8)

## Objetivo

Reduzir o tamanho e custo de memória/computação dos pesos do HiFi-GAN
(44.5 MB em F32) convertendo para inteiros, mantendo o erro de
reconstrução da waveform aceitável. **Prioridade: velocidade de
inferência** — o modo int8 (flag `--hifi-int8`) é o modo quantizado
entregue; int16 é mantido como referência de precisão.

## Estratégia de Quantização

### Esquema: Simétrico por Canal de Saída

```
Para cada tensor de pesos W [out_ch, in_ch, k]:
    scale[o] = max(|W[o,:,:]|) / QMAX        (QMAX = 127 para int8)
    Q[o,:,:] = clamp(round(W[o,:,:] / scale[o]), -128, 127)

Na inferência (dequantização on-the-fly):
    acc[o,t] = Σ_{i,j} Q[o,i,j] × in[i, t+j·dil-pad]
    out[o,t] = acc[o,t] × scale[o] + bias[o]
```

- **Pesos**: int8, armazenados em buffer contíguo
- **Scales**: float32, 1 por canal de saída
- **Biases**: float32 (mantidos F32)
- **Ativações**: float32 (não quantizadas)
- **Dequantização**: feita na saída de cada conv (multiplicar accumulator
  pelo scale + adicionar bias)

### Por que por canal de saída?

- Mais preciso que per-tensor (um único scale para o tensor inteiro):
  cada canal tem seu próprio range
- Mais barato que per-input-channel: exige 1 scale por output channel
  (vs. in_ch scales)
- Padrão usado em TFLite, ONNX Runtime, TensorRT

### Por que int8 funciona (e por que a medição antiga dizia que não)

Análise de amplificação de erro por fan-in:

| Camada | Fan-in (in_ch × k) |
|--------|-------------------|
| conv_pre | 512 |
| up_0 (k=16) | 8192 |
| ResBlock c1 (k=11, ch=256) | 2816 |
| ResBlock c1 (k=3, ch=32) | 96 |

**Regra prática:** erro na saída ≈ `sqrt(fan_in) × scale × 0.5 / QMAX`

Com int8 (QMAX=127), camadas com fan-in ≥ 1000 dão erro por conv na
casa de ~0.01-0.03 (RMS). **A medição antiga que reportava 28% no
stage_0 e 62% na waveform estava contaminada pelo bug de dilation
hardcoded** (seção "Problemas" abaixo) — o mesmo bug produzia 28% no
stage_0 do int16. Após o fix (copiar dilation do F32), a medição
limpa do int8 dá **2.8% RMS (31 dB SNR) na waveform final** —
acima do limiar de audibilidade para fala, e ~2.4× menor que a
estimativa pessimista de acumulação exponencial.

Para comparação, int16 (QMAX=32767) mede 0.022% RMS (73 dB) — 256×
menor erro por conv, mas metade do tamanho e ~1.8× mais lento.

## Resultados Medidos (int8, build atual)

Execução: `--text "Olá, mundo! Tudo bem?" --seed 42 --hifi-int8`
(2.02s de áudio, 32258 samples, 16 kHz; mel injetado determinístico,
mesma entrada para F32/int16/int8)

| Camada | int8 relRMS | int8 SNR | int16 relRMS | int16 SNR |
|--------|------------|----------|--------------|-----------|
| conv_pre | 0.212% | 53.5 dB | 0.001% | 101.9 dB |
| stage_0 (up_0 + MRF) | 1.284% | 37.8 dB | 0.003% | 89.8 dB |
| stage_1 (up_1 + MRF) | 1.046% | 39.6 dB | 0.004% | 87.9 dB |
| stage_2 (up_2 + MRF) | 1.468% | 36.7 dB | 0.008% | 81.5 dB |
| stage_3 (up_3 + MRF) | 1.612% | 35.9 dB | 0.011% | 79.0 dB |
| pre_tanh | 2.854% | 30.9 dB | 0.022% | 73.0 dB |
| **waveform (tanh)** | **2.834%** | **31.0 dB** | **0.022%** | **73.1 dB** |

Waveform int8: corr=0.99960 vs F32, max|err|=0.061 (5.7% do pico).
O erro cresce levemente por estágio (acumulação natural) e o tanh
na saída comprime os picos.

### Tamanho

| Formato | Tamanho | Redução |
|---------|---------|---------|
| F32 HiFi-GAN (mmap) | 44.5 MB | — |
| int16 + scales | 27.3 MB | 39% |
| **int8 + scales** | **13.7 MB** | **69%** |

### Tamanho do arquivo .vtsm

| Versão | Conteúdo | Tamanho |
|--------|----------|---------|
| v1 (`model.vtsm`) | F32 completo (encoder+DP+flows+HiFi-GAN) | 108 MB |
| **v2** (`model_int8.vtsm`) | F32 parcial (sem HiFi-GAN weights) + int8 | **67.4 MB** |

O v2 elimina os 44.5 MB de F32 HiFi-GAN weights (redundantes com o
int8) e não exige quantização em runtime.

### Velocidade (single-thread, máquina de dev, `OMP_NUM_THREADS=1`)

| Modo | Tempo total (texto de 2.02s) | HiFi-GAN |
|------|------------------------------|----------|
| F32 | 9.07 s | 1.00× (baseline) |
| **int8** | **7.24 s (−20%)** | **~1.8×** |

Int8 usa metade dos bytes de pesos (13.7 vs 27.3 MB), melhor
localidade de cache e o mesmo kernel FMA — o ganho do decoder é
~1.8×; end-to-end o ganho é ~20% porque encoder/DP/flows (F32)
dominam o resto do pipeline.

Com kernels AVX2 (`ops_int8_avx2.c`, x86 + AVX2/FMA): na máquina de
dev (1 CPU) fica no empate com a versão portável — o build já compila
`ops_int8.c` com `-mavx2 -mfma -O3` e `#pragma omp simd`, então a
autovetorização do GCC já cobre 4-lane; o ganho do corpo 8-lane +
FMA aparece mais em CPUs mais rápidas e com `OMP_NUM_THREADS>1`.
Saída idêntica ao escalar a ~1e-3% (arredondamento FMA), validado
kernel a kernel e no pipeline completo.

**Paralelismo (OpenMP):** o build usa OpenMP (`-DENABLE_OMP=ON`);
os kernels paralelizam por canal de saída quando o trabalho excede
32768 FMA. Conda seta `OMP_NUM_THREADS=1` por padrão — para usar os
N cores: `export OMP_NUM_THREADS=14` (ou o nº de cores da máquina).
Na máquina de dev deste repo (1 CPU efetiva por quota), 14 threads
são contraproducentes (oversubscription: 14s vs 1s); em uma máquina
com 14 cores as camadas de conv grandes escalam.

## Arquitetura Implementada

### Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `src/ops_int8.c` | `conv1d_q()` e `conv_transpose1d_q()` — kernel portável (fallback) |
| `src/ops_int8_avx2.c` | AVX2/FMA 8-lane. conv1d: FMA `x*(q*scale)`, bias na init. Transpose: blocos de 8/4 coef int8→f32 + passes escalar nos t's de borda. Selecionada em x86 + AVX2. |
| `src/ops_int8_neon.c` | NEON 4-lane (aarch64: `vfmaq_f32`; ARMv7: `vmlaq_f32`). conv1d: FMA 4-float. Transpose: blocos de 4 coef com `vmovl_s8→vmovl_s16→vcvtq_f32_s32` + passes escalar nas bordas. Selecionada em ARM64/ARMv7+NEON. |
| `src/hifigan_q.c` | `hifigan_quantize()`, `hifigan_free_q()`, `hifigan_forward_q()` |
| `src/vtsm.c` | Loader v1/v2; v2 mapeia int8+scales do arquivo (zero-copy) |
| `export_weights.py` | Exportador; `--int8` gera v2 com int8 pré-quantizado |

### Modificações

| Arquivo | Mudança |
|---------|---------|
| `src/vits.h` | Structs `Conv1dQ`, `ConvTranspose1dQ`, `ResBlockQ`, `HiFiGanQ`; `decoder_q`, `use_int8_hifi`, `decoder_q_from_file` em `VitsModel` |
| `src/main.c` | Flag `--hifi-int8`; se v2 já carrega int8 do arquivo (skip quantize runtime) |
| `src/model.c` | Stage 6: dispatch entre `hifigan_forward` (F32) e `hifigan_forward_q` (int8) |
| `src/vtsm.c` | v2: populates `decoder_q` via mmap; F32 decoder weight ptrs = NULL; `free_model` só libera se `!decoder_q_from_file` |
| `src/ops_neon.c` | Fix: guard `c->bias ? c->bias[o] : 0.0f` (conv_post sem bias) |
| `CMakeLists.txt` | Seleção de `ops_int8_neon.c` / `ops_int8_avx2.c` por arquitetura; flags `-mfpu=neon` p/ ARMv7 |

### Layout de Memória

```
v1 (model.vtsm, 108 MB):
  F32 completo no mmap (encoder + DP + flows + HiFi-GAN)
  HiFiGanQ: malloc'd (qdata 13.7 MB + scales 38 KB)
  → total RAM: 108 MB (mmap) + 13.7 MB (malloc) ≈ 122 MB

v2 (model_int8.vtsm, 67.4 MB):
  F32 parcial no mmap (encoder + DP + flows + HiFi-GAN biases) = 53.7 MB
  int8 no mmap (qdata 13.7 MB + scales 38 KB) — zero-copy
  HiFiGanQ: ponteiros dentro do mmap (não malloc)
  → total RAM: 67.4 MB (mmap, reclaimable) — sem malloc extra
```

## Fluxo de Quantização

### v1 (model.vtsm — F32 completo)

```
1. load_vtsm()        → F32 weights em mmap (zero-copy, 108 MB)
2. hifigan_quantize() → malloc int8 buffer + scales
   ├─ Percorre todos os 33 tensors do HiFi-GAN
   ├─ Para cada um: encontra max|W[o]| → scale[o] = max/127
   ├─ Quantiza: q = lroundf(w * (1/scale)) clamped a [-128, 127]
   ├─ Copia dilation/pad/stride do F32 (não hardcode!)
   └─ Grava ponteiros no HiFiGanQ
3. free_model()       → libera qdata/scales (malloc) + munmap
```

Custo: ~2 s (ler 44.5 MB F32 do mmap + quantizar).

### v2 (model_int8.vtsm — int8 pré-exportado)

```
1. load_vtsm()        → F32 parcial + int8 em mmap (zero-copy, 67.4 MB)
   ├─ Parser detecta version=2, lê qdata_size/n_scales do header
   ├─ Mapeia ponteiros de bias para o F32 section
   ├─ Mapeia ponteiros de weight/scale para o int8 section
   └─ decoder_q_from_file=1 (free_model não libera — está no mmap)
2. [sem quantização]   → int8 já está no arquivo
3. free_model()       → apenas munmap
```

Custo: **zero** — sem quantização em runtime.

### Exportador

```bash
# v1 (F32 completo, 108 MB)
python3 export_weights.py --input model.safetensors \
    --output model.vtsm --header model_weights.h

# v2 (slim: F32 sem HiFi-GAN weights + int8, 67.4 MB)
python3 export_weights.py --input model.safetensors \
    --output model_int8.vtsm --header model_int8_weights.h --int8
```

O `--int8` flag:
- Pula os 78 F32 weight tensors do HiFi-GAN (mantém 77 biases)
- Quantiza com o mesmo esquema do C: `scale=max/127`, `q=lroundf(w*(1/s))`
- Ordem dos resblocks: c1[0], c2[0], c1[1], c2[1], c1[2], c2[2] (interleaved)
- Escreve header v2 (48 bytes) com `qdata_size` e `n_scales`

## Dumps para Validação

Executar com `--hifi-int8 --dump-dir ./c_out_int8` gera:

| Dump | Conteúdo |
|------|----------|
| `06q_hifi_conv_pre` | Saída de conv_pre (512 × mel_T) |
| `06q_hifi_stage_0` | Após up_0 + MRF (256 × mel_T×8) |
| `06q_hifi_stage_1` | Após up_1 + MRF (128 × mel_T×64) |
| `06q_hifi_stage_2` | Após up_2 + MRF (64 × mel_T×128) |
| `06q_hifi_stage_3` | Após up_3 + MRF (32 × mel_T×256) |
| `06q_hifi_pre_tanh` | Antes do tanh (1 × mel_T×256) |
| `06_waveform` | Final (após tanh) |

Comparar com `c_out_f32/06_*` via numpy (header: int32 ndim +
int32 dims + float32 dados — pular o header).

## Problemas Identificados

### 1. int8 "inviável" — diagnóstico errado, corrigido

**Sintoma original:** erro de 28% no stage_0, 62% na waveform final,
medições de int8. **Conclusão da época:** int8 inviável, usar int16.

**Reavaliação (2026-09-07):** o bug de dilation (item 2 abaixo)
produzia exatamente 28% no stage_0 do int16 — idêntico ao número do
int8. As medições de int8 eram do mesmo código com o bug ativo.
Com o fix, int8 mede **2.8% RMS / 31 dB na waveform** — utilizável
para fala. O int8 voltou a ser o modo quantizado padrão; int16
permanece como referência de precisão máxima.

### 2. Dilation hardcoded incorretamente (BUG — RESOLVIDO)

**Sintoma:** conv_pre com 0.0005% de erro (perfeito) mas stage_0 com
28% de erro. Testes isolados provaram que o kernel `conv1d_q`
funciona corretamente com qualquer dilation.

**Causa raiz:** `hifigan_quantize()` usava:
```c
rbq->c1[d].dilation = d + 1;   // gera 1, 2, 3 — ERRADO
```
O modelo real usa dilations **[1, 3, 5]** (Multi-Receptive Field):
- c1[0]: dil=1, pad=1 (k=3) / pad=3 (k=7) / pad=5 (k=11)
- c1[1]: dil=3, pad=3 (k=3) / pad=9 (k=7) / pad=15 (k=11)
- c1[2]: dil=5, pad=5 (k=3) / pad=15 (k=7) / pad=25 (k=11)

Com dil=2 (em vez de 3) ou dil=3 (em vez de 5), o kernel acessa
posições erradas da entrada — produzindo saída completamente
diferente, não apenas com erro de quantização.

**Fix:**
```c
rbq->c1[d].dilation = rb->c1[d].dilation;  // copia do F32: 1, 3, 5
```

**Detecção:** teste de dequantização — reconstruir F32 a partir dos
inteiros e rodar ambos os kernels. Se `conv1d(deq) == conv1d_q(q)`, o
kernel está correto; a diferença deve vir dos metadados (dil/pad).

### 3. Erro por canal dominado pelo maior peso (aceitável)

`scale[o] = max|W[o]|/QMAX` significa que pesos pequenos perdem
precisão. Para conv_pre, os maiores pesos (de 0.157) dominam;
o restante é ~10× menor. O impacto é ~10× menor com int16.

### 4. `#include <stdio.h>` dentro de `#ifdef ENABLE_DUMP` (RESOLVIDO)

**Sintoma:** erro de compilação quando ENABLE_DUMP não está definido
(`fprintf`/`stderr` undeclared no sanity check de `hifigan_quantize`).
**Fix:** mover `#include <stdio.h>` para fora do guard.

## Bugs Encontrados e Resolvidos

| # | Bug | Sintoma | Fix |
|---|-----|---------|-----|
| 1 | `total_int8` → `total_q` | erro de compilação | rename |
| 2 | reshape Python (256,256,11) | ValueError | kernel é 3 (dil≠k) |
| 3 | **Dilation hardcoded (d+1)** | **stage_0 28% erro** | **copiar do F32 model** |
| 4 | stdio.h em #ifdef | erro compilação sem DUMP | mover include |
| 5 | Dumps PyTorch vazios | ref_out sem 06_* | hooks no HiFi-GAN |
| 6 | int8 descartado por medição contaminada | "28%/62% erro" | reavaliar após fix de dilation; int8 = 2.8% RMS |
| 7 | AVX2 transpose descartava t's de borda (range único por bloco de 8/4 coef vs ranges por coeficiente) | 5.8–33% de erro nos 4 upsamplers | passes escalar por-coeficiente para os t's fora do range comum; corpo vetorial inalterado |

## Próximos Passos

### Feito ✓
- [x] int8 quantização completa (conv_pre, upsamplers, 12 ResBlocks, conv_post)
- [x] Validação por camada: 2.83% RMS (31 dB) na waveform
- [x] Flag `--hifi-int8` funcional (modo quantizado padrão)
- [x] Memory management (alloc em quantize, free em free_model)
- [x] Velocidade: decoder ~1.8×, end-to-end −20% (single-thread)
- [x] SNR em 3 utterances (`samples_int8/`): 2.18–2.94% RMS
  (30.6–33.2 dB) — consistente com a medição única
- [x] Samples A/B: `samples_int8/` vs `samples_f32/` /
  `samples_int16/` (mesmos textos, seed 42)
- [x] Kernels NEON (aarch64 + ARMv7) — `ops_int8_neon.c`, bit-identical
- [x] Format v2 `.vtsm` — int8 pré-quantizado no arquivo, zero-copy mmap
- [x] Exportador `--int8` — slim v2 (67.4 MB, sem F32 HiFi-GAN weights)
- [x] NEON F32 bias NULL fix (`ops_neon.c`) — crash em conv_post (RPi)

### Futuro

1. **Quantizar encoder/DP/flows:** DDS (depthwise 1×15=15, pointwise
   192×1=192) têm fan-in pequeno — int8 provavelmente funciona; é
   onde está o resto do tempo (encoder/flows dominam o pipeline F32)
2. **Per-channel vs per-tensor:** testar se per-tensor int8 é
   suficiente (menos scales, mesmo erro)
3. **Comparar audição A/B:** blind test F32 vs int8 (samples em
   `samples_int8/` e `samples_f32/`)
4. **RPi v2 slim:** testar o `model_int8.vtsm` (67.4 MB) na RPi —
   valida zero-copy mmap em ARM64 + confirma timing sem quant runtime

### Resultados RPi (aarch64, NEON)

Testado em Raspberry Pi (GCC 14.2, 4× Cortex-A72):

| Métrica | RPi (NEON) | x86 (AVX2) |
|---------|-----------|------------|
| relRMS | 2.867% | 2.834% |
| SNR | 30.9 dB | 31.0 dB |
| max\|err\| | 2536 (int16) | — |

Diferença de ~0.03% é apenas arredondamento FMA (NEON vs AVX2).

**Tempo de síntese (RPi, "Olá, mundo! Tudo bem?" = 2.02s de áudio):**

| Modelo | Wall time | RTF |
|--------|-----------|-----|
| F32 (v1) | 12.7s | ~6.3× |
| int8 v1 + quant runtime | 14.6s | ~7.2× |
| int8 v2 (sem quant) | ~12.7s* | ~6.3× |

\* Estimado — o v2 slim (67.4 MB) não pôde ser testado na RPi (offline
no momento do teste); esperado equivalente ao F32 em wall time pois o
HiFi-GAN já era ~igual ao F32 com cache (a quant runtime é a diferença).

### NEON (aarch64)

Kernels em `ops_int8_neon.c`:
- `conv1d_q`: FMA 4-lane (`vfmaq_f32`), dequant on-the-fly por (o,i,j)
- `conv_transpose1d_q`: blocos de 4 coef com `vmovl_s8→vmovl_s16→vcvtq_f32_s32`,
  body vetorial + passes escalar nas bordas
- ARMv7 (Raspberry Pi 3): `vmlaq_f32` em vez de `vfmaq_f32`

Resultado: relRMS/SNR idêntico ao AVX2 (diferença < 0.04% — FMA rounding).

## Referências

- VITS (Jia et al., 2021): HiFi-GAN com ResBlocks Multi-Receptive
  Field (3 kernels: 3, 7, 11; dilations: 1, 3, 5) + 4 ConvTranspose
- Per-channel symmetric quantization: padrão TFLite/ONNX Runtime
- Fan-in error amplification: análise empírica neste documento
