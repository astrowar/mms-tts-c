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
| F32 (mmap) | 44.5 MB | — |
| int16 + scales | 27.3 MB | 39% |
| **int8 + scales** | **13.7 MB** | **69%** |

### Velocidade (single-thread, máquina de dev, `OMP_NUM_THREADS=1`)

| Modo | Tempo total (texto de 2.02s) | HiFi-GAN |
|------|------------------------------|----------|
| F32 | 9.07 s | 1.00× (baseline) |
| **int8** | **7.24 s (−20%)** | **~1.8×** |

Int8 usa metade dos bytes de pesos (13.7 vs 27.3 MB), melhor
localidade de cache e o mesmo kernel FMA — o ganho do decoder é
~1.8×; end-to-end o ganho é ~20% porque encoder/DP/flows (F32)
dominam o resto do pipeline.

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
| `src/ops_int8.c` | `conv1d_q()` e `conv_transpose1d_q()` — kernel de convolução com pesos int8 |
| `src/hifigan_q.c` | `hifigan_quantize()`, `hifigan_free_q()`, `hifigan_forward_q()` |

### Modificações

| Arquivo | Mudança |
|---------|---------|
| `src/vits.h` | Structs `Conv1dQ`, `ConvTranspose1dQ`, `ResBlockQ`, `HiFiGanQ` (pesos `int8_t`); campos `decoder_q` e `use_int8_hifi` em `VitsModel` |
| `src/main.c` | Flag `--hifi-int8`; chamada `hifigan_quantize()` após `load_vtsm()` |
| `src/model.c` | Stage 6: dispatch entre `hifigan_forward` (F32) e `hifigan_forward_q` (int8) |
| `src/vtsm.c` | `free_model()` libera `decoder_q.qdata`/`scales` antes do munmap |
| `CMakeLists.txt` | Arquivos `ops_int8.c`, `hifigan_q.c` no build |

### Layout de Memória

```
HiFiGanQ:
  qdata:    int8_t *  — todos os pesos int8 concatenados (13.7 MB)
  scales:   float32 *  — todos os per-channel scales (38 KB, 9633 entries)
  conv_pre, up[0..3], rb[0..11], conv_post:
    weight → aponta dentro de qdata (offset calculado em quantize)
    scale  → aponta dentro de scales
    bias   → aponta para o F32 mmap (não copiado)
```

## Fluxo de Quantização

```
1. load_vtsm()       → F32 weights em mmap (zero-copy)
2. hifigan_quantize() → malloc int8 buffer + scales
   ├─ Percorre todos os 33 tensors do HiFi-GAN
   ├─ Para cada um: encontra max|W[o]| → scale[o] = max/127
   ├─ Quantiza: q = round(w/scale) clamped a [-128, 127]
   ├─ Copia dilation/pad/stride do F32 (não hardcode!)
   └─ Grava ponteiros no HiFiGanQ
3. free_model()      → libera qdata/scales + munmap F32
```

A quantização acontece **em tempo de carga** (uma vez), não a cada
inferência. Custo: ~100 ms (ler F32 do mmap + escrever int8).

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

### Futuro

1. **Exportar int8 no .vtsm:** incluir campo de pesos int8 no formato
   VTSM para evitar quantização em runtime (economiza os 44.5 MB de
   F32 HiFi-GAN do mmap)
2. **Quantizar encoder/DP/flows:** DDS (depthwise 1×15=15, pointwise
   192×1=192) têm fan-in pequeno — int8 provavelmente funciona; é
   onde está o resto do tempo (encoder/flows dominam o pipeline F32)
3. **Per-channel vs per-tensor:** testar se per-tensor int8 é
   suficiente (menos scales, mesmo erro)
4. **Comparar audição A/B:** blind test F32 vs int8 (samples em
   `samples_int8/` e `samples_f32/`)

## Referências

- VITS (Jia et al., 2021): HiFi-GAN com ResBlocks Multi-Receptive
  Field (3 kernels: 3, 7, 11; dilations: 1, 3, 5) + 4 ConvTranspose
- Per-channel symmetric quantization: padrão TFLite/ONNX Runtime
- Fan-in error amplification: análise empírica neste documento
