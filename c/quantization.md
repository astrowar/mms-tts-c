# Quantização do HiFi-GAN para Inteiro (int8/int16)

## Objetivo

Reduzir o tamanho e custo de memória/computação dos pesos do HiFi-GAN
(44.5 MB em F32) convertendo para inteiros, mantendo o erro de
reconstrução da waveform aceitável.

## Estratégia de Quantização

### Esquema: Simétrico por Canal de Saída

```
Para cada tensor de pesos W [out_ch, in_ch, k]:
    scale[o] = max(|W[o,:,:]|) / QMAX
    Q[o,:,:] = clamp(round(W[o,:,:] / scale[o]), -QMAX-1, QMAX)

Na inferência (dequantização on-the-fly):
    acc[o,t] = Σ_{i,j} Q[o,i,j] × in[i, t+j·dil-pad]
    out[o,t] = acc[o,t] × scale[o] + bias[o]
```

- **Pesos**: int16 (ou int8), armazenados em buffer contíguo
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

### Por que int16 e não int8?

Análise de amplificação de erro por fan-in:

| Camada | Fan-in (in_ch × k) | Erro int8 (est.) | Erro int16 (est.) |
|--------|-------------------|-----------------|-----------------|
| conv_pre | 512 | ~0.001 | ~0.000004 |
| up_0 (k=16) | 8192 | ~0.03 | ~0.00012 |
| ResBlock c1 (k=11, ch=256) | 2816 | ~0.011 | ~0.000043 |
| ResBlock c1 (k=3, ch=32) | 96 | ~0.0004 | ~0.0000016 |

**Regra prática:** erro na saída ≈ `sqrt(fan_in) × scale × 0.5 / QMAX`

Com int8 (QMAX=127), camadas com fan-in ≥ 1000 ultrapassam ~0.03
de erro absoluto por conv. Após 12+ ResBlocks em série com
LeakyReLU, o erro acumulado atinge ~0.74 (62% do sinal) — **inaudível**.

Com int16 (QMAX=32767), o erro é 256× menor em cada conv —
resultado final: **0.074% de erro** na waveform.

## Resultados Finais (após fix de dilation)

Execução: `--text "Olá, mundo! Tudo bem?" --seed 42 --hifi-int8`
(2.02s de áudio, 32258 samples, 16 kHz)

| Camada | max_abs | mean_abs | ref_max | % do sinal |
|--------|---------|----------|---------|------------|
| conv_pre | 0.00046 | 0.00005 | 50.72 | 0.0009% |
| stage_0 (up_0 + MRF) | 0.0034 | 0.00008 | 25.04 | 0.014% |
| stage_1 (up_1 + MRF) | 0.0012 | 0.00003 | 7.06 | 0.017% |
| stage_2 (up_2 + MRF) | 0.0011 | 0.00001 | 3.10 | 0.035% |
| stage_3 (up_3 + MRF) | 0.0015 | 0.00001 | 3.49 | 0.042% |
| pre_tanh | 0.0006 | 0.00001 | 1.06 | 0.056% |
| **waveform (tanh)** | **0.00058** | **0.00001** | **0.79** | **0.074%** |

O erro cresce levemente por estágio (acumulação natural) mas permanece
extremamente baixo. A waveform final difere do F32 por apenas 0.074%
do pico — **inaudível** para qualquer ouvinte.

### Tamanho

| Formato | Tamanho | Redução |
|---------|---------|---------|
| F32 (mmap) | 44.5 MB | — |
| int16 + scales | 27.3 MB | **39%** |
| (teórico int8) | 13.7 MB | 69% |

## Arquitetura Implementada

### Novos arquivos

| Arquivo | Descrição |
|---------|-----------|
| `src/ops_int8.c` | `conv1d_q()` e `conv_transpose1d_q()` — kernel de convolução com pesos int16 |
| `src/hifigan_q.c` | `hifigan_quantize()`, `hifigan_free_q()`, `hifigan_forward_q()` |

### Modificações

| Arquivo | Mudança |
|---------|---------|
| `src/vits.h` | Structs `Conv1dQ`, `ConvTranspose1dQ`, `ResBlockQ`, `HiFiGanQ`; campos `decoder_q` e `use_int8_hifi` em `VitsModel` |
| `src/main.c` | Flag `--hifi-int8`; chamada `hifigan_quantize()` após `load_vtsm()` |
| `src/model.c` | Stage 6: dispatch entre `hifigan_forward` (F32) e `hifigan_forward_q` (int16) |
| `src/vtsm.c` | `free_model()` libera `decoder_q.qdata`/`scales` antes do munmap |
| `CMakeLists.txt` | Novos arquivos `ops_int8.c`, `hifigan_q.c` no build |

### Layout de Memória

```
HiFiGanQ:
  qdata:    int16_t *  — todos os pesos int16 concatenados (27.3 MB)
  scales:   float32 *  — todos os per-channel scales (38 KB, 9633 entries)
  conv_pre, up[0..3], rb[0..11], conv_post:
    weight → aponta dentro de qdata (offset calculado em quantize)
    scale  → aponta dentro de scales
    bias   → aponta para o F32 mmap (não copiado)
```

## Fluxo de Quantização

```
1. load_vtsm()       → F32 weights em mmap (zero-copy)
2. hifigan_quantize() → malloc int16 buffer + scales
   ├─ Percorre todos os 33 tensors do HiFi-GAN
   ├─ Para cada um: encontra max|W[o]| → scale[o] = max/QMAX
   ├─ Quantiza: q = round(w/scale) clamped a [-32768, 32767]
   ├─ Copia dilation/pad/stride do F32 (não hardcode!)
   └─ Grava ponteiros no HiFiGanQ
3. free_model()      → libera qdata/scales + munmap F32
```

A quantização acontece **em tempo de carga** (uma vez), não a cada
inferência. Custo: ~100 ms (ler F32 do mmap + escrever int16).

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

Comparar com `c_out_f32/06_*` via numpy.

## Problemas Identificados

### 1. int8 não funciona para HiFi-GAN (RESOLVIDO por uso de int16)

**Sintoma:** erro de 28% no stage_0, 62% na waveform final.
**Causa:** fan-in alto (8192 no upsampler, 2816 no ResBlock) ×
255 níveis int8 → erro por conv ~0.04-0.12. Acumulado em 12+
ResBlocks com LeakyReLU, o erro cresce exponencialmente.

**Conclusão:** int8 é inviável para este modelo. int16 é o mínimo.

### 2. Dilation hardcoded incorretamente (BUG — RESOLVIDO)

**Sintoma:** conv_pre com 0.0005% de erro (perfeito) mas stage_0 com
28% de erro (idêntico ao int8). Testes isolados provaram que o kernel
`conv1d_q` funciona corretamente com qualquer dilation.

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
int16 e rodar ambos os kernels. Se `conv1d(deq) == conv1d_q(q)`, o
kernel está correto; a diferença deve vir dos metadados (dil/pad).

### 3. Erro por canal dominado pelo maior peso (aceitável)

`scale[o] = max|W[o]|/QMAX` significa que pesos pequenos perdem
precisão. Para conv_pre, os maiores pesos (de 0.157) dominam;
o restante é ~10× menor. Com int16, o impacto é < 0.001%.

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

## Próximos Passos

### Feito ✓
- [x] int16 quantização completa (conv_pre, upsamplers, 12 ResBlocks, conv_post)
- [x] Validação por camada: 0.074% erro na waveform
- [x] Flag `--hifi-int8` funcional
- [x] Memory management (alloc em quantize, free em free_model)

### Futuro

1. **Exportar int16 no .vtsm:** incluir campo de pesos int16 no formato
   VTSM para evitar quantização em runtime (economiza os 44.5 MB de
   F32 HiFi-GAN do mmap)
2. **Quantizar encoder/flow/DP:** DDS (depthwise 1×15=15, pointwise
   192×1=192) têm fan-in pequeno — int8 provavelmente funciona
3. **Per-channel vs per-tensor:** testar se per-tensor int16 é
   suficiente (menos scales, mesmo erro)
4. **SNR/SI-SDR:** medir qualidade objetiva da WAV int16 vs F32
5. **Comparar audição A/B:** blind test F32 vs int16

## Referências

- VITS (Jia et al., 2021): HiFi-GAN com ResBlocks Multi-Receptive
  Field (3 kernels: 3, 7, 11; dilations: 1, 3, 5) + 4 ConvTranspose
- Per-channel symmetric quantization: padrão TFLite/ONNX Runtime
- Fan-in error amplification: análise empírica neste documento
