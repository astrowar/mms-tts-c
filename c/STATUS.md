# MMS TTS (mms-tts-por) — Implementação C Pura

Status: **TODAS AS STAGES VALIDADAS (1-7)** — pipeline C completa (tokenizer → encoder → DP → prior → flow → HiFi-GAN → WAV) casa com o Python ref em todos os stages. Stage 6 foi desbloqueado em 2026-09-06: a causa era o slope do LeakyReLU final (0.1 no C vs 0.01 no Python — `F.leaky_relu(x)` sem argumento usa default 0.01).

## Visão Geral

Implementação em C puro do modelo VITS `facebook/mms-tts-por` (36.3M params, 16 kHz),
referenciando o código do [huggingface/transformers](https://github.com/huggingface/transformers).

| Propriedade | Valor |
|-------------|-------|
| Arquitetura | VITS (VAE + Flow + HiFi-GAN) |
| Parâmetros | 36.3M (F32) |
| Taxa de amostragem | 16 kHz |
| Vocabulário | 43 caracteres (português) |
| Formato pesos | `.vtsm` (binário flat, gerado por `export_weights.py`) |
| Licença | CC-BY-NC 4.0 |

## Formato de Modelo

O binário usa exclusivamente o formato VTSM, gerado pelo script `export_weights.py`
a partir dos pesos HuggingFace (`.safetensors`).

| Formato | Magic | Extensão | Tamanho | Descrição |
|---------|-------|----------|---------|-----------|
| VTSM | `VTSM` | `.vtsm` | ~108 MB | Binário flat, pre-processado (weight_norm fused, ConvT transposed) |

**Uso:** O `main.c` usa `load_vtsm()`. Se `--model` não for fornecido,
procura `./model.vtsm` no diretório atual.

### Formato VTSM

```
Offset  Size  Field
------  ----  ----------------------------------------------------------
0       4     magic bytes "VTSM"
4       4     version (uint32 LE, currently 1)
8       4     vocab_size (uint32 LE, = 43)
12      4     hidden_size (uint32 LE, = 192)
16      8     total_data_size in bytes (uint64 LE)
24      8     reserved (zero)
32      ...   sequential float32 data (113,583,072 bytes)
```

Todos os tensores já vêm pre-processados:
- `weight_norm` → peso final já calculado (`g * v / ||v||`)
- `ConvTranspose1d` → já transpostos para layout `[out, in, k]`

### Gerando o .vtsm

```bash
cd mms-tts-por/c
python3 export_weights.py \
    --input /path/to/model.safetensors \
    --output model.vtsm \
    --header model_weights.h
```

O script `export_weights.py` gera:
1. **`model.vtsm`** — binário flat com todos os pesos (108 MB)
2. **`model_weights.h`** — header C com:
   - Constantes de tamanho para cada tensor (`WTS_*_SIZE`)
   - Estruturas `WtsTensor` e `WtsSection`
   - Tabela flat `wts_tensors[]` com offset/size de cada um dos 488 tensores
   - Tabela `wts_sections[]` com os 18 agrupamentos lógicos
   - Índices `WTS_SEC_*` e helpers `wts_data()` / `wts_data_at()`

## Estrutura de Arquivos

```
mms-tts-por/c/
├── CMakeLists.txt         # Build: auto-detecta arch (x86_64/AVX2, ARM64/NEON, ARMv7/NEON)
├── STATUS.md              ← este arquivo
├── export_weights.py      # Python: safetensors → .vtsm + .h
├── model.vtsm             # Pesos binários (gerado)
├── model_weights.h        # Header C com layout (gerado)
├── src/
│   ├── vits.h             # Structs, constantes, declarações
│   ├── main.c             # CLI (--inject-dir, --dump-dir, etc.)
│   ├── ops_base.c         # Ops escalar: conv1d, conv_transpose1d, depthwise, layernorm
│   ├── ops_neon.c         # Ops otimizados NEON (aarch64 / ARMv7)
│   ├── ops_avx.c          # Ops otimizados AVX2/FMA (x86_64)
│   ├── tokenizer.c        # text → token IDs (UTF-8, lowercase, add_blank)
│   ├── encoder.c          # Transformer encoder (6 layers, relative position)
│   ├── duration.c         # Stochastic Duration Predictor (DDS + RQS)
│   ├── flow.c             # WaveNet + Residual Coupling Flow (reverse)
│   ├── hifigan.c          # HiFi-GAN vocoder (4 stages × 3 MRF)
│   ├── vtsm.c             # Loader .vtsm (memcpy direto via offsets)
│   ├── model.c            # Pipeline de inferência + dump hooks + injection
│   ├── npy_reader.c       # Minimal .npy loader (float32, F-order transpose)
│   └── wav.c              # WAV writer (16-bit PCM mono)
└── validate/
    ├── stages_ref.py      # Python: gera .npy por stage
    ├── compare.py         # Compara .npy vs .bin (rel error)
    ├── dump.h             # Interface de dump (C)
    └── dump.c             # Implementação de dump
```

## Pipeline de Inferência

```
text
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: TOKENIZER                                                  │
│   lowercase → filter non-vocab → strip → [pad, id0, pad, id1, ...] │
│   Output: int32[T], mask[T]                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 2: TEXT ENCODER                                               │
│   Embed(43×192) × √192                                             │
│   → 6× [SelfAttn(2h, rel_pos) → LN → ConvFFN(192→768→192) → LN]   │
│   → Conv1d(192→384) → split → prior_means[192], log_vars[192]     │
│   Output: hidden[192,T], prior_means[192,T], prior_log_vars[192,T] │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 3: DURATION PREDICTOR (stochastic, reverse)                   │
│   conv_pre(192→192) → DDS(3 layers) → conv_proj(192→192)           │
│   latents = randn(2,T) × 0.8                                       │
│   → [CF4 → CF3 → CF2 → EA] (cada com channel flip + RQS reverse)   │
│   log_duration = latents[:, 0, :]                                    │
│   duration = ceil(exp(log_duration) × mask)                         │
│   Output: durations[T], mel_T = sum(durations)                     │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 4: PRIOR SAMPLING                                             │
│   Expand prior_means/log_vars to mel length (repeat by duration)   │
│   latents = means + randn × exp(log_vars) × 0.667                  │
│   Output: latents[192, mel_T]                                       │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 5: FLOW (reverse: prior → spectrogram)                        │
│   4× [channel flip → conv_pre(96→192) → WaveNet(4) → conv_post]   │
│   second_half -= mean                                               │
│   Output: spectrogram[192, mel_T]                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 6: HiFi-GAN DECODER                                           │
│   conv_pre(192→512, k=7)                                           │
│   4× [LeakyReLU → ConvT(upsample) → MRF(3 ResBlocks) / 3]         │
│   LeakyReLU → conv_post(32→1, k=7) → tanh                         │
│   Output: waveform[1, mel_T × 256]                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 7: WAV OUTPUT                                                 │
│   normalize → int16 × 32767 → RIFF/WAVE header → .wav             │
└─────────────────────────────────────────────────────────────────────┘
```

## Estado por Componente

### ✅ Implementados e compilando

| Componente | Arquivo | Detalhes |
|-----------|---------|----------|
| Constantes/Structs | `vits.h` | Todas dims hardcoded para mms-tts-por (vocab=43, hidden=192, etc.) |
| Weight layout | `model_weights.h` | 488 tensores com offset/size, 18 seções, constants `WTS_*` |
| Exporter | `export_weights.py` | Safetensors → .vtsm (flat) + .h (layout C) |
| Conv1d | `ops_{base,neon,avx}.c` | Suporta dilation, padding, depthwise; SIMD dispatch por arch |
| ConvTranspose1d | `ops_{base,neon,avx}.c` | Scatter com stride (upsampling) |
| LayerNorm | `ops_{base,neon,avx}.c` | Channel-first, eps=1e-5 |
| GELU / LeakyReLU / tanh / sigmoid | `ops_{base,neon,avx}.c` | In-place |
| randn (Box-Muller) | `ops_{base,neon,avx}.c` | Com spare para eficiência |
| Tokenizer | `tokenizer.c` | UTF-8, lowercase, vocab 43, filter non-vocab, strip, add_blank |
| WAV writer | `wav.c` | 16-bit PCM mono, normaliza por peak |
| Encoder | `encoder.c` | 6 layers, multi-head attn (2h×96d), relative pos, ConvFFN |
| Duration Predictor | `duration.c` | DDS (3 layers), RQS (10 bins, tail=5), 4 ConvFlows + ElemAffine |
| Flow | `flow.c` | 4 coupling layers, WaveNet (4 layers, weight_norm pre-computed) |
| HiFi-GAN | `hifigan.c` | 4 stages, MRF (kernels 3/7/11, dils 1/3/5), total upsampling ×256 |
| VTSM loader | `vtsm.c` | memcpy direto de offsets; sem parsing, sem transforms |
| Pipeline | `model.c` | Orquestra todos os stages, dump hooks condicionais |
| CLI | `main.c` | --text, --output, --model, --seed, --multi, --dump-dir, --inject-dir |
| Build | `CMakeLists.txt` | Auto-detecta arch; flags: ENABLE_OMP, ENABLE_DUMP |

### ✅ Scripts de Validação

| Script | Função |
|--------|--------|
| `validate/stages_ref.py` | Roda pipeline Python stage-by-stage, salva intermediates `.npy` + `07_output.wav` |
| `validate/compare.py` | Carrega `.npy` (Python) e `.bin` (C), reporta max/mean abs diff e relative error |
| `validate/dump.c/h` | Dump C no formato: `[int32 ndim][int32 dims][float32 data]` |
| `src/npy_reader.c` | Lê `.npy` float32 (detecta fortran_order, transpõe se necessário) |

**Fluxo de validação:**
```bash
cd mms-tts-por/c

# 1. Build C com dumps
mkdir -p build && cd build
cmake .. -DENABLE_DUMP=ON -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

# 2. Gerar referência Python (inclui WAV para verificação auditiva)
python3 validate/stages_ref.py --text "ola, mundo, tudo bem ?" --seed 42 --out-dir ref_out

# 3. Rodar C com dumps
../build/mms-tts --text "ola, mundo, tudo bem ?" --seed 42 --dump-dir c_out

# 4. Comparar
python3 validate/compare.py --ref-dir ref_out --c-dir c_out
```

### 🐛 Bugs Identificados e Corrigidos

| # | Bug | Causa | Correção | Arquivo |
|---|-----|-------|----------|---------|
| 1 | Segfault ao iniciar | `VitsModel` (~100MB) alocado na stack | `calloc(1, sizeof(VitsModel))` | `main.c` |
| 2 | Stack overflow em flow.c | Arrays `float[384*512]` na stack, mas T real (mel) pode ser 4096 | Heap-allocate com tamanho dinâmico | `flow.c` |
| 3 | Stack overflow em duration.c | DDS buffer `float[192*4096]` = 3MB na stack | Heap-allocate | `duration.c` |
| 4 | Heap overflow em encoder.c (attention) | Buffer `o` alocado com `HEAD_DIM*T` (96×T) mas output projection lia `HIDDEN` (192) canais | Alocar `HIDDEN*T`; projeção movida para após o loop de heads | `encoder.c` |
| 5 | Heap overflow em duration.c (channel flip) | Flip usava offset `DP_CHANNELS*T` (=2T, fora do bounds) em vez de `T` para o segundo channel | Corrigido para `latents[T + t]` | `duration.c` |
| 6 | **Attention: OOB em rel_k/rel_v (head 1)** | `rel_k[rel_idx*HEAD_DIM + hd_off + d]` — tensor é `[9,96]` (compartilhado entre heads), não `[9,192]`. Para head 1 (hd_off=96), lia 96 floats fora do buffer | Remover `hd_off` do índice; `rel_k`/`rel_v` são compartilhados entre heads | `encoder.c` |
| 7 | **DP: ordem errada dos ConvFlows** | C usava `flows[fi-1]` (CF3, CF2, CF1) em vez de `flows[fi]` (CF4, CF3, CF2). O CF1 é o "useless vflow" removido pelo Python (`flows[:-2] + [flows[-1]]`) | Alterar para `flows[fi]` | `duration.c` |
| 8 | **DP: ConvFlow sem global_conditioning** | `convflow_reverse` não adicionava o condicionamento global (output de conv_pre+DDS+conv_proj do DP) à entrada do DDS interno de cada ConvFlow | Adicionar parâmetro `global_cond` e somar ao output de `conv_pre` | `duration.c` |
| 9 | **RQS: múltiplos erros de fórmula** | (a) Falta `1/√192` scaling no output de conv_proj; (b) min scaling usava `(1-2*min*bin)` em vez de `(1-min*bin)`; (c) `delta` usava `(u[b]+u[b+1])*0.5` em vez de `h[b]/w[b]`; (d) `intermediate1` usava `h*(delta-1)` em vez de `u[b]+u[b+1]-2*delta`; (e) busca de bin usava `cumwidths` em vez de `cumheights`; (f) derivada de fronteira usava -0.4587 em vez de 1.0 | Reescrita completa de `rqs_prepare` e `rqs_reverse_one` seguindo a fórmula exata de `_rational_quadratic_spline` do transformers | `duration.c` |
| 10 | **Mask: blanks tratados como padding** | C usava `mask[t] = (ids[t] != 0)`, zerando posições de blank (add_blank). No VITS, o attention mask é só para batch padding — blanks participam normalmente | `mask[t] = 1` para todos | `model.c` |
| 11 | **HiFi-GAN: ConvTranspose1d sem padding** | Upsamplers usam padding (4,4,1,1). Sem isso, output length era `(T-1)*stride+k` em vez de `(T-1)*stride+k-2*pad`, causando OOB write | Adicionar campo `pad` ao struct; op calcula `oT = (T-1)*stride - 2*pad + k` e faz bounds check | `ops_*.c`, `vits.h`, `vtsm.c` |
| 12 | **HiFi-GAN: buffer insuficiente** | `max_size = 32 * wave_len` não cobria o overshoot do kernel (ex: stage 0 gera `T*8+7` samples) | `max_size = 32 * wave_len + 8192` | `hifigan.c` |
| 13 | **Ref: expansão do prior na direção errada** | `stages_ref.py` usava `F.pad(valid, (1,0,0,0))[:, :-1]` — shift na dimensão **mel** (colunas) em vez de **T** (linhas). Produzia valores ±1 em vez de one-hot → prior expansion totalmente incorreto | `F.pad(valid, (0,0,1,0))[:-1, :]` — shift na dimensão T, matching `F.pad(valid, [0,0,1,0,0,0])[:, :-1]` do `model.forward()` | `validate/stages_ref.py` |
| 14 | **Ref: re-seed desnecessário no stage 4** | `stages_ref.py` chamava `torch.manual_seed(seed+1)` antes de gerar noise no stage 4, mas o `model.forward()` não re-seeda — o RNG continua do state pós-DP | Remover o re-seed; o RNG natural após o DP randn produz os mesmos valores do E2E | `validate/stages_ref.py` |
| 15 | **Ref: noise gerado na ordem errada** | `torch.randn_like(pm)` com `pm` shape `[1, mel, 192]` alocava valores em ordem (mel, channel). O E2E faz `randn_like(prior_means)` com shape `[1, 192, mel]` — ordem (channel, mel) | `torch.randn(1, flow_size, mel_T)` + `.permute(0,2,1)` | `validate/stages_ref.py` |
| 16 | **Tokenizer: sem strip()** | O HuggingFace tokenizer faz `filter(char in vocab).strip()` — remove spaces leading/trailing após filtrar chars fora do vocab. C só filtrava, sem strip → T=39 vs T=37 | Adicionar pass de strip (remover space id=32 nas pontas da lista de IDs) | `tokenizer.c` |
| 17 | **DDS: pointwise conv in-place corrompe input** | `conv1d(buf, dim, T, &pc, buf)` com in==out: escrever output channel `o` sobrescreve input channels ainda necessários por `o+1...`. Causa divergência de rel=1.56 no output do DDS | Alocar `buf2` separado: `conv1d(buf, dim, T, &pc, buf2)` | `duration.c` |
| 18 | **RQS: layout channel-first vs time-first em `rqs_prepare`** | `rqs_prepare` assumia `proj` em layout `[T][29]` (time-first): `p = proj + t*(29)`. Mas o buffer real é `[29][T]` (channel-first): elemento (channel c, time t) está em `proj[c*T + t]`. Isso fazia o softmax ler 10 valores de **diferentes time steps** no mesmo channel, em vez de 10 channels no mesmo time step → widths/heights/derivs totalmente incorretos (rel > 0.8). A `conv_proj` em si estava correta (rel=4e-4) | Trocar `proj[t*29 + b]` por `proj[b*T + t]` (widths), `proj[(10+b)*T + t]` (heights), `proj[(20+b)*T + t]` (derivs) | `duration.c` |
| 19 | **npy_reader: Fortran-ordered arrays não transpostos** | `stages_ref.py` salva `04_latents_sampled.npy` via `.permute(1,0).numpy()` que produz arrays F-contiguous (column-major). O reader C lia os bytes como row-major → dados transpostos | Detectar `'fortran_order': True` no header; se 2D, transpor em memória ao carregar | `npy_reader.c` |
| 20 | **HiFi-GAN: LeakyReLU final com slope errado** | Python `VitsHifiGan.forward()` usa `F.leaky_relu(x)` (default slope **0.01**) para a última ativação antes de `conv_post`, enquanto dentro do loop de upsampling usa `F.leaky_relu(x, 0.1)`. O C usava 0.1 em ambos → max abs ~3.4e-2 no waveform | `leaky_relu_f(A, final_size, 0.01f)` no `hifigan.c` (line 149) | `hifigan.c` |

### ✅ Status Atual da Validação

| # | Tarefa | Status |
|---|--------|--------|
| A | Build limpo (`-DENABLE_DUMP=ON` sem errors) | ✅ Compila; ASAN limpo (só leaks de pesos — by design) |
| B | Smoke test (gerar WAV sem crash) | ✅ Completa sem crash (ASAN clean) |
| C | Validar Stage 1 (tokenizer) | ✅ Token IDs idênticos (T=37); mask idêntico |
| D | Validar Stage 2 (encoder) | ✅ `02_hidden_cf` rel=3.8e-7, `prior_means` rel=1.7e-7, `prior_log_vars` rel=6e-7 |
| E | Validar Stage 3 (DP conditioning) | ✅ `03_dp_conv_pre` rel=1.2e-6, `03_dp_dds_out` rel=3.9e-4, `03_dp_condition` rel=3e-4 |
| F | Validar Stage 3 (DP flows/RQS) | ✅ **Com latent injection**: todos os 4 flows PASS (rel < 6e-4), duration EXACT |
| G | Validar Stage 4 (prior sampling) | ✅ **Com latent injection**: `04_latents_sampled` EXACT (injected) |
| H | Validar Stage 5 (flow/WaveNet) | ✅ PASS: `05_flow_output` alinha com o Python ref; correções em `flow.c` (in-place overwrite, ordering do reverse flow, flip semantics) resolveram o mismatch |
| I | Validar Stage 6 (HiFi-GAN) | ✅ PASS: todos os intermediários (conv_pre, stages 0-3, pre_tanh) rel < 3e-3; fix: LeakyReLU final slope 0.1→0.01 (bug #20) |
| J | Comparação final waveform C vs Python | ✅ PASS: `06_waveform` rel=3e-3, `07_audio_normalized` rel=3e-3, int16 max diff=1 ULP |
| K | Limpar warnings (const qualifiers, unused vars) | ⏳ |
| L | README com instruções de uso | ✅ Atualizado |

### Resumo da Sessão de Validação

**Bugs no C (6-12):**

A raiz do problema "mel_T > 4096" era a **combinação** de múltiplos bugs:

1. **Mask errada (bug #10)**: Blanks eram tratados como padding → encoder produzia
   zeros em metade das posições → conditioning do DP estava corrompido.

2. **Attention OOB (bug #6)**: Head 1 lia `rel_k`/`rel_v` fora do buffer →
   valores indefinidos no attention output → encoder já divergia do Python.

3. **RQS completamente errado (bug #9)**: 6 sub-erros de fórmula faziam o RQS
   produzir outputs totalmente diferentes (não era só um desvio de 1 ULP).

4. **Flow order (bug #7) + missing conditioning (bug #8)**: Os ConvFlows estavam
   na ordem errada e sem o conditioning global.

**Bugs na referência Python (13-15):**

A referência `stages_ref.py` gerava áudio inválido (correlação ~0 com E2E). Diagnóstico:

5. **Expansão do prior na direção errada (bug #13)**: O padding para criar a
   matriz one-hot era feito na dimensão mel (colunas) em vez de T (linhas).

6. **Re-seed desnecessário (bug #14)**: `torch.manual_seed(seed+1)` no stage 4
   fazia o RNG "pular" para um state diferente do que o `model.forward()` produz.

7. **Ordem do noise (bug #15)**: `randn_like(pm)` com pm `[1, mel, 192]` gera
   valores em ordem (mel, channel). O E2E usa `randn_like(prior_means)` com
   shape `[1, 192, mel]` — ordem (channel, mel).

**Bugs no C (16-17) — sessão anterior:**

8. **Tokenizer sem strip (bug #16)**: O HuggingFace tokenizer faz
   `filter(char in vocab).strip()`. A implementação C só filtrava chars fora
   do vocab, sem remover spaces leading/trailing → T=39 vs T=37.

9. **DDS pointwise conv in-place (bug #17)**: `conv1d(buf, dim, T, &pc, buf)`
   com `in == out`: ao escrever o output do channel `o`, os valores originais
   dos channels `0..o-1` são sobrescritos, corrompendo os outputs subsequentes.
   Resultado: `03_dp_dds_out` com rel=1.56. Fix: buffer separado (`buf2`).

**Bugs no C (18-19) — encontrados nesta sessão (2026-09-05, latent injection):**

10. **RQS layout channel-first vs time-first (bug #18)**: `rqs_prepare`
    assumia `proj` em layout `[T][29]` (time-first): `p = proj + t*29`.
    Mas o buffer real é `[29][T]` (channel-first): elemento (channel c, time t)
    está em `proj[c*T + t]`. O softmax lia 10 valores de *diferentes time steps*
    no mesmo channel, em vez de 10 channels no mesmo time step.
    A `conv_proj` em si estava correta (rel=4e-4) — o bug estava **apenas**
    no reindexing para extrair widths/heights/derivs.
    Diagnóstico: comparei os parâmetros RQS C vs Python e vi que os valores
    de `proj` eram quase idênticos, mas depois do softmax+min_scale os
    cumwidths eram completamente diferentes (C: sum=10.25, Python: sum=10.0).
    Fix: trocar `proj[t*29 + b]` → `proj[b*T + t]`.

11. **npy Fortran-order (bug #19)**: `stages_ref.py` salva
    `04_latents_sampled.npy` via `.permute(1,0).numpy()` — numpy preserva
    a memória original (F-contiguous / column-major). O reader C lia os bytes
    como row-major → dados transpostos (C: [192,111] mas valores de
    [111,192]). Diagnóstico: comparei C dump vs ref e vi shape idêntico
    mas conteúdo diferente. O header .npy tinha `'fortran_order': True`.
    Fix: detectar `fortran_order: True` no header e transpor ao carregar.

**Estado atual (2026-09-06) — TODAS AS STAGES VALIDADAS:**
- ASAN: limpo
- Pipeline C completa sem crash, mel_T=111 (idêntico ao Python)
- **Stages 1-5 casam com o Python reference**:
  - Stage 3 DP flows: todos os 4 (CF4, CF3, CF2, EA) rel < 6e-4
  - Stage 3 duration: **EXACT** (diferença = 0.0)
  - Stage 4 latents: **EXACT** (injetados, zero diff)
  - Stage 5 flow: **PASS**; `05_flow_output` rel=2.6e-6
- **Stage 6 (HiFi-GAN): ✅ VALIDADO** (bug #20 — LeakyReLU final slope):
  - `06_hifi_conv_pre`: rel=1e-6
  - `06_hifi_stage_0..3`: rel < 3e-3
  - `06_hifi_pre_tanh`: rel=3e-3
  - `06_waveform`: rel=3e-3
- **Stage 7 (WAV): ✅ VALIDADO**:
  - `07_audio_normalized`: rel=3e-3
  - `07_audio_int16`: max_abs=1.0 (1 ULP em int16)
  - `07_output.wav`: gerado no dump dir
- **Próximos passos**: limpar warnings restantes (const-qualifier em duration.c/flow.c),
  e opcionalmente adicionar dumps layer-by-layer do encoder para comparação fina.

### Validação Layer-by-Layer

**Primeira medição (ANTES dos fixes no C, bugs #6-#12):**

```
Stage             C vs Python      Causa
─────────────────────────────────────────────────────────────────────
01_token_ids      ✅ idêntico      (T=19, mesmos IDs)
01_attention_mask ❌ 0.5 rel       Bug #10 (blanks ≠ padding)
02_hidden_cf      ❌ 0.135 rel     Bug #6 (attn OOB) + #10 (mask)
02_prior_means    ❌ divergente    Cascata do encoder
02_prior_log_vars ❌ divergente    Cascata do encoder
03_dp_condition   ❌ divergente    Cascata (consequência do encoder)
03_log_duration   ❌ divergente    Bug #9 (RQS) + #7 (order) + #8 (cond)
03_duration       ❌ divergente    Consequência do log_duration
04_latents        ⚠️ shape dif.   C mel_T=19 vs Python mel_T=83 (DP errado)
05_flow_output    ⚠️ shape dif.   Idem
06_waveform       ⚠️ shape dif.   Idem (C: 4864 vs Python: 21248 samples)
```

**Segunda medição (após bugs #1-#17, seed=42, texto="ola, mundo, tudo bem ?"):**

```
Stage                    C vs Python       Rel Error    Status
─────────────────────────────────────────────────────────────────────────
01_token_ids            ✅ idêntico       —           (T=37, mesmos IDs)
01_attention_mask       ✅ idêntico       0.0
02_hidden_cf            ✅ PASS           3.8e-7
02_prior_means          ✅ PASS           1.7e-7
02_prior_log_vars       ✅ PASS           6.0e-7
03_dp_conv_pre          ✅ PASS           1.2e-6
03_dp_dds_out           ✅ PASS           3.9e-4
03_dp_condition         ✅ PASS           3.0e-4
03_log_duration         ⚠️ RNG            —           srand ≠ torch.manual_seed
03_duration             ⚠️ RNG            —           C: sum=70, Py: sum=111
04_latents_sampled      ⚠️ shape dif.    —           C mel_T=70 vs Py mel_T=111
05_flow_output          ⚠️ shape dif.    —           Cascade de mel_T
06_waveform             ⚠️ shape dif.    —           C: 17920 vs Py: 28416 samples
```

**Terceira medição (após bugs #18-#19, latent injection, texto="ola, mundo, tudo bem ?"):**

```
Stage                     C vs Python       Rel Error    Status
───────────────────────────────────────────────────────────────────────────
01_attention_mask         ✅ idêntico       0.0
02_hidden_cf              ✅ PASS           3.8e-7
02_prior_log_vars         ✅ PASS           6.0e-7
02_prior_means            ✅ PASS           1.7e-6
03_dp_condition           ✅ PASS           3.0e-4
03_dp_conv_pre            ✅ PASS           1.2e-6
03_dp_dds_out             ✅ PASS           3.9e-4
03_dp_flow_0_out          ✅ PASS           3.6e-4    (CF4: conditioning + RQS)
03_dp_flow_1_out          ✅ PASS           2.6e-4    (CF3)
03_dp_flow_2_out          ✅ PASS           5.5e-4    (CF2)
03_dp_flow_3_out          ✅ PASS           5.5e-4    (EA)
03_duration               ✅ EXACT          0.0       (mel_T=111, todos ids)
03_log_duration           ✅ PASS           5.5e-4
04_latents_sampled        ✅ EXACT          0.0       (injected from ref)
05_flow_output            ❌ FAIL           9.4e-1    WaveNet bug (ver abaixo)
06_waveform               ❌ FAIL           1.0e+0    Cascade do stage 5
```

**Stage 5 (flow) detalhe — per coupling layer:**

```
Coupling   C vs Python    Rel Error    Nota
──────────────────────────────────────────────────────────────
flow[3]   (coupling 0)   ❌ 1.14       1º flip + WaveNet diverge
flow[2]   (coupling 1)   ❌ 1.03       Cascata
flow[1]   (coupling 2)   ❌ 1.19       Cascata
flow[0]   (coupling 3)   ❌ 0.94       Cascata
```

Diagnóstico: `conv_pre` (96→192, k=1) casa com rel<1e-3. O bug está no
**WaveNet** (4 layers of in_layers + res_skip_layers com weight_norm).
O mean (output de conv_post) difere de C vs Python com correlação ~0,
indicando erro sistêmico nos pesos ou na estrutura do conv.

**Quarta medição (após bugs #20, latent injection, texto="ola, mundo, tudo bem ?") — FINAL:**

```
Stage                     C vs Python       Rel Error    Status
───────────────────────────────────────────────────────────────────────────
01_token_ids              ✅ idêntico       0.0
01_attention_mask         ✅ idêntico       0.0
02_hidden_cf              ✅ PASS           3.8e-7
02_prior_log_vars         ✅ PASS           6.0e-7
02_prior_means            ✅ PASS           1.7e-6
03_dp_condition           ✅ PASS           3.0e-4
03_dp_conv_pre            ✅ PASS           1.2e-6
03_dp_dds_out             ✅ PASS           3.9e-4
03_dp_flow_0_out          ✅ PASS           3.6e-4    (CF4)
03_dp_flow_1_out          ✅ PASS           2.6e-4    (CF3)
03_dp_flow_2_out          ✅ PASS           5.5e-4    (CF2)
03_dp_flow_3_out          ✅ PASS           5.5e-4    (EA)
03_duration               ✅ EXACT          0.0       (mel_T=111)
03_log_duration           ✅ PASS           5.5e-4
04_latents_sampled        ✅ EXACT          0.0       (injected)
05_flow_output            ✅ PASS           2.6e-6
06_hifi_conv_pre          ✅ PASS           1e-6
06_hifi_stage_0           ✅ PASS           2e-6
06_hifi_stage_1           ✅ PASS           2e-6
06_hifi_stage_2           ✅ PASS           3e-6
06_hifi_stage_3           ✅ PASS           2e-6
06_hifi_pre_tanh          ✅ PASS           3e-3
06_waveform               ✅ PASS           3e-3
07_audio_normalized       ✅ PASS           3e-3
07_audio_int16            ✅ PASS           max=1.0   (1 ULP)
───────────────────────────────────────────────────────────────────────────
✅ ALL STAGES PASS
```

**Conclusão:** Todos os 20 bugs identificados foram corrigidos. A pipeline C
pura agora casa com o Python reference (HF transformers) em **todos** os
stages, do tokenizer ao WAV final, com rel < 3e-3 em todos os pontos.
O stage 6 foi desbloqueado pelo bug #20: o LeakyReLU final no HiFi-GAN
usava slope 0.1 (mesmo dos loops de upsampling) enquanto o Python usa o
default 0.01 (`F.leaky_relu(x)` sem argumento).

## Decisões de Design

| Decisão | Justificativa |
|---------|--------------|
| Channel-first `[C][T]` em toda a stack | Consistente com conv1d ops; transformer faz transpose interno |
| Dims hardcoded (não ler config.json) | Modelo fixo (mms-tts-por), simplifica C |
| Weight_norm pre-computado no load | WaveNet usa `weight_g`/`weight_v`; HiFi-GAN já vem fused |
| `.vtsm` flat + offsets no header C | Load = 1 mmap + N pointer assignments; sem parsing, sem mallocs |
| Único formato de pesos (.vtsm) | Pesos sempre exportados via `export_weights.py`; sem parsing de .safetensors em runtime |
| **100% zero-copy (todos os campos = ponteiros no mmap)** | Struct = 6.4 KB metadata; zero heap para pesos; 1 `munmap` libera tudo |
| RQS implementado com `softplus` + `softmax` | Mesma fórmula do `_rational_quadratic_spline` do transformers |
| Stack vs Heap: heap para tudo >64KB (buffers temporários) | Evita segfault com buffers de 1-18MB |
| `#ifdef ENABLE_DUMP` | Dumps são opcionais, não afetam performance de produção |
| `rand()` (libc) + Box-Muller | Sem dependência externa; seed via `srand()` |
| `--inject-dir` para validação | Lê `.npy` do Python ref; bypass randn; enables deterministic comparison |
| `npy_reader.c` (self-contained) | Parser mínimo do formato .npy; detecta fortran_order; sem deps externas |

## Diferenças vs Python (documentadas)

- **Seeds diferentes**: `srand(42)` em C ≠ `torch.manual_seed(42)` em PyTorch.
  Os valores de `randn` serão diferentes entre as implementações.
  **Solução implementada**: `--inject-dir` flag injeta latents do Python ref,
  permitindo validação determinística de todos os stages.
  Alternativas futuras: (b) comparar estatísticas, (c) implementar PCG64.

- **Stages 1-7 (com latent injection)**: ✅ Verificado — todos casam com
  rel_err < 3e-3. Duration: EXACT (mel_T=111). Prior latents: EXACT (injected).
  HiFi-GAN intermediates: rel < 3e-3. Final waveform: rel < 3e-3.

- **LeakyReLU no HiFi-GAN (diferença sutil vs config)**: O Python
  `VitsHifiGan.forward()` usa `F.leaky_relu(x, 0.1)` dentro do loop de
  upsampling, mas `F.leaky_relu(x)` (default 0.01) para a última ativação
  antes de `conv_post`. O C replica este comportamento (bug #20).

- **Arredondamento float32**: Operações em C podem diferir em ±1 ULP do PyTorch
  (ordem de soma, FMA, GELU tanh-approx). Tolerância observada: < 6e-4 para
  blocos com múltiplas ops (DDS, RQS); < 1e-6 para ops simples (conv1d, LN);
  < 3e-3 para o path completo do HiFi-GAN (4 stages × 9 convs + 4 convT).

## Latent Injection (feature nova)

Mecanismo para validação determinística dos stages estocásticos:

```bash
# 1. Gerar referência Python (salva latents como .npy)
python3 validate/stages_ref.py --text "ola, mundo, tudo bem ?" --seed 42 --out-dir ref_out

# 2. Rodar C com injection (lê 03_dp_latents_init.npy e 04_latents_sampled.npy)
./mms-tts --text "ola, mundo, tudo bem ?" --seed 42 --inject-dir ref_out --dump-dir c_out

# 3. Comparar
python3 validate/compare.py --ref-dir ref_out --c-dir c_out
```

**Arquivos injetados:**
| Arquivo .npy | Shape | Ordem | Stage |
|---|---|---|---|
| `03_dp_latents_init.npy` | [2, T] | C-contiguous | Stage 3 (DP latents iniciais) |
| `04_latents_sampled.npy` | [192, mel_T] | **Fortran** (column-major) | Stage 4 (prior latents finais) |

**Implementação:**
- `src/npy_reader.c`: parser minimalista do formato .npy (magic, version,
  header dict, data). Detecta `fortran_order: True` e transpõe 2D arrays.
- `model.c`: se `--inject-dir` fornecido, carrega os .npy e substitui
  `randn_f()` nos stages 3 e 4. Fallback para randn se arquivos não existirem.
- `main.c`: novo flag `--inject-dir DIR`.

**Caveat**: A `04_latents_sampled.npy` é F-contiguous porque o Python faz
`latents[0].permute(1,0).numpy()` — o permute muda a view mas não a memória.
O `np.save()` preserva a ordem original. O reader C detecta e corrige.

## Como Compilar e Rodar

```bash
cd mms-tts-por/c

# 1. Gerar pesos binários (uma vez)
python3 export_weights.py \
    --input ~/.cache/huggingface/hub/models--facebook--mms-tts-por/snapshots/*/model.safetensors \
    --output model.vtsm \
    --header model_weights.h

# 2. Build
make

# 3. Rodar (usa model.vtsm no cwd por padrão)
./mms-tts --text "Olá, mundo!"
./mms-tts --text "Bom dia" --output bom_dia.wav --seed -1
./mms-tts --multi

# Usar outro .vtsm
./mms-tts --model /path/to/other_model.vtsm --text "Olá!"

# Validação determinística com latent injection
./mms-tts --text "ola, mundo, tudo bem ?" --seed 42 \
    --inject-dir ref_out --dump-dir c_out
python3 validate/compare.py --ref-dir ref_out --c-dir c_out

# Build com debug + dumps
mkdir -p build && cd build
cmake .. -DENABLE_DUMP=ON -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)

# Build com AddressSanitizer (debug)
mkdir -p build_asan && cd build_asan
cmake .. -DENABLE_DUMP=ON -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-O0 -g -fsanitize=address -Wall -Wextra -Wno-unused-parameter" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" && make -j$(nproc)

# Validação stage-by-stage (automated)
# 1. python3 validate/stages_ref.py --text "..." --seed 42 --out-dir ref_out
# 2. ./build/mms-tts --text "..." --seed 42 --inject-dir ref_out --dump-dir c_out
# 3. python3 validate/compare.py --ref-dir ref_out --c-dir c_out
```

> **Nota:** Para build de validação, usar `cmake .. -DENABLE_DUMP=ON`.
> O flag é necessário para compilar `validate/dump.c` e ativar os hooks de dump
> em `model.c`.

## Modelo

| Fonte | Caminho | Tamanho |
|-------|---------|---------|
| VTSM (default) | `./model.vtsm` (diretório atual) | ~108 MB |

O `.vtsm` é o único formato suportado: binário flat com todos os pesos
pre-processados (weight_norm fused, ConvTranspose transposed).

Gerado a partir do `.safetensors` do HuggingFace via:
```bash
python3 export_weights.py --input model.safetensors --output model.vtsm --header model_weights.h
```

### Estratégia de memória: mmap + 100% zero-copy

O loader usa `mmap` (file-backed, `PROT_READ | MAP_PRIVATE`) em vez de
`malloc` + `fread`. **Todos** os campos de peso no struct `VitsModel`
são ponteiros que apontam diretamente para a região mapeada — zero memcpy:

| Propriedade | Valor |
|-------------|-------|
| `sizeof(VitsModel)` | 6,584 bytes (6.4 KB) |
| Memória de pesos (heap) | **0 bytes** — tudo no mmap |
| Copy no load | **Nenhuma** |
| `free_model()` | 1 `munmap` |

**Como funciona:**

```
mmap(model.vtsm)  →  região mapeada (108 MB, file-backed, reclaimable)
                         │
    embed_w ─────────────┤
    layers[0].attn.q_w ──┤
    layers[0].ffn1_w ────┤
    dp.conv_pre_w ───────┤
    dp.flows[0].conv_proj_w ──┤
    flow.flows[0].wavenet.in_w[0] ──┤
    decoder.conv_post_w ─┤
    ... (488 ponteiros no total)
```

O struct é alocado com `calloc` (6.4 KB de metadata: ints + pointers).
O `free_model()` faz `munmap` — todos os ponteiros ficam inválidos
simultaneamente. O `free(model)` libera os 6.4 KB.

**Vantagens vs malloc+fread (anterior):**
- **Zero heap para pesos** — ASAN limpo sem suppression file
- **Páginas file-backed reclaimable** — kernel pode evictar sob pressão
  (re-read do disco se necessário); anonymous heap exigiria swap
- **1 `munmap`** libera tudo (antes: 298 `free()`)
- **Pico de RSS reduzido** — sem 49 MB de struct inline em heap
- **Cache-friendly** — pages são compartilhadas via page tables; múltiplos
  processos lendo o mesmo .vtsm compartilham as páginas físicas

**Linha do tempo de memória (texto curto):**
```
open+mmap          →  ~0 MB (page tables only)
inference          → ~133 MB (pages faulted sob demanda, file-backed)
free_model()       →   ~0 MB (munmap libera todas as pages)
free(model)        →   ~0 MB (6.4 KB struct)
```

**Nota:** o `ru_maxrss` (high-water mark) reflete o pico durante o
inference (~133 MB), mas a memória **private** (irreclaimable) é apenas
6.4 KB do struct. Todos os 108 MB de pesos são file-backed e podem ser
evictados pelo kernel sem swap.
