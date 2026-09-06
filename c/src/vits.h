#ifndef VITS_H
#define VITS_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdbool.h>

/* ============================================================
 * Model dimensions — hardcoded for facebook/mms-tts-por
 * ============================================================ */
#define VOCAB_SIZE        43
#define HIDDEN            192
#define NUM_HEADS         2
#define HEAD_DIM          96
#define NUM_LAYERS        6
#define WINDOW            4
#define REL_SIZE          (2 * WINDOW + 1)   /* 9 */
#define FFN_DIM           768
#define FFN_KERNEL        3
#define FLOW_SIZE         192
#define HALF_FLOW         96
#define WAVE_KERNEL       5
#define NUM_PRIOR_FLOWS   4
#define NUM_WAVE          4                  /* wavenet layers per coupling */
#define DP_BINS           10
#define DP_KERNEL         3
#define DP_NUM_FLOWS      4
#define DP_TAIL           5.0f
#define DP_CHANNELS       2
#define DP_DDS_LAYERS     3
#define HIFI_INIT_CH      512
#define NUM_UP            4
#define RF_DILS           3                  /* dilations per MRF block */
#define TOTAL_UP          256                /* 8*8*2*2 */
#define SAMPLE_RATE       16000
#define NOISE_SCALE       0.667f
#define NOISE_SCALE_DUR   0.8f
#define LN_EPS            1e-5f
#define LEAKY_SLOPE       0.1f
#define ATTEN_SCALE       (1.0f / sqrtf(HEAD_DIM))
#define EMBED_SCALE       sqrtf((float)HIDDEN)

/* Max sequence lengths */
#define MAX_TEXT_LEN      512                /* max chars in input */
#define MAX_TOK_LEN       (2 * MAX_TEXT_LEN + 1)  /* with blanks */
#define MAX_MEL_LEN       4096               /* max mel frames */
#define MAX_WAV_LEN       (MAX_MEL_LEN * TOTAL_UP)

/* ============================================================
 * Basic building blocks
 * ============================================================ */

typedef struct {
    int in_ch, out_ch, k, pad, dilation;
    float *weight;   /* [out_ch][in_ch][k]  row-major */
    float *bias;     /* [out_ch] */
} Conv1d;

typedef struct {
    int in_ch, out_ch, k, stride, pad;
    float *weight;   /* [out_ch][in_ch][k] */
    float *bias;     /* [out_ch] */
} ConvTranspose1d;

typedef struct {
    int dim;
    float *weight;   /* [dim] */
    float *bias;     /* [dim] */
} LayerNorm;

/* Dilated Depth-Separable Conv block */
typedef struct {
    int num_layers;
    int dim;
    int kernel;
    /* depthwise: weight shape [dim][1][kernel] */
    float *dw_w[3];  float *dw_b[3];
    /* pointwise: weight shape [dim][dim][1] */
    float *pw_w[3];  float *pw_b[3];
    LayerNorm norm1[3];
    LayerNorm norm2[3];
} DDS;

/* Self-attention with relative positions */
typedef struct {
    float q_w[HIDDEN * HIDDEN], q_b[HIDDEN];
    float k_w[HIDDEN * HIDDEN], k_b[HIDDEN];
    float v_w[HIDDEN * HIDDEN], v_b[HIDDEN];
    float o_w[HIDDEN * HIDDEN], o_b[HIDDEN];
    float rel_k[REL_SIZE * HEAD_DIM];
    float rel_v[REL_SIZE * HEAD_DIM];
} Attn;

typedef struct {
    Attn attn;
    /* FFN: Conv1d(192, 768, 3) -> ReLU -> Conv1d(768, 192, 3) */
    float ffn1_w[FFN_DIM * HIDDEN * FFN_KERNEL];
    float ffn1_b[FFN_DIM];
    float ffn2_w[HIDDEN * FFN_DIM * FFN_KERNEL];
    float ffn2_b[HIDDEN];
    LayerNorm ln1, ln2;
} EncLayer;

/* Elementwise affine (DP flow[0]) */
typedef struct {
    float translate[DP_CHANNELS];
    float log_scale[DP_CHANNELS];
} ElemAffine;

/* RQS coupling layer (DP flows 1..4) */
typedef struct {
    float conv_pre_w[HIDDEN], conv_pre_b[HIDDEN];  /* Conv1d(1,192,1) */
    DDS dds;
    float conv_proj_w[(DP_BINS * 3 - 1) * HIDDEN]; /* Conv1d(192, 29, 1) */
    float conv_proj_b[DP_BINS * 3 - 1];            /* 29 = 10+10+9 */
} ConvFlow;

/* Stochastic Duration Predictor */
typedef struct {
    float conv_pre_w[HIDDEN * HIDDEN], conv_pre_b[HIDDEN];
    DDS conv_dds;
    float conv_proj_w[HIDDEN * HIDDEN], conv_proj_b[HIDDEN];
    /* Flows for inference (reverse): [ElemAffine, ConvFlow x 4] */
    ElemAffine flow_0;
    ConvFlow  flows[DP_NUM_FLOWS];   /* flows[1..4] */
} StochDP;

/* WaveNet (weight-normalised convs, precomputed at load) */
typedef struct {
    int num_layers;
    /* in_layers[i]: Conv1d(192, 384, 5, dil=1, pad=2) */
    float in_w[16][384 * HIDDEN * WAVE_KERNEL];
    float in_b[16][384];
    /* res_skip_layers[i]: Conv1d(192, 384 or 192, 1) */
    int   rs_out[16];
    float rs_w[16][384 * HIDDEN];    /* max 384 outputs */
    float rs_b[16][384];
} WaveNet;

/* Residual coupling layer (one flow stage) */
typedef struct {
    float conv_pre_w[HIDDEN * HALF_FLOW], conv_pre_b[HIDDEN];
    WaveNet wavenet;
    float conv_post_w[HALF_FLOW * HIDDEN], conv_post_b[HALF_FLOW];
} CouplingLayer;

typedef struct {
    int num_flows;
    CouplingLayer flows[NUM_PRIOR_FLOWS];
} CouplingBlock;

/* HiFi-GAN residual block */
typedef struct {
    int ch, kernel;
    int dil[RF_DILS];
    Conv1d c1[RF_DILS];
    Conv1d c2[RF_DILS];
} ResBlock;

typedef struct {
    Conv1d conv_pre;          /* 192 -> 512, k=7 */
    ConvTranspose1d up[4];
    ResBlock rb[12];          /* 4 stages x 3 MRF */
    float conv_post_w[32 * 7];  /* 32 -> 1, k=7, no bias */
} HiFiGan;

/* ============================================================
 * Full model
 * ============================================================ */
typedef struct {
    int sampling_rate;

    /* mmap'd weight file (for zero-copy pointer fields) */
    const unsigned char *vtsm_map;
    size_t vtsm_map_size;

    /* Text encoder */
    float embed_w[VOCAB_SIZE * HIDDEN];
    EncLayer layers[NUM_LAYERS];
    float proj_w[HIDDEN * 2 * HIDDEN];  /* Conv1d(192, 384, 1) */
    float proj_b[2 * HIDDEN];

    /* Duration predictor */
    StochDP dp;

    /* Flow (prior encoder) */
    CouplingBlock flow;

    /* HiFi-GAN decoder */
    HiFiGan decoder;
} VitsModel;

/* ============================================================
 * Vocab / tokenizer
 * ============================================================ */
typedef struct {
    int vocab_size;
    char chars[VOCAB_SIZE][8];   /* UTF-8 */
    int id_map[128];             /* ascii byte -> vocab id, -1 if not found */
} Vocab;

/* ============================================================
 * Function declarations
 * ============================================================ */

/* --- ops.c --- */
void conv1d(const float *in, int in_ch, int T,
            const Conv1d *c, float *out);
void conv1d_depthwise(const float *in, int ch, int T,
                      const float *w, const float *b,
                      int k, int dilation, int pad,
                      float *out);
void conv_transpose1d(const float *in, int in_ch, int T,
                      const ConvTranspose1d *c, float *out, int *out_T);
void layer_norm(const float *in, int T, int dim,
                const LayerNorm *ln, float *out);
void gelu(float *x, int n);
void leaky_relu_f(float *x, int n, float slope);
void tanh_f(float *x, int n);
void sigmoid_f(float *x, int n);
void relu_f(float *x, int n);
void softmax_f(float *x, int n);
void gating_f(const float *lo, const float *hi, float *out, int n);
void mask_zero_f(float *x, const int *mask, int C, int T);
void masked_axpy_f(float *dst, const float *src, const int *mask, int C, int T);
void axpy_f(float *dst, const float *src, int n);
void scal_f(float *x, int n, float alpha);
float randn_f(void);

/* --- vtsm.c --- */
int  load_vtsm(const char *path, VitsModel *model, Vocab *vocab);
void free_model(VitsModel *model);

/* --- tokenizer.c --- */
void vocab_init(Vocab *v);
int  vocab_char_to_id(const Vocab *v, const char *utf8_char);
void tokenize(const Vocab *v, const char *text, int32_t *ids, int *n_ids);

/* --- encoder.c --- */
void encoder_forward(const VitsModel *m,
                     const int32_t *input_ids, int T,
                     const int *mask,
                     float *hidden,    /* [HIDDEN][T] */
                     float *prior_means,       /* [HIDDEN][T] */
                     float *prior_log_vars);   /* [HIDDEN][T] */

/* --- npy_reader.c --- */
float *load_npy_f32(const char *path, int *n_elements, int expected_n);

/* --- duration.c --- */
void dp_reverse(const StochDP *dp,
                const float *cond, int T,
                const int *mask,
                const float *inject_latents,  /* NULL = generate with randn; else [2][T] */
                float *log_duration);  /* [T] */
#ifdef ENABLE_DUMP
void dp_set_dump_dir(const char *dir);
#endif

/* --- flow.c --- */
void flow_reverse(const CouplingBlock *block,
                  float *latents, int T,
                  const int *mask);
#ifdef ENABLE_DUMP
void flow_set_dump_dir(const char *dir);
#endif

/* --- hifigan.c --- */
void hifigan_forward(const HiFiGan *dec,
                     const float *mel, int mel_T,
                     float *wave, int *wave_T);
#ifdef ENABLE_DUMP
void hifigan_set_dump_dir(const char *dir);
#endif

/* --- model.c --- */
int vits_synthesize(const VitsModel *m, const Vocab *vocab,
                    const char *text, int seed,
                    const char *inject_dir,   /* NULL = no injection */
                    float *waveform, int *wave_len);
void set_dump_dir(const char *dir);

/* --- wav.c --- */
int write_wav16(const char *path, const float *samples,
                int n_samples, int sample_rate);

#endif /* VITS_H */
