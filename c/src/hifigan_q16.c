#include "vits.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_DUMP
#include "../validate/dump.h"
static const char *hifi_q16_dump_dir = NULL;
void hifigan_q16_set_dump_dir(const char *dir) { hifi_q16_dump_dir = dir; }
#endif

/* ============================================================
 * Helper: fill a Conv1dQ16 from a Conv1dQ
 *
 * Computes:
 *   alpha_q[o] = round( 2^(in_exp - out_exp + ALPHA_BITS) * scale[o] )
 *   bias_q[o]  = round( bias_f32[o] * 2^(-out_exp) )
 *
 * p is a cursor into the params buffer; advances by 2*out_ch.
 * ============================================================ */
static void fill_conv1d_q16(Conv1dQ16 *dst, const Conv1dQ *src,
                            int in_exp, int out_exp,
                            int has_bias, int32_t **p)
{
    dst->in_ch    = src->in_ch;
    dst->out_ch   = src->out_ch;
    dst->k        = src->k;
    dst->pad      = src->pad;
    dst->dilation = src->dilation;
    dst->weight   = src->weight;
    dst->in_exp   = in_exp;
    dst->out_exp  = out_exp;

    const int n = src->out_ch;

    dst->bias_q  = *p;  *p += n;
    dst->alpha_q = *p;  *p += n;

    const double alpha_scale =
        ldexp(1.0, (double)(in_exp - out_exp + Q16_ALPHA_BITS));
    const double bias_scale  =
        ldexp(1.0, (double)(-out_exp));

    for (int o = 0; o < n; o++) {
        if (has_bias && src->bias)
            dst->bias_q[o]  = (int32_t)llround(bias_scale * (double)src->bias[o]);
        else
            dst->bias_q[o]  = 0;
        dst->alpha_q[o] = (int32_t)llround(alpha_scale * (double)src->scale[o]);
    }
}

/* Same for ConvTranspose1dQ16 */
static void fill_conv_transpose_q16(ConvTranspose1dQ16 *dst,
                                    const ConvTranspose1dQ *src,
                                    int in_exp, int out_exp,
                                    int has_bias, int32_t **p)
{
    dst->in_ch  = src->in_ch;
    dst->out_ch = src->out_ch;
    dst->k      = src->k;
    dst->stride = src->stride;
    dst->pad    = src->pad;
    dst->weight = src->weight;
    dst->in_exp  = in_exp;
    dst->out_exp = out_exp;

    const int n = src->out_ch;

    dst->bias_q  = *p;  *p += n;
    dst->alpha_q = *p;  *p += n;

    const double alpha_scale =
        ldexp(1.0, (double)(in_exp - out_exp + Q16_ALPHA_BITS));
    const double bias_scale  =
        ldexp(1.0, (double)(-out_exp));

    for (int o = 0; o < n; o++) {
        if (has_bias && src->bias)
            dst->bias_q[o]  = (int32_t)llround(bias_scale * (double)src->bias[o]);
        else
            dst->bias_q[o]  = 0;
        dst->alpha_q[o] = (int32_t)llround(alpha_scale * (double)src->scale[o]);
    }
}

/* ============================================================
 * hifigan_q16_init
 *
 * Build the Q16 decoder from the existing INT8 decoder.
 * Computes all alpha_q / bias_q arrays in a single allocation.
 * Returns 0 on success, -1 on OOM.
 * ============================================================ */
int hifigan_q16_init(HiFiGanQ16 *q16, const HiFiGanQ *q)
{
    memset(q16, 0, sizeof(*q16));

    /* Count total int32 params: 2 * out_ch per conv layer */
    size_t total = 0;
    total += 2 * (size_t)q->conv_pre.out_ch;
    for (int i = 0; i < NUM_UP; i++)
        total += 2 * (size_t)q->up[i].out_ch;
    for (int i = 0; i < NUM_UP * RF_DILS; i++) {
        total += 2 * (size_t)q->rb[i].c1[0].out_ch;
        total += 2 * (size_t)q->rb[i].c2[0].out_ch;
        /* c1[1], c1[2], c2[1], c2[2] have same out_ch as c1[0] */
        total += 4 * 2 * (size_t)q->rb[i].c1[0].out_ch;
    }
    total += 2 * (size_t)q->conv_post.out_ch;

    q16->params = (int32_t *)calloc(total, sizeof(int32_t));
    if (!q16->params)
        return -1;
    q16->params_size = total;

    int32_t *p = q16->params;

    /* conv_pre: 192 → 512, k=7, pad=3, dil=1, has bias */
    fill_conv1d_q16(&q16->conv_pre, &q->conv_pre,
                    Q16_MEL_EXP, Q16_CONV_PRE_EXP, 1, &p);

    /* 4 upsample stages */
    static const int stage_exp[NUM_UP] = {
        Q16_STAGE0_EXP, Q16_STAGE1_EXP,
        Q16_STAGE2_EXP, Q16_STAGE3_EXP
    };
    int prev_exp = Q16_CONV_PRE_EXP;

    for (int i = 0; i < NUM_UP; i++) {
        fill_conv_transpose_q16(&q16->up[i], &q->up[i],
                                prev_exp, stage_exp[i], 1, &p);

        /* 3 ResBlocks per stage */
        for (int j = 0; j < RF_DILS; j++) {
            const ResBlockQ *rb_q = &q->rb[i * RF_DILS + j];
            ResBlockQ16 *rb = &q16->rb[i * RF_DILS + j];

            rb->ch     = rb_q->ch;
            rb->kernel = rb_q->kernel;
            for (int d = 0; d < RF_DILS; d++)
                rb->dil[d] = rb_q->dil[d];

            /* All convs within a stage use the same exponent */
            for (int d = 0; d < RF_DILS; d++) {
                fill_conv1d_q16(&rb->c1[d], &rb_q->c1[d],
                                stage_exp[i], stage_exp[i], 1, &p);
                fill_conv1d_q16(&rb->c2[d], &rb_q->c2[d],
                                stage_exp[i], stage_exp[i], 1, &p);
            }
        }

        prev_exp = stage_exp[i];
    }

    /* conv_post: 32 → 1, k=7, no bias */
    fill_conv1d_q16(&q16->conv_post, &q->conv_post,
                    prev_exp, Q16_CONV_POST_EXP, 0, &p);

    return 0;
}

void hifigan_q16_free(HiFiGanQ16 *q16)
{
    free(q16->params);
    q16->params = NULL;
    q16->params_size = 0;
}

/* ============================================================
 * hifigan_forward_q16
 *
 * Full fixed-point HiFi-GAN forward pass.
 *
 * Input:  mel [192, mel_T] float32 (from flow output)
 * Output: pcm [mel_T * 256] int16 (after tanh LUT)
 *
 * Pipeline:
 *   mel F32 → quantize → INT16
 *   → conv_pre → LeakyReLU → up[0] → MRF0
 *   → LeakyReLU → up[1] → MRF1
 *   → LeakyReLU → up[2] → MRF2
 *   → LeakyReLU → up[3] → MRF3
 *   → LeakyReLU(0.01) → conv_post → tanh LUT → PCM INT16
 *
 * No float32 in the hot path after the initial quantization.
 * ============================================================ */
void hifigan_forward_q16(const HiFiGanQ16 *q16,
                         const float *mel, int mel_T,
                         int16_t *pcm_out, int *pcm_len)
{
    const int wave_len = mel_T * TOTAL_UP;

    /* Max buffer: largest stage output (stages 1-3 all same size) */
    int last_ch = q16->up[NUM_UP - 1].out_ch;
    size_t max_elems = (size_t)last_ch * wave_len;
    if (max_elems < 8192) max_elems = 8192;

    /* INT16 buffers */
    int16_t *A = (int16_t *)malloc(max_elems * sizeof(int16_t));
    int16_t *B = (int16_t *)malloc(max_elems * sizeof(int16_t));
    int16_t *C = (int16_t *)malloc(max_elems * sizeof(int16_t));
    int16_t *D = (int16_t *)malloc(max_elems * sizeof(int16_t));
    int16_t *E = (int16_t *)malloc(max_elems * sizeof(int16_t));
    if (!A || !B || !C || !D || !E) {
        free(A); free(B); free(C); free(D); free(E);
        *pcm_len = 0;
        return;
    }

#ifdef ENABLE_DUMP
    static int q16_call_count = 0;
    q16_call_count++;
    int dump_this_call = (q16_call_count == 1);
#endif

    /* ---- 1. Quantize mel: FP32 → INT16 ---- */
    size_t mel_size = (size_t)q16->conv_pre.in_ch * mel_T;
    quantize_f32_to_q16(mel, (int)mel_size, Q16_MEL_EXP, A);

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q16_dump_dir) {
        /* Dump as float for compatibility with existing compare tools */
        float *tmp = (float *)malloc(mel_size * sizeof(float));
        if (tmp) {
            for (size_t i = 0; i < mel_size; i++)
                tmp[i] = (float)A[i] * (float)ldexp(1.0, (double)Q16_MEL_EXP);
            int dim2[2] = {q16->conv_pre.in_ch, mel_T};
            dump_f32(hifi_q16_dump_dir, "06q16_hifi_mel_q16", tmp, 2, dim2);
            free(tmp);
        }
    }
#endif

    /* ---- 2. conv_pre (write to B to avoid aliasing: in_ch < out_ch) ---- */
    conv1d_q16(A, q16->conv_pre.in_ch, mel_T, &q16->conv_pre, B);
    { int16_t *tmp = A; A = B; B = tmp; }

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q16_dump_dir) {
        int dim2[2] = {q16->conv_pre.out_ch, mel_T};
        float *tmp = (float *)malloc((size_t)q16->conv_pre.out_ch * mel_T * sizeof(float));
        if (tmp) {
            double s = ldexp(1.0, (double)Q16_CONV_PRE_EXP);
            size_t n = (size_t)q16->conv_pre.out_ch * mel_T;
            for (size_t i = 0; i < n; i++) tmp[i] = (float)((double)A[i] * s);
            dump_f32(hifi_q16_dump_dir, "06q16_hifi_conv_pre", tmp, 2, dim2);
            free(tmp);
        }
    }
#endif

    int cur_ch = q16->conv_pre.out_ch;
    int cur_T  = mel_T;

    /* LeakyReLU slope: 13/128 ≈ 0.1016 */
    const int LR_NUM = 13, LR_DEN = 128;

    /* ---- 3. Four upsample + MRF stages ---- */
    for (int i = 0; i < NUM_UP; i++) {
        size_t cur_size = (size_t)cur_ch * cur_T;

        leaky_relu_q16(A, (int)cur_size, LR_NUM, LR_DEN);

        /* Upsampler */
        int new_T;
        conv_transpose1d_q16(A, cur_ch, cur_T, &q16->up[i], B, &new_T);
        int new_ch = q16->up[i].out_ch;
        size_t new_size = (size_t)new_ch * new_T;

        /* MRF: 3 ResBlocks in parallel */
        memset(A, 0, new_size * sizeof(int16_t));

        for (int j = 0; j < RF_DILS; j++) {
            const ResBlockQ16 *rb = &q16->rb[i * RF_DILS + j];

            memcpy(C, B, new_size * sizeof(int16_t));

            for (int d = 0; d < RF_DILS; d++) {
                memcpy(D, C, new_size * sizeof(int16_t));

                leaky_relu_q16(C, (int)new_size, LR_NUM, LR_DEN);
                conv1d_q16(C, new_ch, new_T, &rb->c1[d], E);
                leaky_relu_q16(E, (int)new_size, LR_NUM, LR_DEN);
                conv1d_q16(E, new_ch, new_T, &rb->c2[d], C);
                residual_add_q16(C, D, (int)new_size);
            }

            residual_add_q16(A, C, (int)new_size);
        }

        mrf_div3_q16(A, (int)new_size);

#ifdef ENABLE_DUMP
        if (dump_this_call && hifi_q16_dump_dir) {
            char name[64];
            snprintf(name, sizeof(name), "06q16_hifi_stage_%d", i);
            int dims[2] = {new_ch, new_T};
            float *tmp = (float *)malloc(new_size * sizeof(float));
            if (tmp) {
                double s = ldexp(1.0, (double)stage_exp[i]);
                for (size_t t = 0; t < new_size; t++)
                    tmp[t] = (float)((double)A[t] * s);
                dump_f32(hifi_q16_dump_dir, name, tmp, 2, dims);
                free(tmp);
            }
        }
#endif

        cur_ch  = new_ch;
        cur_T   = new_T;
    }

    /* ---- 4. Output: LeakyReLU(0.01) → conv_post → tanh LUT ---- */
    size_t final_size = (size_t)cur_ch * cur_T;

    /* slope 0.01 ≈ 1/128 */
    leaky_relu_q16(A, (int)final_size, 1, 128);

    conv1d_q16(A, cur_ch, cur_T, &q16->conv_post, B);

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q16_dump_dir) {
        int d1 = wave_len;
        float *tmp = (float *)malloc(wave_len * sizeof(float));
        if (tmp) {
            double s = ldexp(1.0, (double)Q16_CONV_POST_EXP);
            for (int t = 0; t < wave_len; t++)
                tmp[t] = (float)((double)B[t] * s);
            dump_f32(hifi_q16_dump_dir, "06q16_hifi_pre_tanh", tmp, 1, &d1);
            free(tmp);
        }
    }
#endif

    /* ---- 5. Tanh LUT → PCM INT16 ---- */
    static int16_t tanh_lut[32768];
    static int lut_exp = -999;
    if (lut_exp != Q16_CONV_POST_EXP) {
        tanh_lut_init(tanh_lut, Q16_CONV_POST_EXP);
        lut_exp = Q16_CONV_POST_EXP;
    }

    int n_clamped = tanh_q16(B, wave_len, Q16_CONV_POST_EXP, tanh_lut, pcm_out);

    (void)n_clamped;  /* could log for diagnostics */

    *pcm_len = wave_len;

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
}
