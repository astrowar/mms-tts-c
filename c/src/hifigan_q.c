#include "vits.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_DUMP
#include "../validate/dump.h"
static const char *hifi_q_dump_dir = NULL;
void hifigan_q_set_dump_dir(const char *dir) { hifi_q_dump_dir = dir; }
#endif

/* ============================================================
 * Quantization: F32 → int16 per-output-channel (symmetric)
 *
 * For each conv weight tensor [out_ch, in_ch, k]:
 *   scale[o] = max(|W[o,:,:]|) / 32767.0
 *   Q[o,:,:] = clamp(round(W[o,:,:] / scale[o]), -32768, 32767)
 *
 * All int16 data is stored in one contiguous buffer (qdata),
 * all scales in another (scales).
 * ============================================================ */

#define QMAX 32767

/* Helper: quantize a single F32 weight tensor, write int16 + scale */
static void quantize_tensor(
    const float *w,
    int out_ch, int in_ch, int k,
    int16_t *q_out,
    float *scale_out)
{
    for (int o = 0; o < out_ch; o++) {
        const float *w_o = w + (size_t)o * in_ch * k;

        /* Find max absolute value for this output channel */
        float max_val = 0.0f;
        for (int i = 0; i < in_ch * k; i++) {
            float a = fabsf(w_o[i]);
            if (a > max_val) max_val = a;
        }

        float scale = max_val / (float)QMAX;
        if (scale < 1e-10f) scale = 1.0f;
        scale_out[o] = scale;

        const float inv_scale = 1.0f / scale;

        int16_t *q_o = q_out + (size_t)o * in_ch * k;
        for (int i = 0; i < in_ch * k; i++) {
            float v = w_o[i] * inv_scale;
            /* Round to nearest, clamp to [-32768, 32767] */
            long q = (long)lroundf(v);
            if (q > QMAX)     q = QMAX;
            if (q < -QMAX - 1) q = -QMAX - 1;
            q_o[i] = (int16_t)q;
        }
    }
}


int hifigan_quantize(const HiFiGan *f32, HiFiGanQ *q)
{
    memset(q, 0, sizeof(*q));

    /*
     * Quantize ALL HiFi-GAN layers to int16 per-output-channel.
     * int16 (32767 levels) has enough precision even for large
     * fan-in layers (upsamplers with 8192 inputs).
     *
     * Layout:
     *   conv_pre:    [512, 192, 7]
     *   up[0..3]:    [out, in, k] each
     *   rb[0..11]:   each has c1[3] + c2[3]
     *   conv_post:   [1, 32, 7]
     */
    size_t total_q = 0;
    int total_scales = 0;

    /* conv_pre */
    total_q += (size_t)512 * 192 * 7;
    total_scales += 512;

    /* upsamplers */
    static const int up_in[]  = {512, 256, 128, 64};
    static const int up_out[] = {256, 128, 64, 32};
    static const int up_k[]   = {16, 16, 4, 4};
    for (int i = 0; i < NUM_UP; i++) {
        total_q += (size_t)up_out[i] * up_in[i] * up_k[i];
        total_scales += up_out[i];
    }

    /* resblocks */
    static const int rb_ch[] = {256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32};
    static const int rb_k[]  = {3, 7, 11, 3, 7, 11, 3, 7, 11, 3, 7, 11};
    for (int r = 0; r < 12; r++) {
        for (int d = 0; d < RF_DILS; d++) {
            total_q += 2 * (size_t)rb_ch[r] * rb_ch[r] * rb_k[r];
            total_scales += 2 * rb_ch[r];
        }
    }

    /* conv_post */
    total_q += (size_t)1 * 32 * 7;
    total_scales += 1;

    q->qdata_size = total_q;
    q->n_scales = total_scales;

    q->qdata = (int16_t *)malloc(total_q * sizeof(int16_t));
    q->scales = (float *)malloc((size_t)total_scales * sizeof(float));
    if (!q->qdata || !q->scales) {
        free(q->qdata);
        free(q->scales);
        q->qdata = NULL;
        q->scales = NULL;
        return -1;
    }

    /*
     * Quantize tensors in order, tracking offsets.
     */
    size_t qoff = 0;
    int soff = 0;

    /* conv_pre: [512, 192, 7] */
    quantize_tensor(f32->conv_pre.weight, 512, 192, 7,
                    q->qdata + qoff, q->scales + soff);
    q->conv_pre.in_ch = 192;
    q->conv_pre.out_ch = 512;
    q->conv_pre.k = 7;
    q->conv_pre.pad = 3;
    q->conv_pre.dilation = 1;
    q->conv_pre.weight = q->qdata + qoff;
    q->conv_pre.scale = q->scales + soff;
    q->conv_pre.bias = f32->conv_pre.bias;
    qoff += (size_t)512 * 192 * 7;
    soff += 512;

    /* upsamplers: quantized to int16 */
    for (int i = 0; i < NUM_UP; i++) {
        int in_ch = up_in[i], out_ch = up_out[i], k = up_k[i];
        quantize_tensor(f32->up[i].weight, out_ch, in_ch, k,
                        q->qdata + qoff, q->scales + soff);
        q->up[i].in_ch  = in_ch;
        q->up[i].out_ch = out_ch;
        q->up[i].k      = k;
        q->up[i].stride = f32->up[i].stride;
        q->up[i].pad    = f32->up[i].pad;
        q->up[i].weight = q->qdata + qoff;
        q->up[i].scale  = q->scales + soff;
        q->up[i].bias   = f32->up[i].bias;
        qoff += (size_t)out_ch * in_ch * k;
        soff += out_ch;
    }

    /* resblocks */
    for (int r = 0; r < 12; r++) {
        ResBlockQ *rbq = &q->rb[r];
        const ResBlock *rb = &f32->rb[r];
        int ch = rb_ch[r], k = rb_k[r];

        rbq->ch = ch;
        rbq->kernel = k;
        for (int d = 0; d < RF_DILS; d++)
            rbq->dil[d] = rb->dil[d];

        for (int d = 0; d < RF_DILS; d++) {
            /* c1[d]: Conv1d(ch, ch, k, dil from model: 1,3,5) */
            quantize_tensor(rb->c1[d].weight, ch, ch, k,
                            q->qdata + qoff, q->scales + soff);
            rbq->c1[d].in_ch = ch;
            rbq->c1[d].out_ch = ch;
            rbq->c1[d].k = k;
            rbq->c1[d].pad = rb->c1[d].pad;
            rbq->c1[d].dilation = rb->c1[d].dilation;
            rbq->c1[d].weight = q->qdata + qoff;
            rbq->c1[d].scale = q->scales + soff;
            rbq->c1[d].bias = rb->c1[d].bias;
            qoff += (size_t)ch * ch * k;
            soff += ch;

            /* c2[d]: Conv1d(ch, ch, k, dil=1) */
            quantize_tensor(rb->c2[d].weight, ch, ch, k,
                            q->qdata + qoff, q->scales + soff);
            rbq->c2[d].in_ch = ch;
            rbq->c2[d].out_ch = ch;
            rbq->c2[d].k = k;
            rbq->c2[d].pad = rb->c2[d].pad;
            rbq->c2[d].dilation = 1;
            rbq->c2[d].weight = q->qdata + qoff;
            rbq->c2[d].scale = q->scales + soff;
            rbq->c2[d].bias = rb->c2[d].bias;
            qoff += (size_t)ch * ch * k;
            soff += ch;
        }
    }

    /* conv_post: [1, 32, 7], no bias */
    {
        /* Build a temporary Conv1d view of the F32 conv_post */
        Conv1d cp = {
            .in_ch = 32, .out_ch = 1, .k = 7,
            .pad = 3, .dilation = 1,
            .weight = f32->conv_post_w, .bias = NULL
        };
        (void)cp;
        quantize_tensor(f32->conv_post_w, 1, 32, 7,
                        q->qdata + qoff, q->scales + soff);
        q->conv_post.in_ch = 32;
        q->conv_post.out_ch = 1;
        q->conv_post.k = 7;
        q->conv_post.pad = 3;
        q->conv_post.dilation = 1;
        q->conv_post.weight = q->qdata + qoff;
        q->conv_post.scale = q->scales + soff;
        q->conv_post.bias = NULL;  /* no bias for conv_post */
        qoff += (size_t)1 * 32 * 7;
        soff += 1;
    }

    /* Sanity check */
    if (qoff != total_q || soff != total_scales) {
        fprintf(stderr, "[hifigan_quantize] offset mismatch: qoff=%zu/%zu soff=%d/%d\n",
                qoff, total_q, soff, total_scales);
        return -1;
    }

    return 0;
}


void hifigan_free_q(HiFiGanQ *q)
{
    if (!q) return;
    free(q->qdata);
    free(q->scales);
    q->qdata = NULL;
    q->scales = NULL;
    q->qdata_size = 0;
    q->n_scales = 0;
}


/* ============================================================
 * Quantized HiFi-GAN forward pass (hybrid)
 *
 * Strategy: quantize conv_pre + ResBlocks + conv_post to int8.
 * Keep upsamplers in F32 — their large fan-in (512×16=8192)
 * amplifies per-weight quantization error too much.
 *
 * The `f32` parameter provides F32 upsampler weights from the mmap.
 * Dumps prefixed "06q_" to distinguish from F32 dumps.
 * ============================================================ */
void hifigan_forward_q(const HiFiGanQ *dec, const HiFiGan *f32,
                       const float *mel, int mel_T,
                       float *wave, int *wave_T)
{
    int wave_len = mel_T * TOTAL_UP;

    size_t max_size = (size_t)32 * wave_len + 8192;

    float *A = (float *)malloc(max_size * sizeof(float));
    float *B = (float *)malloc(max_size * sizeof(float));
    float *C = (float *)malloc(max_size * sizeof(float));
    float *D = (float *)malloc(max_size * sizeof(float));
    float *E = (float *)malloc(max_size * sizeof(float));

#ifdef ENABLE_DUMP
    static int hifi_q_call_count = 0;
    hifi_q_call_count++;
    int dump_this_call = (hifi_q_call_count == 1);
#endif

    /* ---- 1. conv_pre (int8) ---- */
    conv1d_q(mel, 192, mel_T, &dec->conv_pre, A);

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q_dump_dir) {
        int dim2[2] = {HIFI_INIT_CH, mel_T};
        dump_f32(hifi_q_dump_dir, "06q_hifi_conv_pre", A, 2, dim2);
    }
#endif

    int cur_ch = HIFI_INIT_CH;
    int cur_T = mel_T;

    /* ---- 2. Four upsampling + MRF stages ---- */
    for (int i = 0; i < NUM_UP; i++) {
        size_t cur_size = (size_t)cur_ch * cur_T;

        leaky_relu_f(A, cur_size, LEAKY_SLOPE);

        /* Upsampler: int16 quantized */
        int new_T;
        conv_transpose1d_q(A, cur_ch, cur_T, &dec->up[i], B, &new_T);
        int new_ch = dec->up[i].out_ch;
        size_t new_size = (size_t)new_ch * new_T;

        /* MRF: 3 ResBlocks in parallel on B (int8) */
        memset(A, 0, new_size * sizeof(float));

        for (int j = 0; j < 3; j++) {
            const ResBlockQ *rb = &dec->rb[i * 3 + j];

            memcpy(C, B, new_size * sizeof(float));

            for (int d = 0; d < RF_DILS; d++) {
                memcpy(D, C, new_size * sizeof(float));

                leaky_relu_f(C, new_size, LEAKY_SLOPE);

                conv1d_q(C, new_ch, new_T, &rb->c1[d], E);

                leaky_relu_f(E, new_size, LEAKY_SLOPE);

                conv1d_q(E, new_ch, new_T, &rb->c2[d], C);

                axpy_f(C, D, (int)new_size);
            }

            axpy_f(A, C, (int)new_size);
        }

        scal_f(A, (int)new_size, 1.0f / 3.0f);

#ifdef ENABLE_DUMP
        if (dump_this_call && hifi_q_dump_dir) {
            char name[64];
            snprintf(name, sizeof(name), "06q_hifi_stage_%d", i);
            int dims[2] = {new_ch, new_T};
            dump_f32(hifi_q_dump_dir, name, A, 2, dims);
        }
#endif

        cur_ch = new_ch;
        cur_T = new_T;
    }

    /* ---- 3. Output: LeakyReLU(0.01) -> conv_post (int8) -> tanh ---- */
    size_t final_size = (size_t)cur_ch * cur_T;
    leaky_relu_f(A, final_size, 0.01f);

    conv1d_q(A, cur_ch, cur_T, &dec->conv_post, wave);

    *wave_T = wave_len;

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q_dump_dir) {
        int dim1[1] = {wave_len};
        dump_f32(hifi_q_dump_dir, "06q_hifi_pre_tanh", wave, 1, dim1);
    }
#endif

    tanh_f(wave, wave_len);

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
}
