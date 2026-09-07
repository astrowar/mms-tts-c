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
 * Quantization: F32 → int8 per-output-channel (symmetric)
 *
 * For each conv weight tensor [out_ch, in_ch, k]:
 *   scale[o] = max(|W[o,:,:]|) / 127.0
 *   Q[o,:,:] = clamp(round(W[o,:,:] / scale[o]), -128, 127)
 *
 * All int8 data is stored in one contiguous buffer (qdata),
 * all scales in another (scales).
 *
 * ALL dimensions are read from the F32 model struct — nothing
 * is hardcoded. This ensures the quantizer stays correct if
 * the architecture changes (different channels, kernels, etc).
 * ============================================================ */

#define QMAX 127

/* Helper: quantize a single F32 weight tensor, write int8 + scale */
static void quantize_tensor(
    const float *w,
    int out_ch, int in_ch, int k,
    int8_t *q_out,
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

        int8_t *q_o = q_out + (size_t)o * in_ch * k;
        for (int i = 0; i < in_ch * k; i++) {
            float v = w_o[i] * inv_scale;
            /* Round to nearest, clamp to [-128, 127] */
            long q = (long)lroundf(v);
            if (q > QMAX)     q = QMAX;
            if (q < -QMAX - 1) q = -QMAX - 1;
            q_o[i] = (int8_t)q;
        }
    }
}


int hifigan_quantize(const HiFiGan *f32, HiFiGanQ *q)
{
    memset(q, 0, sizeof(*q));

    /*
     * Quantize ALL HiFi-GAN layers to int8 per-output-channel.
     * All dimensions read from the F32 model — no hardcoded values.
     */
    size_t total_q = 0;
    int total_scales = 0;

    /* conv_pre */
    {
        int oc = f32->conv_pre.out_ch;
        int ic = f32->conv_pre.in_ch;
        int k  = f32->conv_pre.k;
        total_q += (size_t)oc * ic * k;
        total_scales += oc;
    }

    /* upsamplers */
    for (int i = 0; i < NUM_UP; i++) {
        int oc = f32->up[i].out_ch;
        int ic = f32->up[i].in_ch;
        int k  = f32->up[i].k;
        total_q += (size_t)oc * ic * k;
        total_scales += oc;
    }

    /* resblocks */
    for (int r = 0; r < NUM_UP * RF_DILS; r++) {
        for (int d = 0; d < RF_DILS; d++) {
            int oc1 = f32->rb[r].c1[d].out_ch;
            int ic1 = f32->rb[r].c1[d].in_ch;
            int k1  = f32->rb[r].c1[d].k;
            int oc2 = f32->rb[r].c2[d].out_ch;
            int ic2 = f32->rb[r].c2[d].in_ch;
            int k2  = f32->rb[r].c2[d].k;
            total_q += (size_t)oc1 * ic1 * k1 + (size_t)oc2 * ic2 * k2;
            total_scales += oc1 + oc2;
        }
    }

    /* conv_post */
    {
        int oc = f32->conv_post.out_ch;
        int ic = f32->conv_post.in_ch;
        int k  = f32->conv_post.k;
        total_q += (size_t)oc * ic * k;
        total_scales += oc;
    }

    q->qdata_size = total_q;
    q->n_scales = total_scales;

    q->qdata = (int8_t *)malloc(total_q * sizeof(int8_t));
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
     * All metadata (ch, k, pad, dilation) copied from F32 struct.
     */
    size_t qoff = 0;
    int soff = 0;

    /* conv_pre */
    {
        int oc = f32->conv_pre.out_ch;
        int ic = f32->conv_pre.in_ch;
        int k  = f32->conv_pre.k;
        quantize_tensor(f32->conv_pre.weight, oc, ic, k,
                        q->qdata + qoff, q->scales + soff);
        q->conv_pre.in_ch = ic;
        q->conv_pre.out_ch = oc;
        q->conv_pre.k = k;
        q->conv_pre.pad = f32->conv_pre.pad;
        q->conv_pre.dilation = f32->conv_pre.dilation;
        q->conv_pre.weight = q->qdata + qoff;
        q->conv_pre.scale = q->scales + soff;
        q->conv_pre.bias = f32->conv_pre.bias;
        qoff += (size_t)oc * ic * k;
        soff += oc;
    }

    /* upsamplers */
    for (int i = 0; i < NUM_UP; i++) {
        int oc = f32->up[i].out_ch;
        int ic = f32->up[i].in_ch;
        int k  = f32->up[i].k;
        quantize_tensor(f32->up[i].weight, oc, ic, k,
                        q->qdata + qoff, q->scales + soff);
        q->up[i].in_ch  = ic;
        q->up[i].out_ch = oc;
        q->up[i].k      = k;
        q->up[i].stride = f32->up[i].stride;
        q->up[i].pad    = f32->up[i].pad;
        q->up[i].weight = q->qdata + qoff;
        q->up[i].scale  = q->scales + soff;
        q->up[i].bias   = f32->up[i].bias;
        qoff += (size_t)oc * ic * k;
        soff += oc;
    }

    /* resblocks */
    for (int r = 0; r < NUM_UP * RF_DILS; r++) {
        ResBlockQ *rbq = &q->rb[r];
        const ResBlock *rb = &f32->rb[r];

        rbq->ch = rb->ch;
        rbq->kernel = rb->kernel;
        for (int d = 0; d < RF_DILS; d++)
            rbq->dil[d] = rb->dil[d];

        for (int d = 0; d < RF_DILS; d++) {
            /* c1[d] */
            {
                int oc = rb->c1[d].out_ch;
                int ic = rb->c1[d].in_ch;
                int k  = rb->c1[d].k;
                quantize_tensor(rb->c1[d].weight, oc, ic, k,
                                q->qdata + qoff, q->scales + soff);
                rbq->c1[d].in_ch = ic;
                rbq->c1[d].out_ch = oc;
                rbq->c1[d].k = k;
                rbq->c1[d].pad = rb->c1[d].pad;
                rbq->c1[d].dilation = rb->c1[d].dilation;
                rbq->c1[d].weight = q->qdata + qoff;
                rbq->c1[d].scale = q->scales + soff;
                rbq->c1[d].bias = rb->c1[d].bias;
                qoff += (size_t)oc * ic * k;
                soff += oc;
            }

            /* c2[d] */
            {
                int oc = rb->c2[d].out_ch;
                int ic = rb->c2[d].in_ch;
                int k  = rb->c2[d].k;
                quantize_tensor(rb->c2[d].weight, oc, ic, k,
                                q->qdata + qoff, q->scales + soff);
                rbq->c2[d].in_ch = ic;
                rbq->c2[d].out_ch = oc;
                rbq->c2[d].k = k;
                rbq->c2[d].pad = rb->c2[d].pad;
                rbq->c2[d].dilation = rb->c2[d].dilation;
                rbq->c2[d].weight = q->qdata + qoff;
                rbq->c2[d].scale = q->scales + soff;
                rbq->c2[d].bias = rb->c2[d].bias;
                qoff += (size_t)oc * ic * k;
                soff += oc;
            }
        }
    }

    /* conv_post */
    {
        int oc = f32->conv_post.out_ch;
        int ic = f32->conv_post.in_ch;
        int k  = f32->conv_post.k;
        quantize_tensor(f32->conv_post.weight, oc, ic, k,
                        q->qdata + qoff, q->scales + soff);
        q->conv_post.in_ch = ic;
        q->conv_post.out_ch = oc;
        q->conv_post.k = k;
        q->conv_post.pad = f32->conv_post.pad;
        q->conv_post.dilation = f32->conv_post.dilation;
        q->conv_post.weight = q->qdata + qoff;
        q->conv_post.scale = q->scales + soff;
        q->conv_post.bias = f32->conv_post.bias;
        qoff += (size_t)oc * ic * k;
        soff += oc;
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
 * Quantized HiFi-GAN forward pass
 *
 * All conv layers use int8 weights with per-channel dequantization.
 * Activations remain float32 throughout.
 * Dumps prefixed "06q_" to distinguish from F32 dumps.
 * ============================================================ */
void hifigan_forward_q(const HiFiGanQ *dec, const HiFiGan *f32,
                       const float *mel, int mel_T,
                       float *wave, int *wave_T)
{
    (void)f32;  /* no longer needed — all convs are quantized */

    int wave_len = mel_T * TOTAL_UP;

    /* Buffer size: largest intermediate is at the last upstage output */
    int last_ch = dec->up[NUM_UP - 1].out_ch;
    size_t max_size = (size_t)last_ch * wave_len + 8192;

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

    /* ---- 1. conv_pre ---- */
    conv1d_q(mel, dec->conv_pre.in_ch, mel_T, &dec->conv_pre, A);

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q_dump_dir) {
        int dim2[2] = {dec->conv_pre.out_ch, mel_T};
        dump_f32(hifi_q_dump_dir, "06q_hifi_conv_pre", A, 2, dim2);
    }
#endif

    int cur_ch = dec->conv_pre.out_ch;
    int cur_T = mel_T;

    /* ---- 2. Four upsampling + MRF stages ---- */
    for (int i = 0; i < NUM_UP; i++) {
        size_t cur_size = (size_t)cur_ch * cur_T;

        leaky_relu_f(A, cur_size, LEAKY_SLOPE);

        /* Upsampler */
        int new_T;
        conv_transpose1d_q(A, cur_ch, cur_T, &dec->up[i], B, &new_T);
        int new_ch = dec->up[i].out_ch;
        size_t new_size = (size_t)new_ch * new_T;

        /* MRF: 3 ResBlocks in parallel on B */
        memset(A, 0, new_size * sizeof(float));

        for (int j = 0; j < RF_DILS; j++) {
            const ResBlockQ *rb = &dec->rb[i * RF_DILS + j];

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

        scal_f(A, (int)new_size, 1.0f / (float)RF_DILS);

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

    /* ---- 3. Output: LeakyReLU(0.01) -> conv_post -> tanh ---- */
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
