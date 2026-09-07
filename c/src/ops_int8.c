#include "vits.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef VITS_OMP_WORK_THRESHOLD
#define VITS_OMP_WORK_THRESHOLD 32768
#endif

#if defined(_MSC_VER)
#define VITS_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define VITS_RESTRICT __restrict__
#else
#define VITS_RESTRICT restrict
#endif

static inline int imin_i(int a, int b) { return a < b ? a : b; }
static inline int imax_i(int a, int b) { return a > b ? a : b; }


/* ============================================================
 * Quantized Conv1D (int16 weights, float32 activations)
 *
 * Weights:  int16 [out_ch][in_ch][k]  — quantized with per-channel scale
 * Scales:   float32 [out_ch]
 * Bias:     float32 [out_ch]
 *
 * Dequantization:
 *   real_w[o,i,j] = (float)q_w[o,i,j] * scale[o]
 *   out[o,t] = sum_{i,j} real_w[o,i,j] * in[i, t+j*dil-pad] + bias[o]
 *
 * Computed as:
 *   acc[o,t] = sum_{i,j} (float)q_w[o,i,j] * in[i, t+j*dil-pad]
 *   out[o,t] = acc[o,t] * scale[o] + bias[o]
 * ============================================================ */
void conv1d_q(
    const float *VITS_RESTRICT in,
    int in_ch,
    int T,
    const Conv1dQ *VITS_RESTRICT c,
    float *VITS_RESTRICT out)
{
    const int out_ch = c->out_ch;
    const int k      = c->k;
    const int pad    = c->pad;
    const int dil    = c->dilation;

    if (T <= 0 || in_ch <= 0 || out_ch <= 0)
        return;

    const size_t work =
        (size_t)out_ch * (size_t)in_ch * (size_t)k * (size_t)T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        float *VITS_RESTRICT out_o = out + (size_t)o * T;
        const float scale_o = c->scale[o];
        const float bias_o  = c->bias ? c->bias[o] : 0.0f;

        const int16_t *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        /* Zero accumulator */
        memset(out_o, 0, sizeof(float) * (size_t)T);

        /* Accumulate raw int16 * float32 products */
        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i = in + (size_t)i * T;
            const int16_t *VITS_RESTRICT w_i = w_o + (size_t)i * k;

            for (int j = 0; j < k; j++) {

                const float weight = (float)w_i[j];
                if (weight == 0.0f)
                    continue;

                const int shift = j * dil - pad;

                int t0 = 0, t1 = T;
                if (shift < 0) t0 = -shift;
                if (shift > 0) t1 = T - shift;
                t0 = imax_i(t0, 0);
                t1 = imin_i(t1, T);
                if (t0 >= t1) continue;

#ifdef _OPENMP
#pragma omp simd
#endif
                for (int t = t0; t < t1; t++) {
                    out_o[t] += weight * in_i[t + shift];
                }
            }
        }

        /* Dequantize: multiply by scale, add bias */
        if (scale_o == 1.0f && bias_o == 0.0f) {
            /* No-op — skip for speed (only if scale is exactly 1) */
        } else if (bias_o == 0.0f) {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < T; t++)
                out_o[t] *= scale_o;
        } else {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < T; t++)
                out_o[t] = out_o[t] * scale_o + bias_o;
        }
    }
}


/* ============================================================
 * Quantized ConvTranspose1D (int16 weights, float32 activations)
 *
 * Same dequantization scheme as conv1d_q.
 * Weight layout: [out_ch][in_ch][k]
 * ============================================================ */
void conv_transpose1d_q(
    const float *VITS_RESTRICT in,
    int in_ch,
    int T,
    const ConvTranspose1dQ *VITS_RESTRICT c,
    float *VITS_RESTRICT out,
    int *out_T)
{
    const int out_ch = c->out_ch;
    const int k      = c->k;
    const int stride = c->stride;
    const int pad    = c->pad;

    if (T <= 0 || in_ch <= 0 || out_ch <= 0) {
        *out_T = 0;
        return;
    }

    const int oT = (T - 1) * stride - 2 * pad + k;
    *out_T = oT;

    if (oT <= 0)
        return;

    const size_t work =
        (size_t)out_ch * (size_t)in_ch * (size_t)T * (size_t)k;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        float *VITS_RESTRICT out_o = out + (size_t)o * oT;
        const float scale_o = c->scale[o];
        const float bias_o  = c->bias ? c->bias[o] : 0.0f;

        const int16_t *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        /* Zero accumulator */
        memset(out_o, 0, sizeof(float) * (size_t)oT);

        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i = in + (size_t)i * T;
            const int16_t *VITS_RESTRICT w_i = w_o + (size_t)i * k;

            for (int j = 0; j < k; j++) {

                const float weight = (float)w_i[j];
                if (weight == 0.0f)
                    continue;

                /* pos = t*stride - pad + j, need 0 <= pos < oT */
                int t0 = 0, t1 = T;

                const int lower = pad - j;
                if (lower > 0)
                    t0 = (lower + stride - 1) / stride;

                const int upper = oT - 1 + pad - j;
                if (upper < 0)
                    continue;

                const int max_t = upper / stride;
                t1 = imin_i(t1, max_t + 1);
                t0 = imax_i(t0, 0);

                if (t0 >= t1)
                    continue;

                for (int t = t0; t < t1; t++) {
                    const int pos = t * stride - pad + j;
                    out_o[pos] += weight * in_i[t];
                }
            }
        }

        /* Dequantize: multiply by scale, add bias */
        if (scale_o == 1.0f && bias_o == 0.0f) {
            /* No-op */
        } else if (bias_o == 0.0f) {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < oT; t++)
                out_o[t] *= scale_o;
        } else {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < oT; t++)
                out_o[t] = out_o[t] * scale_o + bias_o;
        }
    }
}
