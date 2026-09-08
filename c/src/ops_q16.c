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

static inline int16_t saturate_i16(int32_t x)
{
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

/* ============================================================
 * Quantize FP32 → Q16 (INT16 with power-of-2 scale)
 *
 *   real = int16 * 2^exponent
 *   int16 = round(real * 2^(-exponent)), saturated
 *
 * This is the single FP32→integer conversion at the HiFi-GAN
 * input. All subsequent ops are pure integer.
 * ============================================================ */
void quantize_f32_to_q16(const float *VITS_RESTRICT in, int n,
                         int exponent, int16_t *VITS_RESTRICT out)
{
    const float scale = (float)ldexp(1.0, (double)(-exponent));
    for (int i = 0; i < n; i++) {
        float v = in[i] * scale;
        int32_t q = (int32_t)lroundf(v);
        out[i] = saturate_i16(q);
    }
}

/* ============================================================
 * Conv1D: INT16 input × INT8 weights → INT32 acc → INT16 output
 *
 *   acc[o,t] = Σ_{i,j} in[i, t+j*dil-pad] × w[o,i,j]
 *   out[o,t] = saturate( (acc × alpha_q[o] + 2^(A-1)) >> A
 *                        + bias_q[o] )
 *
 * where alpha_q[o] = round(2^(in_exp - out_exp) × scale[o] × 2^A)
 *       A = Q16_ALPHA_BITS
 * ============================================================ */
void conv1d_q16(
    const int16_t *VITS_RESTRICT in,
    int in_ch,
    int T,
    const Conv1dQ16 *VITS_RESTRICT c,
    int16_t *VITS_RESTRICT out)
{
    const int out_ch = c->out_ch;
    const int k      = c->k;
    const int pad    = c->pad;
    const int dil    = c->dilation;

    if (T <= 0 || in_ch <= 0 || out_ch <= 0)
        return;

    const int shift = Q16_ALPHA_BITS - 1;

    /* Accumulator rows allocated up front (structured-block safe); each
       `o` uses its own disjoint slice of length T. */
    int32_t *acc_all =
        (int32_t *)malloc(sizeof(int32_t) * (size_t)out_ch * (size_t)T);
    if (!acc_all)
        return;

#ifdef _OPENMP
    const size_t work =
        (size_t)out_ch * (size_t)in_ch * (size_t)k * (size_t)T;
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        int16_t *VITS_RESTRICT out_o = out + (size_t)o * T;
        const int32_t alpha_o  = c->alpha_q[o];
        const int32_t bias_o   = c->bias_q[o];
        const int8_t  *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        int32_t *VITS_RESTRICT acc = acc_all + (size_t)o * T;

        for (int t = 0; t < T; t++)
            acc[t] = 0;

        for (int i = 0; i < in_ch; i++) {
            const int16_t *VITS_RESTRICT in_i = in + (size_t)i * T;
            const int8_t  *VITS_RESTRICT w_i  = w_o + (size_t)i * k;

            for (int j = 0; j < k; j++) {
                const int32_t wv = (int32_t)w_i[j];
                if (wv == 0)
                    continue;

                const int shift_t = j * dil - pad;

                int t0 = 0, t1 = T;
                if (shift_t < 0) t0 = -shift_t;
                if (shift_t > 0) t1 = T - shift_t;
                t0 = imax_i(t0, 0);
                t1 = imin_i(t1, T);
                if (t0 >= t1) continue;

                for (int t = t0; t < t1; t++) {
                    acc[t] += (int32_t)in_i[t + shift_t] * wv;
                }
            }
        }

        /* Requantize: (acc * alpha + 2^(A-1)) >> A + bias */
        for (int t = 0; t < T; t++) {
            int64_t scaled = (int64_t)acc[t] * alpha_o;
            int32_t result = (int32_t)((scaled + (1 << shift)) >> Q16_ALPHA_BITS);
            result += bias_o;
            out_o[t] = saturate_i16(result);
        }
    }
    free(acc_all);
}

/* ============================================================
 * ConvTranspose1D: INT16 input × INT8 weights → INT32 → INT16
 *
 * Weight layout: [out_ch][in_ch][k]
 * out pos = t*stride - pad + j
 *
 * Vectorized over j (kernel dim) for SIMD-friendly inner loop.
 * ============================================================ */
void conv_transpose1d_q16(
    const int16_t *VITS_RESTRICT in,
    int in_ch,
    int T,
    const ConvTranspose1dQ16 *VITS_RESTRICT c,
    int16_t *VITS_RESTRICT out,
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

    const int shift = Q16_ALPHA_BITS - 1;

    /* Accumulator rows allocated up front (structured-block safe); each
       `o` uses its own disjoint slice of length oT. */
    int32_t *acc_all =
        (int32_t *)malloc(sizeof(int32_t) * (size_t)out_ch * (size_t)oT);
    if (!acc_all)
        return;

#ifdef _OPENMP
    const size_t work =
        (size_t)out_ch * (size_t)in_ch * (size_t)T * (size_t)k;
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        int16_t *VITS_RESTRICT out_o = out + (size_t)o * oT;
        const int32_t alpha_o  = c->alpha_q[o];
        const int32_t bias_o   = c->bias_q[o];
        const int8_t  *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        int32_t *VITS_RESTRICT acc = acc_all + (size_t)o * oT;

        for (int t = 0; t < oT; t++)
            acc[t] = 0;

        for (int i = 0; i < in_ch; i++) {
            const int16_t *VITS_RESTRICT in_i = in + (size_t)i * T;
            const int8_t  *VITS_RESTRICT w_i  = w_o + (size_t)i * k;

            for (int j = 0; j < k; j++) {
                const int32_t wv = (int32_t)w_i[j];
                if (wv == 0)
                    continue;

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
                    acc[pos] += (int32_t)in_i[t] * wv;
                }
            }
        }

        for (int t = 0; t < oT; t++) {
            int64_t scaled = (int64_t)acc[t] * alpha_o;
            int32_t result = (int32_t)((scaled + (1 << shift)) >> Q16_ALPHA_BITS);
            result += bias_o;
            out_o[t] = saturate_i16(result);
        }
    }
    free(acc_all);
}

/* ============================================================
 * Integer LeakyReLU
 *
 *   x >= 0:  y = x
 *   x < 0:   y = (x * slope_num) / slope_den   (truncated toward 0)
 *
 * For slope=0.1: slope_num=13, slope_den=128  (13/128≈0.1016)
 * For slope=0.01: slope_num=1,  slope_den=128 (1/128≈0.0078)
 * ============================================================ */
void leaky_relu_q16(int16_t *VITS_RESTRICT x, int n,
                    int slope_num, int slope_den)
{
    const int32_t num  = slope_num;
    const int32_t den  = slope_den;
    const int32_t half = den / 2;

    if (num == den) {
        /* slope = 1.0: no-op */
        return;
    }

    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)x[i];
        if (v < 0) {
            /* Round: add half denominator before truncating */
            int32_t neg = -v;
            int32_t result = (neg * num + half) / den;
            v = -result;
        }
        x[i] = (int16_t)v;
    }
}

/* ============================================================
 * Residual add: dst += src (both INT16, same scale)
 * ============================================================ */
void residual_add_q16(int16_t *VITS_RESTRICT dst,
                      const int16_t *VITS_RESTRICT src, int n)
{
    for (int i = 0; i < n; i++) {
        int32_t sum = (int32_t)dst[i] + (int32_t)src[i];
        dst[i] = saturate_i16(sum);
    }
}

/* ============================================================
 * MRF divide-by-3: x = x / 3 (integer, rounded)
 *
 *   1/3 ≈ 21845 / 65536
 * ============================================================ */
void mrf_div3_q16(int16_t *VITS_RESTRICT x, int n)
{
    const int32_t MUL = 21845;   /* 1/3 in Q15.16 */
    const int32_t RND = 32768;   /* 2^15 */

    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)x[i];
        int32_t result = (v * MUL + RND) >> 16;
        x[i] = (int16_t)result;
    }
}

/* ============================================================
 * Tanh LUT (Phase 10)
 *
 * LUT maps |x| in [0, 4] to tanh(x) * 32767.
 * x is INT16 with the given exponent: real = x * 2^exp
 *
 * lut size: round(4 / 2^exp) + 1 entries for non-negative values.
 * Negative input: negate the LUT result.
 * |real| > 4: clamp to ±32767.
 * ============================================================ */
void tanh_lut_init(int16_t lut[], int exp)
{
    const int lut_size = 32768;
    const float scale = (float)ldexp(1.0, (double)exp);

    for (int i = 0; i < lut_size; i++) {
        float x = (float)i * scale;
        float t = tanhf(x);
        if (t > 1.0f) t = 1.0f;
        lut[i] = (int16_t)lroundf(t * 32767.0f);
    }
}

int16_t tanh_q16(const int16_t *VITS_RESTRICT in, int n, int exp,
                 const int16_t *VITS_RESTRICT lut, int16_t *VITS_RESTRICT out)
{
    (void)exp;
    const int lut_size = 32768;
    int clamped = 0;

    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)in[i];
        int idx = v < 0 ? -v : v;
        if (idx >= lut_size) {
            idx = lut_size - 1;
            clamped++;
        }
        int16_t result = lut[idx];
        if (v < 0) result = (int16_t)(-result);
        out[i] = result;
    }
    return clamped;
}
