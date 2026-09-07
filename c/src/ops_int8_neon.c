#include "vits.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VITS_HAS_NEON 1
#else
#define VITS_HAS_NEON 0
#endif

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

static inline int imin_i(int a, int b)
{
    return a < b ? a : b;
}

static inline int imax_i(int a, int b)
{
    return a > b ? a : b;
}

/* ============================================================
 * NEON helpers
 * ============================================================ */

#if VITS_HAS_NEON

static inline float32x4_t vits_mlaq_f32(float32x4_t acc,
                                        float32x4_t x,
                                        float32x4_t w)
{
#if defined(__aarch64__)
    return vfmaq_f32(acc, x, w);
#else
    return vmlaq_f32(acc, x, w);
#endif
}

static inline float32x4_t vits_mlaq_n_f32(float32x4_t acc,
                                          float32x4_t x,
                                          float w)
{
#if defined(__aarch64__)
    return vfmaq_n_f32(acc, x, w);
#else
    return vmlaq_n_f32(acc, x, w);
#endif
}

static inline void fill_f32_neon(float *VITS_RESTRICT dst,
                                 int n,
                                 float value)
{
    if (value == 0.0f) {
        memset(dst, 0, sizeof(float) * (size_t)n);
        return;
    }

    int i = 0;
    const float32x4_t vv = vdupq_n_f32(value);

    for (; i + 15 < n; i += 16) {
        vst1q_f32(dst + i + 0,  vv);
        vst1q_f32(dst + i + 4,  vv);
        vst1q_f32(dst + i + 8,  vv);
        vst1q_f32(dst + i + 12, vv);
    }
    for (; i + 3 < n; i += 4)
        vst1q_f32(dst + i, vv);
    for (; i < n; i++)
        dst[i] = value;
}

/*
 * Convert 4 int8 values to float32x4.
 *
 * aarch64:
 *   int8x8 → (vmovl_s8) → int16x8 → (vmovl_s16 low) → int32x4 → (vcvtq) → float32x4
 *
 * ARMv7:
 *   Same logic via 32-bit register intrinsics.
 */
static inline float32x4_t int8x4_to_f32x4(const int8_t *p)
{
#if defined(__aarch64__)
    const int8x8_t q8 = vld1_s8(p);
    const int16x8_t q16 = vmovl_s8(q8);
    const int32x4_t q32 = vmovl_s16(vget_low_s16(q16));
    return vcvtq_f32_s32(q32);
#else
    const int8x8_t q8 = vld1_s8(p);
    const int16x4_t q16 = vmovl_s8(vget_low_s8(q8));
    const int32x2_t q32_lo = vmovl_s16(vget_low_s16(q16));
    const int32x2_t q32_hi = vmovl_s16(vget_high_s16(q16));
    const float32x2_t f_lo = vcvt_f32_s32(q32_lo);
    const float32x2_t f_hi = vcvt_f32_s32(q32_hi);
    return vcombine_f32(f_lo, f_hi);
#endif
}

#endif /* VITS_HAS_NEON */


/* ============================================================
 * Quantized Conv1D
 *
 * Weight:
 *
 *   int8 [out_ch][in_ch][k]
 *
 * Computed as:
 *
 *   output = bias
 *   output += (q * scale) * x
 *
 * This removes one complete pass over output.
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

    if (T <= 0 ||
        in_ch <= 0 ||
        out_ch <= 0)
        return;

    const size_t work =
        (size_t)out_ch *
        (size_t)in_ch *
        (size_t)k *
        (size_t)T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        float *VITS_RESTRICT out_o =
            out + (size_t)o * T;

        const float scale_o = c->scale[o];

        const float bias_o =
            c->bias ? c->bias[o] : 0.0f;

        const int8_t *VITS_RESTRICT w_o =
            c->weight +
            (size_t)o * in_ch * k;

        /*
         * Start accumulator with bias.
         */
#if VITS_HAS_NEON
        fill_f32_neon(out_o, T, bias_o);
#else
        for (int t = 0; t < T; t++)
            out_o[t] = bias_o;
#endif

        /*
         * scale == 0 means every weight becomes zero.
         */
        if (scale_o == 0.0f)
            continue;

        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i =
                in + (size_t)i * T;

            const int8_t *VITS_RESTRICT w_i =
                w_o + (size_t)i * k;

            for (int j = 0; j < k; j++) {

                const int q = (int)w_i[j];

                if (q == 0)
                    continue;

                /*
                 * Dequantize weight once.
                 */
                const float weight =
                    (float)q * scale_o;

                const int shift =
                    j * dil - pad;

                int t0 = 0;
                int t1 = T;

                if (shift < 0)
                    t0 = -shift;

                if (shift > 0)
                    t1 = T - shift;

                t0 = imax_i(t0, 0);
                t1 = imin_i(t1, T);

                if (t0 >= t1)
                    continue;

                const float *VITS_RESTRICT src =
                    in_i + shift;

                int t = t0;

#if VITS_HAS_NEON
                /*
                 * 4 floats per iteration.
                 */
                for (; t + 3 < t1; t += 4) {

                    const float32x4_t x =
                        vld1q_f32(src + t);

                    float32x4_t y =
                        vld1q_f32(out_o + t);

                    y = vits_mlaq_n_f32(y, x, weight);

                    vst1q_f32(out_o + t, y);
                }
#endif

                /*
                 * Scalar tail.
                 */
                for (; t < t1; t++) {
                    out_o[t] +=
                        weight * src[t];
                }
            }
        }
    }
}


/* ============================================================
 * ConvTranspose1D
 *
 * Vectorizes the kernel dimension (4 coefficients at a time).
 * For one input sample, 4 contiguous weights update 4 contiguous
 * output positions.
 *
 * Strategy:
 *   for j_block (4 at a time)
 *       load/convert weights ONCE
 *       for t
 *           FMA into 4 contiguous outputs
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

    if (T <= 0 ||
        in_ch <= 0 ||
        out_ch <= 0 ||
        stride <= 0) {

        *out_T = 0;
        return;
    }

    const int oT =
        (T - 1) * stride
        - 2 * pad
        + k;

    *out_T = oT;

    if (oT <= 0)
        return;

    const size_t work =
        (size_t)out_ch *
        (size_t)in_ch *
        (size_t)T *
        (size_t)k;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        float *VITS_RESTRICT out_o =
            out + (size_t)o * oT;

        const float scale_o =
            c->scale[o];

        const float bias_o =
            c->bias ? c->bias[o] : 0.0f;

        const int8_t *VITS_RESTRICT w_o =
            c->weight +
            (size_t)o * in_ch * k;

        /*
         * Bias initialization.
         */
#if VITS_HAS_NEON
        fill_f32_neon(out_o, oT, bias_o);
#else
        for (int p = 0; p < oT; p++)
            out_o[p] = bias_o;
#endif

        if (scale_o == 0.0f)
            continue;

        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i =
                in + (size_t)i * T;

            const int8_t *VITS_RESTRICT w_i =
                w_o + (size_t)i * k;

            int j = 0;

            /* ==================================================
             * 4 kernel coefficients at once.
             * ================================================== */
#if VITS_HAS_NEON
            for (; j + 3 < k; j += 4) {

                /*
                 * Load 4 int8, convert to 4 float32,
                 * dequantize once for this block.
                 */
                float32x4_t wv = int8x4_to_f32x4(w_i + j);
                wv = vmulq_f32(wv, vdupq_n_f32(scale_o));

                /*
                 * Shared valid t range for all 4 coefficients:
                 *
                 *   t0 = max over jj of lo(j+jj)
                 *   t1 = min over jj of hi(j+jj)
                 *
                 * Since lo decreases with j and hi also decreases
                 * with j:
                 *   t0 = lo(j)        (first coefficient starts latest)
                 *   t1 = hi(j+3)      (last coefficient ends earliest)
                 */
                int t0 = 0;
                int t1 = T;

                const int lower = pad - j;
                if (lower > 0)
                    t0 = (lower + stride - 1) / stride;

                const int upper = oT - 1 + pad - j - 3;
                if (upper < 0) {
                    t1 = 0;
                } else {
                    const int max_t = upper / stride;
                    t1 = imin_i(t1, max_t + 1);
                }

                t0 = imax_i(t0, 0);

                if (t0 < t1) {
                    /*
                     * Left boundary: coefficients jj > 0 may start
                     * earlier than t0.
                     */
                    for (int jj = 1; jj < 4; jj++) {
                        const int jabs = j + jj;

                        int lo = 0;
                        const int lb = pad - jabs;
                        if (lb > 0)
                            lo = (lb + stride - 1) / stride;
                        lo = imax_i(lo, 0);

                        if (lo >= t0)
                            continue;

                        const float weight =
                            (float)w_i[jabs] * scale_o;

                        if (weight == 0.0f)
                            continue;

                        for (int t = lo; t < t0; t++) {
                            const int pos =
                                t * stride - pad + jabs;
                            out_o[pos] += weight * in_i[t];
                        }
                    }

                    /*
                     * Vector body: 4 contiguous outputs.
                     */
                    for (int t = t0; t < t1; t++) {

                        const int base =
                            t * stride
                            - pad
                            + j;

                        const float32x4_t xv =
                            vdupq_n_f32(in_i[t]);

                        float32x4_t y =
                            vld1q_f32(out_o + base);

                        y = vits_mlaq_f32(y, xv, wv);

                        vst1q_f32(out_o + base, y);
                    }

                    /*
                     * Right boundary: coefficients jj < 3 may end
                     * later than t1.
                     */
                    for (int jj = 0; jj < 3; jj++) {
                        const int jabs = j + jj;

                        int hi = 0;
                        const int ub = oT - 1 + pad - jabs;
                        if (ub >= 0)
                            hi = imin_i(T, ub / stride + 1);

                        if (hi <= t1)
                            continue;

                        const float weight =
                            (float)w_i[jabs] * scale_o;

                        if (weight == 0.0f)
                            continue;

                        for (int t = t1; t < hi; t++) {
                            const int pos =
                                t * stride - pad + jabs;
                            out_o[pos] += weight * in_i[t];
                        }
                    }
                } else {
                    /*
                     * No shared interior — per-coefficient scalar.
                     */
                    for (int jj = 0; jj < 4; jj++) {
                        const int jabs = j + jj;

                        const float weight =
                            (float)w_i[jabs] * scale_o;

                        if (weight == 0.0f)
                            continue;

                        int lo = 0;
                        const int lb = pad - jabs;
                        if (lb > 0)
                            lo = (lb + stride - 1) / stride;

                        int hi = 0;
                        const int ub = oT - 1 + pad - jabs;
                        if (ub < 0)
                            continue;
                        hi = imin_i(T, ub / stride + 1);

                        for (int t = lo; t < hi; t++) {
                            const int pos =
                                t * stride - pad + jabs;
                            out_o[pos] += weight * in_i[t];
                        }
                    }
                }
            }
#endif

            /* ==================================================
             * Scalar tail: at most 3 coefficients.
             * ================================================== */
            for (; j < k; j++) {

                const int q =
                    (int)w_i[j];

                if (q == 0)
                    continue;

                const float weight =
                    (float)q * scale_o;

                int t0 = 0;
                int t1 = T;

                const int lower =
                    pad - j;

                if (lower > 0)
                    t0 =
                        (lower + stride - 1)
                        / stride;

                const int upper =
                    oT - 1
                    + pad
                    - j;

                if (upper < 0)
                    continue;

                const int max_t =
                    upper / stride;

                t1 =
                    imin_i(
                        t1,
                        max_t + 1
                    );

                t0 =
                    imax_i(
                        t0,
                        0
                    );

                for (int t = t0;
                     t < t1;
                     t++) {

                    const int pos =
                        t * stride
                        - pad
                        + j;

                    out_o[pos] +=
                        weight * in_i[t];
                }
            }
        }
    }
}
