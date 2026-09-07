#include "vits.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

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
 * AVX2 helpers
 * ============================================================ */

static inline __m256 vits_fmadd256(
    __m256 a,
    __m256 b,
    __m256 c)
{
#if defined(__FMA__) || defined(_MSC_VER)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(
        _mm256_mul_ps(a, b),
        c
    );
#endif
}


static inline __m128 vits_fmadd128(
    __m128 a,
    __m128 b,
    __m128 c)
{
#if defined(__FMA__) || defined(_MSC_VER)
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(
        _mm_mul_ps(a, b),
        c
    );
#endif
}


/*
 * Fill float array using AVX2.
 */
static inline void fill_f32_avx2(
    float *VITS_RESTRICT dst,
    int n,
    float value)
{
    if (value == 0.0f) {
        memset(dst, 0, sizeof(float) * (size_t)n);
        return;
    }

    const __m256 v = _mm256_set1_ps(value);

    int i = 0;

    /* 4x unroll */
    for (; i + 31 < n; i += 32) {
        _mm256_storeu_ps(dst + i +  0, v);
        _mm256_storeu_ps(dst + i +  8, v);
        _mm256_storeu_ps(dst + i + 16, v);
        _mm256_storeu_ps(dst + i + 24, v);
    }

    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(dst + i, v);
    }

    for (; i < n; i++) {
        dst[i] = value;
    }
}


/* ============================================================
 * Quantized Conv1D
 *
 * Weight:
 *
 *   int8 [out_ch][in_ch][k]
 *
 * Instead of:
 *
 *   accum += q * x
 *   output = accum * scale + bias
 *
 * we calculate:
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
         *
         * This means we no longer need the final
         *
         *     output *= scale
         *     output += bias
         *
         * pass.
         */
        fill_f32_avx2(out_o, T, bias_o);

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

                const __m256 wv =
                    _mm256_set1_ps(weight);

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

                /*
                 * 32 floats per iteration.
                 *
                 * 4 independent dependency chains help
                 * hide load/FMA latency.
                 */
                for (; t + 31 < t1; t += 32) {

                    __m256 y0 =
                        _mm256_loadu_ps(out_o + t + 0);
                    __m256 y1 =
                        _mm256_loadu_ps(out_o + t + 8);
                    __m256 y2 =
                        _mm256_loadu_ps(out_o + t + 16);
                    __m256 y3 =
                        _mm256_loadu_ps(out_o + t + 24);

                    const __m256 x0 =
                        _mm256_loadu_ps(src + t + 0);
                    const __m256 x1 =
                        _mm256_loadu_ps(src + t + 8);
                    const __m256 x2 =
                        _mm256_loadu_ps(src + t + 16);
                    const __m256 x3 =
                        _mm256_loadu_ps(src + t + 24);

                    y0 = vits_fmadd256(x0, wv, y0);
                    y1 = vits_fmadd256(x1, wv, y1);
                    y2 = vits_fmadd256(x2, wv, y2);
                    y3 = vits_fmadd256(x3, wv, y3);

                    _mm256_storeu_ps(
                        out_o + t + 0,
                        y0
                    );

                    _mm256_storeu_ps(
                        out_o + t + 8,
                        y1
                    );

                    _mm256_storeu_ps(
                        out_o + t + 16,
                        y2
                    );

                    _mm256_storeu_ps(
                        out_o + t + 24,
                        y3
                    );
                }

                /*
                 * AVX2 remainder.
                 */
                for (; t + 7 < t1; t += 8) {

                    const __m256 x =
                        _mm256_loadu_ps(src + t);

                    __m256 y =
                        _mm256_loadu_ps(out_o + t);

                    y = vits_fmadd256(x, wv, y);

                    _mm256_storeu_ps(
                        out_o + t,
                        y
                    );
                }

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
 * The obvious AVX implementation:
 *
 *      vectorize(t)
 *
 * is BAD for stride > 1 because output becomes:
 *
 *      out[t*stride]
 *
 * AVX2 has no scatter instruction.
 *
 *
 * Instead:
 *
 *      for t
 *          vectorize(j)
 *
 * because:
 *
 *      out[t*stride + j]
 *
 * is contiguous over j.
 *
 *
 * We go one step further and use:
 *
 *      for j_block
 *          load/convert weights ONCE
 *          for t
 *
 * This avoids int8->float conversion for every sample.
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
         * Bias initialization also removes the final
         * scale/bias output pass.
         */
        fill_f32_avx2(
            out_o,
            oT,
            bias_o
        );

        if (scale_o == 0.0f)
            continue;

        const __m256 scale8 =
            _mm256_set1_ps(scale_o);

        const __m128 scale4 =
            _mm_set1_ps(scale_o);

        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i =
                in + (size_t)i * T;

            const int8_t *VITS_RESTRICT w_i =
                w_o + (size_t)i * k;

            int j = 0;

            /* ==================================================
             * 8 kernel coefficients at once.
             * ================================================== */
            for (; j + 7 < k; j += 8) {

                /*
                 * Load:
                 *
                 *    8 x int8
                 *
                 * Convert:
                 *
                 *    int8
                 *      ↓
                 *    int32
                 *      ↓
                 *    float32
                 */

                const __m128i q8 =
                    _mm_loadl_epi64(
                        (const __m128i *)(w_i + j)
                    );

                const __m256i q32 =
                    _mm256_cvtepi8_epi32(q8);

                __m256 wv =
                    _mm256_cvtepi32_ps(q32);

                /*
                 * Dequantize once for this block.
                 */
                wv =
                    _mm256_mul_ps(
                        wv,
                        scale8
                    );

                /*
                 * [t0, t1) is the range where ALL 8 coefficients
                 * have a valid output position:
                 *
                 *    t0 = tightest start = max over jj of lo(jj)
                 *         (lo decreases with jj, so it is lo(j))
                 *    t1 = tightest end   = min over jj of hi(jj)
                 *         (hi decreases with jj, so it is hi(j+7))
                 *
                 * Coefficients near the block edges may start
                 * earlier or end later than that shared range.
                 * Those boundary t's need a per-coefficient scalar
                 * pass (a handful of samples per row).
                 */

                int t0 = 0;
                int t1 = T;

                const int lower =
                    pad - j;

                if (lower > 0) {
                    t0 =
                        (lower + stride - 1)
                        / stride;
                }

                const int upper =
                    oT - 1
                    + pad
                    - j
                    - 7;

                if (upper >= 0) {
                    t1 =
                        imin_i(
                            t1,
                            upper / stride + 1
                        );
                } else {
                    t1 = 0;
                }

                t0 =
                    imax_i(
                        t0,
                        0
                    );

                if (t0 >= t1) {
                    /*
                     * No shared interior (tiny output):
                     * per-coefficient scalar pass over each
                     * coefficient's full [lo, hi) range.
                     */
                    for (int jj = 0; jj < 8; jj++) {
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
                        if (ub >= 0)
                            hi = imin_i(T, ub / stride + 1);

                        for (int t = lo; t < hi; t++) {
                            const int pos =
                                t * stride - pad + jabs;
                            out_o[pos] +=
                                weight * in_i[t];
                        }
                    }
                    continue;
                }

                /*
                 * Left boundary: coefficients jj > j may start at
                 * an earlier t than t0.
                 */
                for (int jj = 1; jj < 8; jj++) {
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
                        out_o[pos] +=
                            weight * in_i[t];
                    }
                }

                /*
                 * Vector body: for each input sample t in [t0, t1)
                 *
                 * x = scalar
                 *
                 * x * [w0..w7]
                 *
                 * goes to 8 contiguous outputs.
                 */
                for (int t = t0; t < t1; t++) {

                    const int base =
                        t * stride
                        - pad
                        + j;

                    const __m256 xv =
                        _mm256_set1_ps(
                            in_i[t]
                        );

                    __m256 y =
                        _mm256_loadu_ps(
                            out_o + base
                        );

                    y =
                        vits_fmadd256(
                            xv,
                            wv,
                            y
                        );

                    _mm256_storeu_ps(
                        out_o + base,
                        y
                    );
                }

                /*
                 * Right boundary: coefficients jj < j+7 may end at
                 * a later t than t1.
                 */
                for (int jj = 0; jj < 7; jj++) {
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
                        out_o[pos] +=
                            weight * in_i[t];
                    }
                }
            }

            /* ==================================================
             * 4 kernel coefficients.
             *
             * Important for VITS because some transposed
             * convolution kernels are only k=4.
             *
             * AVX2 implies SSE4.1 support, so use 128-bit SIMD.
             * ================================================== */
            if (j + 3 < k) {

                uint32_t packed;

                memcpy(
                    &packed,
                    w_i + j,
                    sizeof(packed)
                );

                const __m128i q8 =
                    _mm_cvtsi32_si128(
                        (int)packed
                    );

                const __m128i q32 =
                    _mm_cvtepi8_epi32(q8);

                __m128 wv =
                    _mm_cvtepi32_ps(q32);

                wv =
                    _mm_mul_ps(
                        wv,
                        scale4
                    );

                /*
                 * Same scheme as the 8-coefficient block above,
                 * with the shared range [lo(j), hi(j+3)) for the
                 * vector body and scalar passes for the boundary
                 * t's of the 4 coefficients.
                 */

                int t0 = 0;
                int t1 = T;

                const int lower =
                    pad - j;

                if (lower > 0) {
                    t0 =
                        (lower + stride - 1)
                        / stride;
                }

                const int upper =
                    oT - 1
                    + pad
                    - j
                    - 3;

                if (upper >= 0) {
                    t1 =
                        imin_i(
                            t1,
                            upper / stride + 1
                        );
                } else {
                    t1 = 0;
                }

                t0 =
                    imax_i(
                        t0,
                        0
                    );

                if (t0 >= t1) {
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
                        if (ub >= 0)
                            hi = imin_i(T, ub / stride + 1);

                        for (int t = lo; t < hi; t++) {
                            const int pos =
                                t * stride - pad + jabs;
                            out_o[pos] +=
                                weight * in_i[t];
                        }
                    }
                } else {

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
                            out_o[pos] +=
                                weight * in_i[t];
                        }
                    }

                    for (int t = t0;
                         t < t1;
                         t++) {

                        const int base =
                            t * stride
                            - pad
                            + j;

                        const __m128 xv =
                            _mm_set1_ps(
                                in_i[t]
                            );

                        __m128 y =
                            _mm_loadu_ps(
                                out_o + base
                            );

                        y =
                            vits_fmadd128(
                                xv,
                                wv,
                                y
                            );

                        _mm_storeu_ps(
                            out_o + base,
                            y
                        );
                    }

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
                            out_o[pos] +=
                                weight * in_i[t];
                        }
                    }
                }

                j += 4;
            }

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

                if (lower > 0) {
                    t0 =
                        (lower + stride - 1)
                        / stride;
                }

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