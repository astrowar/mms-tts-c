#include "vits.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef VITS_OMP_WORK_THRESHOLD
#define VITS_OMP_WORK_THRESHOLD 32768
#endif

#ifndef VITS_OMP_VECTOR_THRESHOLD
#define VITS_OMP_VECTOR_THRESHOLD 65536
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

#ifdef __AVX2__

static inline __m256 vits_fmadd256(__m256 a, __m256 b, __m256 c)
{
#ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

static inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);

    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);

    return _mm_cvtss_f32(sum);
}

static inline float hmax256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);

    __m128 m = _mm_max_ps(lo, hi);
    __m128 s = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));
    m = _mm_max_ps(m, s);
    s = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2));
    m = _mm_max_ps(m, s);

    return _mm_cvtss_f32(m);
}

static inline void fill_f32_avx2(float *VITS_RESTRICT dst, int n, float value)
{
    int i = 0;
    const __m256 vv = _mm256_set1_ps(value);

    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, vv);

    for (; i < n; i++)
        dst[i] = value;
}

#endif

/* ============================================================
 * Conv1D k=1 specialized
 *
 * Very common in VITS.
 *
 * Processes:
 *
 *      4 output channels
 *      x
 *      8 temporal positions
 *
 * per inner block.
 *
 * Input vector is loaded once and reused by 4 output channels.
 * ============================================================ */

#ifdef __AVX2__

static void conv1d_k1_avx2(
    const float *VITS_RESTRICT in,
    int in_ch,
    int T,
    const Conv1d *VITS_RESTRICT c,
    float *VITS_RESTRICT out)
{
    const int out_ch = c->out_ch;
    const int blocks = (out_ch + 3) / 4;
    const size_t work = (size_t)out_ch * in_ch * T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int block = 0; block < blocks; block++) {

        const int o0 = block * 4;
        const int nout = imin_i(4, out_ch - o0);
        int t = 0;

        /*
         * 8 time samples simultaneously.
         */
        for (; t + 7 < T; t += 8) {

            __m256 a0 = _mm256_set1_ps(c->bias[o0]);
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();

            if (nout > 1)
                a1 = _mm256_set1_ps(c->bias[o0 + 1]);

            if (nout > 2)
                a2 = _mm256_set1_ps(c->bias[o0 + 2]);

            if (nout > 3)
                a3 = _mm256_set1_ps(c->bias[o0 + 3]);

            for (int i = 0; i < in_ch; i++) {

                const __m256 x = _mm256_loadu_ps(in + (size_t)i * T + t);

                /*
                 * k=1:
                 *
                 * weight[o][i]
                 */
                {
                    const __m256 w = _mm256_set1_ps(
                        c->weight[(size_t)o0 * in_ch + i]);
                    a0 = vits_fmadd256(x, w, a0);
                }

                if (nout > 1) {
                    const __m256 w = _mm256_set1_ps(
                        c->weight[(size_t)(o0 + 1) * in_ch + i]);
                    a1 = vits_fmadd256(x, w, a1);
                }

                if (nout > 2) {
                    const __m256 w = _mm256_set1_ps(
                        c->weight[(size_t)(o0 + 2) * in_ch + i]);
                    a2 = vits_fmadd256(x, w, a2);
                }

                if (nout > 3) {
                    const __m256 w = _mm256_set1_ps(
                        c->weight[(size_t)(o0 + 3) * in_ch + i]);
                    a3 = vits_fmadd256(x, w, a3);
                }
            }

            _mm256_storeu_ps(out + (size_t)o0 * T + t, a0);

            if (nout > 1)
                _mm256_storeu_ps(out + (size_t)(o0 + 1) * T + t, a1);

            if (nout > 2)
                _mm256_storeu_ps(out + (size_t)(o0 + 2) * T + t, a2);

            if (nout > 3)
                _mm256_storeu_ps(out + (size_t)(o0 + 3) * T + t, a3);
        }

        /*
         * Scalar tail.
         */
        for (; t < T; t++) {

            for (int q = 0; q < nout; q++) {

                const int o = o0 + q;
                float acc = c->bias[o];
                const float *w = c->weight + (size_t)o * in_ch;

                for (int i = 0; i < in_ch; i++)
                    acc += w[i] * in[(size_t)i * T + t];

                out[(size_t)o * T + t] = acc;
            }
        }
    }
}

#endif

/* ============================================================
 * Conv1D
 * ============================================================ */

void conv1d(
    const float *VITS_RESTRICT in,
    int in_ch,
    int T,
    const Conv1d *VITS_RESTRICT c,
    float *VITS_RESTRICT out)
{
    const int out_ch = c->out_ch;
    const int k      = c->k;
    const int pad    = c->pad;
    const int dil    = c->dilation;

    if (T <= 0 || in_ch <= 0 || out_ch <= 0)
        return;

    static const float zero_f = 0.0f;
    const float *bias = c->bias ? c->bias : &zero_f;

#ifdef __AVX2__

    /*
     * Important fast path.
     *
     * Conv1D 1x1 becomes essentially GEMM-like.
     */
    if (k == 1 && pad == 0) {
        conv1d_k1_avx2(in, in_ch, T, c, out);
        return;
    }

#endif

    /*
     * Determine temporal region where every kernel tap
     * is valid.
     *
     * This allows removing all bounds checks from the
     * AVX2 inner loop.
     */
    int min_shift = -pad;
    int max_shift = -pad;

    for (int j = 1; j < k; j++) {
        const int s = j * dil - pad;
        if (s < min_shift)
            min_shift = s;
        if (s > max_shift)
            max_shift = s;
    }

    int interior_begin = imax_i(0, -min_shift);
    int interior_end   = imin_i(T, T - max_shift);
    if (interior_begin > T) interior_begin = T;
    if (interior_end < interior_begin) interior_end = interior_begin;

    const size_t work = (size_t)out_ch * in_ch * k * T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        float *VITS_RESTRICT out_o = out + (size_t)o * T;
        const float *VITS_RESTRICT w_o = c->weight + (size_t)o * in_ch * k;

        /*
         * ----------------------------------------------------
         * Left border
         * ----------------------------------------------------
         */
        for (int t = 0; t < interior_begin; t++) {

            float acc = bias[o];

            for (int i = 0; i < in_ch; i++) {

                const float *in_i = in + (size_t)i * T;
                const float *w_i = w_o + (size_t)i * k;

                for (int j = 0; j < k; j++) {
                    const int idx = t + j * dil - pad;
                    if ((unsigned)idx < (unsigned)T)
                        acc += w_i[j] * in_i[idx];
                }
            }

            out_o[t] = acc;
        }

        /*
         * ----------------------------------------------------
         * Main AVX2 region
         * ----------------------------------------------------
         */
        int t = interior_begin;

#ifdef __AVX2__

        /*
         * Keep output accumulator completely in
         * YMM register.
         *
         * This is much better than repeatedly
         * loading/storing output for each input channel.
         */
        for (; t + 7 < interior_end; t += 8) {

            __m256 acc = _mm256_set1_ps(bias[o]);

            for (int i = 0; i < in_ch; i++) {

                const float *VITS_RESTRICT in_i = in + (size_t)i * T;
                const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;

                for (int j = 0; j < k; j++) {

                    const float weight = w_i[j];
                    if (weight == 0.0f)
                        continue;

                    const int shift = j * dil - pad;

                    const __m256 x = _mm256_loadu_ps(in_i + t + shift);
                    const __m256 wv = _mm256_set1_ps(weight);

                    acc = vits_fmadd256(x, wv, acc);
                }
            }

            _mm256_storeu_ps(out_o + t, acc);
        }

#endif

        /*
         * Remaining interior samples.
         */
        for (; t < interior_end; t++) {

            float acc = bias[o];

            for (int i = 0; i < in_ch; i++) {

                const float *in_i = in + (size_t)i * T;
                const float *w_i = w_o + (size_t)i * k;

                for (int j = 0; j < k; j++)
                    acc += w_i[j] * in_i[t + j * dil - pad];
            }

            out_o[t] = acc;
        }

        /*
         * ----------------------------------------------------
         * Right border
         * ----------------------------------------------------
         */
        for (t = interior_end; t < T; t++) {

            float acc = bias[o];

            for (int i = 0; i < in_ch; i++) {

                const float *in_i = in + (size_t)i * T;
                const float *w_i = w_o + (size_t)i * k;

                for (int j = 0; j < k; j++) {
                    const int idx = t + j * dil - pad;
                    if ((unsigned)idx < (unsigned)T)
                        acc += w_i[j] * in_i[idx];
                }
            }

            out_o[t] = acc;
        }
    }
}

/* ============================================================
 * Depthwise Conv1D
 * ============================================================ */

void conv1d_depthwise(
    const float *VITS_RESTRICT in,
    int ch,
    int T,
    const float *VITS_RESTRICT w,
    const float *VITS_RESTRICT b,
    int k,
    int dilation,
    int pad,
    float *VITS_RESTRICT out)
{
    if (T <= 0 || ch <= 0)
        return;

    int min_shift = -pad;
    int max_shift = -pad;

    for (int j = 1; j < k; j++) {
        const int shift = j * dilation - pad;
        if (shift < min_shift)
            min_shift = shift;
        if (shift > max_shift)
            max_shift = shift;
    }

    int interior_begin = imax_i(0, -min_shift);
    int interior_end   = imin_i(T, T - max_shift);
    if (interior_begin > T) interior_begin = T;
    if (interior_end < interior_begin) interior_end = interior_begin;

    const size_t work = (size_t)ch * k * T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int cidx = 0; cidx < ch; cidx++) {

        const float *VITS_RESTRICT in_c = in + (size_t)cidx * T;
        float *VITS_RESTRICT out_c = out + (size_t)cidx * T;
        const float *VITS_RESTRICT w_c = w + (size_t)cidx * k;

        /*
         * Left edge.
         */
        for (int t = 0; t < interior_begin; t++) {

            float acc = b[cidx];

            for (int j = 0; j < k; j++) {
                const int idx = t + j * dilation - pad;
                if ((unsigned)idx < (unsigned)T)
                    acc += w_c[j] * in_c[idx];
            }

            out_c[t] = acc;
        }

        int t = interior_begin;

#ifdef __AVX2__

        /*
         * 8 time samples simultaneously.
         */
        for (; t + 7 < interior_end; t += 8) {

            __m256 acc = _mm256_set1_ps(b[cidx]);

            for (int j = 0; j < k; j++) {

                const float weight = w_c[j];
                if (weight == 0.0f)
                    continue;

                const int shift = j * dilation - pad;

                const __m256 x = _mm256_loadu_ps(in_c + t + shift);
                const __m256 ww = _mm256_set1_ps(weight);

                acc = vits_fmadd256(x, ww, acc);
            }

            _mm256_storeu_ps(out_c + t, acc);
        }

#endif

        for (; t < interior_end; t++) {

            float acc = b[cidx];

            for (int j = 0; j < k; j++)
                acc += w_c[j] * in_c[t + j * dilation - pad];

            out_c[t] = acc;
        }

        /*
         * Right edge.
         */
        for (t = interior_end; t < T; t++) {

            float acc = b[cidx];

            for (int j = 0; j < k; j++) {
                const int idx = t + j * dilation - pad;
                if ((unsigned)idx < (unsigned)T)
                    acc += w_c[j] * in_c[idx];
            }

            out_c[t] = acc;
        }
    }
}

/* ============================================================
 * ConvTranspose1D
 *
 * AVX2 vectorizes kernel dimension.
 *
 * Since kernel coefficients and corresponding output samples
 * are contiguous, AVX2 can process 8 kernel taps simultaneously.
 * ============================================================ */

void conv_transpose1d(
    const float *VITS_RESTRICT in,
    int in_ch,
    int T,
    const ConvTranspose1d *VITS_RESTRICT c,
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

    static const float zero_f = 0.0f;
    const float *bias = c->bias ? c->bias : &zero_f;

    const int oT = (T - 1) * stride - 2 * pad + k;
    *out_T = oT;

    if (oT <= 0)
        return;

    const size_t work = (size_t)out_ch * in_ch * T * k;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; o++) {

        float *VITS_RESTRICT out_o = out + (size_t)o * oT;

        /*
         * Initialize with bias.
         */
#ifdef __AVX2__
        fill_f32_avx2(out_o, oT, bias[o]);
#else
        for (int p = 0; p < oT; p++)
            out_o[p] = bias[o];
#endif

        const float *VITS_RESTRICT w_o = c->weight + (size_t)o * in_ch * k;

        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i = in + (size_t)i * T;
            const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;

            for (int t = 0; t < T; t++) {

                const float x = in_i[t];
                if (x == 0.0f)
                    continue;

                const int base = t * stride - pad;

                /*
                 * Determine valid kernel interval.
                 */
                int j0 = imax_i(0, -base);
                int j1 = imin_i(k, oT - base);

                if (j0 >= j1)
                    continue;

                int j = j0;

#ifdef __AVX2__

                const __m256 xv = _mm256_set1_ps(x);

                for (; j + 7 < j1; j += 8) {

                    __m256 ov = _mm256_loadu_ps(out_o + base + j);
                    const __m256 wv = _mm256_loadu_ps(w_i + j);

                    ov = vits_fmadd256(xv, wv, ov);

                    _mm256_storeu_ps(out_o + base + j, ov);
                }

#endif

                for (; j < j1; j++)
                    out_o[base + j] += x * w_i[j];
            }
        }
    }
}

/* ============================================================
 * LayerNorm
 *
 * Input layout:
 *
 *     [dim][T]
 *
 * Instead of vectorizing across dim, vectorize 8 independent
 * time positions.
 *
 * This fits the memory layout perfectly:
 *
 * in[i*T + t ... t+7]
 * ============================================================ */

void layer_norm(
    const float *VITS_RESTRICT in,
    int T,
    int dim,
    const LayerNorm *VITS_RESTRICT ln,
    float *VITS_RESTRICT out)
{
    if (T <= 0 || dim <= 0)
        return;

#ifdef __AVX2__

    const int blocks = T / 8;
    const size_t work = (size_t)T * dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int block = 0; block < blocks; block++) {

        const int t = block * 8;

        /*
         * Mean for 8 temporal positions.
         */
        __m256 mean = _mm256_setzero_ps();

        for (int i = 0; i < dim; i++) {
            const __m256 x = _mm256_loadu_ps(in + (size_t)i * T + t);
            mean = _mm256_add_ps(mean, x);
        }

        mean = _mm256_mul_ps(mean, _mm256_set1_ps(1.0f / (float)dim));

        /*
         * Variance.
         */
        __m256 var = _mm256_setzero_ps();

        for (int i = 0; i < dim; i++) {
            const __m256 x = _mm256_loadu_ps(in + (size_t)i * T + t);
            const __m256 d = _mm256_sub_ps(x, mean);
            var = vits_fmadd256(d, d, var);
        }

        var = _mm256_mul_ps(var, _mm256_set1_ps(1.0f / (float)dim));
        var = _mm256_add_ps(var, _mm256_set1_ps(LN_EPS));

        /*
         * Accurate sqrt instead of approximate rsqrt.
         */
        const __m256 inv = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(var));

        /*
         * Normalize.
         */
        for (int i = 0; i < dim; i++) {

            const __m256 x = _mm256_loadu_ps(in + (size_t)i * T + t);

            __m256 y = _mm256_sub_ps(x, mean);
            y = _mm256_mul_ps(y, inv);

            const __m256 weight = _mm256_set1_ps(ln->weight[i]);
            const __m256 bias   = _mm256_set1_ps(ln->bias[i]);

            y = vits_fmadd256(y, weight, bias);

            _mm256_storeu_ps(out + (size_t)i * T + t, y);
        }
    }

    /*
     * Scalar tail.
     */
    for (int t = blocks * 8; t < T; t++) {

        float mean = 0.0f;

        for (int i = 0; i < dim; i++)
            mean += in[(size_t)i * T + t];

        mean /= (float)dim;

        float var = 0.0f;

        for (int i = 0; i < dim; i++) {
            const float d = in[(size_t)i * T + t] - mean;
            var += d * d;
        }

        var /= (float)dim;

        const float inv = 1.0f / sqrtf(var + LN_EPS);

        for (int i = 0; i < dim; i++) {

            const size_t idx = (size_t)i * T + t;
            out[idx] = ln->weight[i] * (in[idx] - mean) * inv + ln->bias[i];
        }
    }

#else

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < T; t++) {

        float mean = 0.0f;

        for (int i = 0; i < dim; i++)
            mean += in[(size_t)i * T + t];

        mean /= (float)dim;

        float var = 0.0f;

        for (int i = 0; i < dim; i++) {
            const float d = in[(size_t)i * T + t] - mean;
            var += d * d;
        }

        var /= (float)dim;

        const float inv = 1.0f / sqrtf(var + LN_EPS);

        for (int i = 0; i < dim; i++) {

            const size_t idx = (size_t)i * T + t;
            out[idx] = ln->weight[i] * (in[idx] - mean) * inv + ln->bias[i];
        }
    }

#endif
}

/* ============================================================
 * Activations
 * ============================================================ */

void gelu(float *x, int n)
{
    /*
     * AVX2 itself has no vector tanh.
     *
     * With:
     *
     *     -O3 -ffast-math
     *
     * GCC/Clang may vectorize this through libmvec.
     */
    const float alpha = 0.7978845608028654f;
    const float coeff = 0.044715f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float v = x[i];
        const float v3 = v * v * v;
        const float inner = alpha * (v + coeff * v3);

        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void relu_f(float *VITS_RESTRICT x, int n)
{
    int i = 0;

#ifdef __AVX2__

    const __m256 zero = _mm256_setzero_ps();

    for (; i + 7 < n; i += 8) {

        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(x + i, v);
    }

#endif

    for (; i < n; i++) {
        if (x[i] < 0.0f)
            x[i] = 0.0f;
    }
}

void leaky_relu_f(float *VITS_RESTRICT x, int n, float slope)
{
    int i = 0;

#ifdef __AVX2__

    const __m256 zero = _mm256_setzero_ps();
    const __m256 sv = _mm256_set1_ps(slope);

    for (; i + 7 < n; i += 8) {

        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 neg = _mm256_mul_ps(v, sv);
        const __m256 mask = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
        const __m256 r = _mm256_blendv_ps(v, neg, mask);

        _mm256_storeu_ps(x + i, r);
    }

#endif

    for (; i < n; i++) {
        if (x[i] < 0.0f)
            x[i] *= slope;
    }
}

void tanh_f(float *x, int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++)
        x[i] = tanhf(x[i]);
}

void sigmoid_f(float *x, int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float v = x[i];

        /*
         * Numerically stable sigmoid.
         */
        if (v >= 0.0f) {
            const float z = expf(-v);
            x[i] = 1.0f / (1.0f + z);
        } else {
            const float z = expf(v);
            x[i] = z / (1.0f + z);
        }
    }
}

/* ============================================================
 * Softmax
 * ============================================================ */

void softmax_f(float *x, int n)
{
    if (n <= 0)
        return;

    float maxv = -FLT_MAX;
    int i = 0;

#ifdef __AVX2__

    __m256 vmax = _mm256_set1_ps(-FLT_MAX);

    for (; i + 7 < n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    maxv = hmax256_ps(vmax);

#endif

    for (; i < n; i++) {
        if (x[i] > maxv)
            maxv = x[i];
    }

    /*
     * expf remains scalar unless compiler uses vector libm.
     */
    for (i = 0; i < n; i++)
        x[i] = expf(x[i] - maxv);

    float sum = 0.0f;

#ifdef __AVX2__

    __m256 vsum = _mm256_setzero_ps();
    i = 0;

    for (; i + 7 < n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        vsum = _mm256_add_ps(vsum, v);
    }

    sum = hsum256_ps(vsum);

#else
    i = 0;
#endif

    for (; i < n; i++)
        sum += x[i];

    const float inv = 1.0f / sum;

    i = 0;

#ifdef __AVX2__

    const __m256 vinv = _mm256_set1_ps(inv);

    for (; i + 7 < n; i += 8) {

        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_mul_ps(v, vinv);
        _mm256_storeu_ps(x + i, v);
    }

#endif

    for (; i < n; i++)
        x[i] *= inv;
}

/* ============================================================
 * Gating
 * ============================================================ */

void gating_f(
    const float *VITS_RESTRICT lo,
    const float *VITS_RESTRICT hi,
    float *VITS_RESTRICT out,
    int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float h = hi[i];
        float sigmoid;

        if (h >= 0.0f) {
            const float z = expf(-h);
            sigmoid = 1.0f / (1.0f + z);
        } else {
            const float z = expf(h);
            sigmoid = z / (1.0f + z);
        }

        out[i] = tanhf(lo[i]) * sigmoid;
    }
}

/* ============================================================
 * Mask operations
 * ============================================================ */

void mask_zero_f(
    float *VITS_RESTRICT x,
    const int *VITS_RESTRICT mask,
    int C,
    int T)
{
    if (!mask)
        return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if((size_t)C*T >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < C; c++) {

        float *VITS_RESTRICT xc = x + (size_t)c * T;
        int t = 0;

#ifdef __AVX2__

        const __m256i zero_i = _mm256_setzero_si256();

        for (; t + 7 < T; t += 8) {

            const __m256i m = _mm256_loadu_si256((const __m256i *)(mask + t));
            const __m256i iszero = _mm256_cmpeq_epi32(m, zero_i);
            const __m256 v = _mm256_loadu_ps(xc + t);
            const __m256 r = _mm256_blendv_ps(v, _mm256_setzero_ps(),
                                              _mm256_castsi256_ps(iszero));

            _mm256_storeu_ps(xc + t, r);
        }

#endif

        for (; t < T; t++) {
            if (!mask[t])
                xc[t] = 0.0f;
        }
    }
}

void masked_axpy_f(
    float *VITS_RESTRICT dst,
    const float *VITS_RESTRICT src,
    const int *VITS_RESTRICT mask,
    int C,
    int T)
{
    const size_t n = (size_t)C * T;

    if (!mask) {
        axpy_f(dst, src, (int)n);
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < C; c++) {

        float *VITS_RESTRICT d = dst + (size_t)c * T;
        const float *VITS_RESTRICT s = src + (size_t)c * T;
        int t = 0;

#ifdef __AVX2__

        const __m256i zero_i = _mm256_setzero_si256();

        for (; t + 7 < T; t += 8) {

            const __m256 dv = _mm256_loadu_ps(d + t);
            const __m256 sv = _mm256_loadu_ps(s + t);
            const __m256 sum = _mm256_add_ps(dv, sv);
            const __m256i mv = _mm256_loadu_si256((const __m256i *)(mask + t));
            const __m256i iszero = _mm256_cmpeq_epi32(mv, zero_i);
            const __m256 r = _mm256_blendv_ps(sum, _mm256_setzero_ps(),
                                              _mm256_castsi256_ps(iszero));

            _mm256_storeu_ps(d + t, r);
        }

#endif

        for (; t < T; t++)
            d[t] = mask[t] ? d[t] + s[t] : 0.0f;
    }
}

/* ============================================================
 * Vector operations
 * ============================================================ */

void axpy_f(
    float *VITS_RESTRICT dst,
    const float *VITS_RESTRICT src,
    int n)
{
    int i = 0;

#ifdef __AVX2__

    for (; i + 7 < n; i += 8) {

        const __m256 a = _mm256_loadu_ps(dst + i);
        const __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }

#endif

    for (; i < n; i++)
        dst[i] += src[i];
}

void scal_f(
    float *VITS_RESTRICT x,
    int n,
    float alpha)
{
    int i = 0;

#ifdef __AVX2__

    const __m256 a = _mm256_set1_ps(alpha);

    for (; i + 7 < n; i += 8) {

        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_mul_ps(v, a);
        _mm256_storeu_ps(x + i, v);
    }

#endif

    for (; i < n; i++)
        x[i] *= alpha;
}

/* ============================================================
 * RNG
 * ============================================================ */

/*
 * Box-Muller transform.
 *
 * The generator itself is normally not a relevant inference
 * bottleneck, therefore the implementation is intentionally
 * kept simple.
 *
 * Do not call concurrently from multiple OpenMP threads.
 */
float randn_f(void)
{
    static int has_spare = 0;
    static float spare = 0.0f;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u1;
    float u2;

    do {
        u1 = (float)rand() / ((float)RAND_MAX + 1.0f);
    } while (u1 <= 0.0f);

    u2 = (float)rand() / ((float)RAND_MAX + 1.0f);

    const float r = sqrtf(-2.0f * logf(u1));
    const float angle = 2.0f * (float)M_PI * u2;

    spare = r * sinf(angle);
    has_spare = 1;

    return r * cosf(angle);
}
