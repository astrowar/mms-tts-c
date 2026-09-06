#include "vits.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VITS_HAS_NEON 1
#else
#define VITS_HAS_NEON 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef VITS_OMP_WORK_THRESHOLD
#define VITS_OMP_WORK_THRESHOLD 16384
#endif

#ifndef VITS_OMP_VECTOR_THRESHOLD
#define VITS_OMP_VECTOR_THRESHOLD 32768
#endif

#if defined(_MSC_VER)
#define VITS_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define VITS_RESTRICT __restrict__
#else
#define VITS_RESTRICT restrict
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VITS_LIKELY(x)   __builtin_expect(!!(x), 1)
#define VITS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define VITS_LIKELY(x)   (x)
#define VITS_UNLIKELY(x) (x)
#endif

static inline int imin_i(int a, int b) { return a < b ? a : b; }
static inline int imax_i(int a, int b) { return a > b ? a : b; }

/* Forward declarations used by masked_axpy_f. */
void axpy_f(float *VITS_RESTRICT dst,
            const float *VITS_RESTRICT src,
            int n);

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

static inline float hsumq_f32(float32x4_t v)
{
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
#endif
}

static inline float hmaxq_f32(float32x4_t v)
{
#if defined(__aarch64__)
    return vmaxvq_f32(v);
#else
    float32x2_t m = vmax_f32(vget_low_f32(v), vget_high_f32(v));
    m = vpmax_f32(m, m);
    return vget_lane_f32(m, 0);
#endif
}

/*
 * 1/sqrt(x) using NEON reciprocal-square-root estimate + two Newton steps.
 * This is much faster than four scalar sqrtf calls and is accurate enough
 * for inference LayerNorm on Cortex-A72.
 */
static inline float32x4_t vits_rsqrtq_f32(float32x4_t x)
{
    float32x4_t y = vrsqrteq_f32(x);
    y = vmulq_f32(y, vrsqrtsq_f32(vmulq_f32(y, y), x));
    y = vmulq_f32(y, vrsqrtsq_f32(vmulq_f32(y, y), x));
    return y;
}

static inline void fill_f32_neon(float *VITS_RESTRICT dst,
                                 int n,
                                 float value)
{
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
    for (; i < n; ++i)
        dst[i] = value;
}

#endif /* VITS_HAS_NEON */

/* ============================================================
 * Conv1D k=1 optimized micro-kernel
 *
 * Computes 4 output channels x 4 time positions at once.
 * The input vector is loaded once and reused by four outputs.
 * ============================================================ */

#if VITS_HAS_NEON
static void conv1d_k1_neon(const float *VITS_RESTRICT in,
                           int in_ch,
                           int T,
                           const Conv1d *VITS_RESTRICT c,
                           float *VITS_RESTRICT out)
{
    const int out_ch = c->out_ch;
    const int blocks = (out_ch + 3) >> 2;
    const size_t work = (size_t)out_ch * (size_t)in_ch * (size_t)T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int block = 0; block < blocks; ++block) {
        const int o0 = block << 2;
        const int nout = imin_i(4, out_ch - o0);
        int t = 0;

        for (; t + 3 < T; t += 4) {
            float32x4_t a0 = vdupq_n_f32(c->bias[o0]);
            float32x4_t a1 = vdupq_n_f32(nout > 1 ? c->bias[o0 + 1] : 0.0f);
            float32x4_t a2 = vdupq_n_f32(nout > 2 ? c->bias[o0 + 2] : 0.0f);
            float32x4_t a3 = vdupq_n_f32(nout > 3 ? c->bias[o0 + 3] : 0.0f);

            for (int i = 0; i < in_ch; ++i) {
                const float32x4_t x = vld1q_f32(in + (size_t)i * T + t);

                a0 = vits_mlaq_n_f32(a0, x,
                    c->weight[(size_t)o0 * in_ch + i]);

                if (nout > 1)
                    a1 = vits_mlaq_n_f32(a1, x,
                        c->weight[(size_t)(o0 + 1) * in_ch + i]);
                if (nout > 2)
                    a2 = vits_mlaq_n_f32(a2, x,
                        c->weight[(size_t)(o0 + 2) * in_ch + i]);
                if (nout > 3)
                    a3 = vits_mlaq_n_f32(a3, x,
                        c->weight[(size_t)(o0 + 3) * in_ch + i]);
            }

            vst1q_f32(out + (size_t)o0 * T + t, a0);
            if (nout > 1) vst1q_f32(out + (size_t)(o0 + 1) * T + t, a1);
            if (nout > 2) vst1q_f32(out + (size_t)(o0 + 2) * T + t, a2);
            if (nout > 3) vst1q_f32(out + (size_t)(o0 + 3) * T + t, a3);
        }

        for (; t < T; ++t) {
            for (int q = 0; q < nout; ++q) {
                const int o = o0 + q;
                const float *VITS_RESTRICT w = c->weight + (size_t)o * in_ch;
                float acc = c->bias[o];
                for (int i = 0; i < in_ch; ++i)
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

void conv1d(const float *VITS_RESTRICT in,
            int in_ch,
            int T,
            const Conv1d *VITS_RESTRICT c,
            float *VITS_RESTRICT out)
{
    const int out_ch = c->out_ch;
    const int k = c->k;
    const int pad = c->pad;
    const int dil = c->dilation;

    if (VITS_UNLIKELY(T <= 0 || in_ch <= 0 || out_ch <= 0))
        return;

#if VITS_HAS_NEON
    if (k == 1 && pad == 0) {
        conv1d_k1_neon(in, in_ch, T, c, out);
        return;
    }
#endif

    const int min_shift = -pad;
    const int max_shift = (k - 1) * dil - pad;
    const int interior_begin = imin_i(T, imax_i(0, -min_shift));
    const int interior_end   = imax_i(0, imin_i(T, T - max_shift));
    const size_t work = (size_t)out_ch * in_ch * k * T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; ++o) {
        float *VITS_RESTRICT out_o = out + (size_t)o * T;
        const float *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        /* If the kernel footprint is wider than T, there is no
         * bounds-check-free interior region. Process the full row
         * with the scalar border kernel to avoid overlap/OOB writes. */
        if (interior_begin >= interior_end) {
            for (int t = 0; t < T; ++t) {
                float acc = c->bias[o];
                for (int i = 0; i < in_ch; ++i) {
                    const float *VITS_RESTRICT in_i = in + (size_t)i * T;
                    const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;
                    for (int j = 0; j < k; ++j) {
                        const int idx = t + j * dil - pad;
                        if ((unsigned)idx < (unsigned)T)
                            acc += w_i[j] * in_i[idx];
                    }
                }
                out_o[t] = acc;
            }
            continue;
        }

        /* Left border. */
        for (int t = 0; t < interior_begin; ++t) {
            float acc = c->bias[o];
            for (int i = 0; i < in_ch; ++i) {
                const float *VITS_RESTRICT in_i = in + (size_t)i * T;
                const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;
                for (int j = 0; j < k; ++j) {
                    const int idx = t + j * dil - pad;
                    if ((unsigned)idx < (unsigned)T)
                        acc += w_i[j] * in_i[idx];
                }
            }
            out_o[t] = acc;
        }

        int t = interior_begin;

#if VITS_HAS_NEON
        /* Main NEON path: four time positions per vector. */
        for (; t + 3 < interior_end; t += 4) {
            float32x4_t acc = vdupq_n_f32(c->bias[o]);

            for (int i = 0; i < in_ch; ++i) {
                const float *VITS_RESTRICT in_i = in + (size_t)i * T;
                const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;

                for (int j = 0; j < k; ++j) {
                    const float weight = w_i[j];
                    if (weight == 0.0f)
                        continue;
                    const int shift = j * dil - pad;
                    const float32x4_t x = vld1q_f32(in_i + t + shift);
                    acc = vits_mlaq_n_f32(acc, x, weight);
                }
            }
            vst1q_f32(out_o + t, acc);
        }
#endif

        for (; t < interior_end; ++t) {
            float acc = c->bias[o];
            for (int i = 0; i < in_ch; ++i) {
                const float *VITS_RESTRICT in_i = in + (size_t)i * T;
                const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;
                for (int j = 0; j < k; ++j)
                    acc += w_i[j] * in_i[t + j * dil - pad];
            }
            out_o[t] = acc;
        }

        /* Right border. */
        for (t = interior_end; t < T; ++t) {
            float acc = c->bias[o];
            for (int i = 0; i < in_ch; ++i) {
                const float *VITS_RESTRICT in_i = in + (size_t)i * T;
                const float *VITS_RESTRICT w_i = w_o + (size_t)i * k;
                for (int j = 0; j < k; ++j) {
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

void conv1d_depthwise(const float *VITS_RESTRICT in,
                      int ch,
                      int T,
                      const float *VITS_RESTRICT w,
                      const float *VITS_RESTRICT b,
                      int k,
                      int dilation,
                      int pad,
                      float *VITS_RESTRICT out)
{
    if (VITS_UNLIKELY(T <= 0 || ch <= 0))
        return;

    const int min_shift = -pad;
    const int max_shift = (k - 1) * dilation - pad;
    const int interior_begin = imin_i(T, imax_i(0, -min_shift));
    const int interior_end   = imax_i(0, imin_i(T, T - max_shift));
    const size_t work = (size_t)ch * k * T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int cidx = 0; cidx < ch; ++cidx) {
        const float *VITS_RESTRICT in_c = in + (size_t)cidx * T;
        const float *VITS_RESTRICT w_c  = w  + (size_t)cidx * k;
        float *VITS_RESTRICT out_c = out + (size_t)cidx * T;

        if (interior_begin >= interior_end) {
            for (int t = 0; t < T; ++t) {
                float acc = b[cidx];
                for (int j = 0; j < k; ++j) {
                    const int idx = t + j * dilation - pad;
                    if ((unsigned)idx < (unsigned)T)
                        acc += w_c[j] * in_c[idx];
                }
                out_c[t] = acc;
            }
            continue;
        }

        for (int t = 0; t < interior_begin; ++t) {
            float acc = b[cidx];
            for (int j = 0; j < k; ++j) {
                const int idx = t + j * dilation - pad;
                if ((unsigned)idx < (unsigned)T)
                    acc += w_c[j] * in_c[idx];
            }
            out_c[t] = acc;
        }

        int t = interior_begin;

#if VITS_HAS_NEON
        for (; t + 3 < interior_end; t += 4) {
            float32x4_t acc = vdupq_n_f32(b[cidx]);
            for (int j = 0; j < k; ++j) {
                const float weight = w_c[j];
                if (weight == 0.0f)
                    continue;
                const int shift = j * dilation - pad;
                const float32x4_t x = vld1q_f32(in_c + t + shift);
                acc = vits_mlaq_n_f32(acc, x, weight);
            }
            vst1q_f32(out_c + t, acc);
        }
#endif

        for (; t < interior_end; ++t) {
            float acc = b[cidx];
            for (int j = 0; j < k; ++j)
                acc += w_c[j] * in_c[t + j * dilation - pad];
            out_c[t] = acc;
        }

        for (t = interior_end; t < T; ++t) {
            float acc = b[cidx];
            for (int j = 0; j < k; ++j) {
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
 * Vectorizes the kernel dimension. For one input sample, four
 * contiguous weights update four contiguous output positions.
 * ============================================================ */

void conv_transpose1d(const float *VITS_RESTRICT in,
                      int in_ch,
                      int T,
                      const ConvTranspose1d *VITS_RESTRICT c,
                      float *VITS_RESTRICT out,
                      int *out_T)
{
    const int out_ch = c->out_ch;
    const int k = c->k;
    const int stride = c->stride;
    const int pad = c->pad;

    if (VITS_UNLIKELY(T <= 0 || in_ch <= 0 || out_ch <= 0)) {
        *out_T = 0;
        return;
    }

    const int oT = (T - 1) * stride - 2 * pad + k;
    *out_T = oT;
    if (oT <= 0)
        return;

    const size_t work = (size_t)out_ch * in_ch * T * k;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int o = 0; o < out_ch; ++o) {
        float *VITS_RESTRICT out_o = out + (size_t)o * oT;

#if VITS_HAS_NEON
        fill_f32_neon(out_o, oT, c->bias[o]);
#else
        for (int p = 0; p < oT; ++p)
            out_o[p] = c->bias[o];
#endif

        const float *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        for (int i = 0; i < in_ch; ++i) {
            const float *VITS_RESTRICT in_i = in + (size_t)i * T;
            const float *VITS_RESTRICT w_i  = w_o + (size_t)i * k;

            for (int t = 0; t < T; ++t) {
                const float x = in_i[t];
                if (x == 0.0f)
                    continue;

                const int base = t * stride - pad;
                const int j0 = imax_i(0, -base);
                const int j1 = imin_i(k, oT - base);
                if (j0 >= j1)
                    continue;

                int j = j0;

#if VITS_HAS_NEON
                for (; j + 3 < j1; j += 4) {
                    float32x4_t ov = vld1q_f32(out_o + base + j);
                    const float32x4_t wv = vld1q_f32(w_i + j);
                    ov = vits_mlaq_n_f32(ov, wv, x);
                    vst1q_f32(out_o + base + j, ov);
                }
#endif

                for (; j < j1; ++j)
                    out_o[base + j] += x * w_i[j];
            }
        }
    }
}

/* ============================================================
 * LayerNorm [dim][T]
 *
 * The tensor is channel-first, therefore the efficient SIMD
 * direction is time: four independent time positions per vector.
 * ============================================================ */

void layer_norm(const float *VITS_RESTRICT in,
                int T,
                int dim,
                const LayerNorm *VITS_RESTRICT ln,
                float *VITS_RESTRICT out)
{
    if (VITS_UNLIKELY(T <= 0 || dim <= 0))
        return;

#if VITS_HAS_NEON
    const int blocks = T >> 2;
    const size_t work = (size_t)T * dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int block = 0; block < blocks; ++block) {
        const int t = block << 2;
        float32x4_t mean = vdupq_n_f32(0.0f);

        for (int i = 0; i < dim; ++i)
            mean = vaddq_f32(mean, vld1q_f32(in + (size_t)i * T + t));

        mean = vmulq_n_f32(mean, 1.0f / (float)dim);

        float32x4_t var = vdupq_n_f32(0.0f);
        for (int i = 0; i < dim; ++i) {
            const float32x4_t x = vld1q_f32(in + (size_t)i * T + t);
            const float32x4_t d = vsubq_f32(x, mean);
            var = vits_mlaq_f32(var, d, d);
        }

        var = vmulq_n_f32(var, 1.0f / (float)dim);
        var = vaddq_f32(var, vdupq_n_f32(LN_EPS));
        const float32x4_t inv = vits_rsqrtq_f32(var);

        for (int i = 0; i < dim; ++i) {
            float32x4_t y = vld1q_f32(in + (size_t)i * T + t);
            y = vsubq_f32(y, mean);
            y = vmulq_f32(y, inv);
            y = vmulq_n_f32(y, ln->weight[i]);
            y = vaddq_f32(y, vdupq_n_f32(ln->bias[i]));
            vst1q_f32(out + (size_t)i * T + t, y);
        }
    }

    /* Scalar tail: at most three time positions. */
    for (int t = blocks << 2; t < T; ++t) {
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i)
            mean += in[(size_t)i * T + t];
        mean /= (float)dim;

        float var = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float d = in[(size_t)i * T + t] - mean;
            var += d * d;
        }
        var /= (float)dim;

        const float inv = 1.0f / sqrtf(var + LN_EPS);
        for (int i = 0; i < dim; ++i) {
            const size_t idx = (size_t)i * T + t;
            out[idx] = ln->weight[i] * (in[idx] - mean) * inv + ln->bias[i];
        }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < T; ++t) {
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i)
            mean += in[(size_t)i * T + t];
        mean /= (float)dim;

        float var = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float d = in[(size_t)i * T + t] - mean;
            var += d * d;
        }
        var /= (float)dim;

        const float inv = 1.0f / sqrtf(var + LN_EPS);
        for (int i = 0; i < dim; ++i) {
            const size_t idx = (size_t)i * T + t;
            out[idx] = ln->weight[i] * (in[idx] - mean) * inv + ln->bias[i];
        }
    }
#endif
}

/* ============================================================
 * Activations
 * ============================================================ */

void relu_f(float *VITS_RESTRICT x, int n)
{
    int i = 0;
#if VITS_HAS_NEON
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 15 < n; i += 16) {
        float32x4_t a = vmaxq_f32(vld1q_f32(x + i + 0), zero);
        float32x4_t b = vmaxq_f32(vld1q_f32(x + i + 4), zero);
        float32x4_t c = vmaxq_f32(vld1q_f32(x + i + 8), zero);
        float32x4_t d = vmaxq_f32(vld1q_f32(x + i + 12), zero);
        vst1q_f32(x + i + 0, a);
        vst1q_f32(x + i + 4, b);
        vst1q_f32(x + i + 8, c);
        vst1q_f32(x + i + 12, d);
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vmaxq_f32(vld1q_f32(x + i), zero);
        vst1q_f32(x + i, v);
    }
#endif
    for (; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

void leaky_relu_f(float *VITS_RESTRICT x, int n, float slope)
{
    int i = 0;
#if VITS_HAS_NEON
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        const uint32x4_t negmask = vcltq_f32(v, zero);
        const float32x4_t neg = vmulq_n_f32(v, slope);
        vst1q_f32(x + i, vbslq_f32(negmask, neg, v));
    }
#endif
    for (; i < n; ++i)
        if (x[i] < 0.0f) x[i] *= slope;
}

/*
 * ARM NEON does not provide native vector exp/tanh instructions.
 * Keep libm semantics here; with -O3 -ffast-math the compiler is free
 * to use suitable optimized math implementations when available.
 */
void gelu(float *x, int n)
{
    const float alpha = 0.7978845608028654f;
    const float coeff = 0.044715f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        const float inner = alpha * (v + coeff * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void tanh_f(float *x, int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; ++i)
        x[i] = tanhf(x[i]);
}

void sigmoid_f(float *x, int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (v >= 0.0f) {
            const float z = expf(-v);
            x[i] = 1.0f / (1.0f + z);
        } else {
            const float z = expf(v);
            x[i] = z / (1.0f + z);
        }
    }
}

void gating_f(const float *VITS_RESTRICT lo,
              const float *VITS_RESTRICT hi,
              float *VITS_RESTRICT out,
              int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; ++i) {
        const float h = hi[i];
        float sig;
        if (h >= 0.0f) {
            const float z = expf(-h);
            sig = 1.0f / (1.0f + z);
        } else {
            const float z = expf(h);
            sig = z / (1.0f + z);
        }
        out[i] = tanhf(lo[i]) * sig;
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

#if VITS_HAS_NEON
    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
    for (; i + 3 < n; i += 4)
        vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
    maxv = hmaxq_f32(vmax);
#endif

    for (; i < n; ++i)
        if (x[i] > maxv) maxv = x[i];

    for (i = 0; i < n; ++i)
        x[i] = expf(x[i] - maxv);

    float sum = 0.0f;
    i = 0;
#if VITS_HAS_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4)
        vsum = vaddq_f32(vsum, vld1q_f32(x + i));
    sum = hsumq_f32(vsum);
#endif
    for (; i < n; ++i)
        sum += x[i];

    const float inv = 1.0f / sum;
    i = 0;
#if VITS_HAS_NEON
    for (; i + 3 < n; i += 4) {
        const float32x4_t v = vmulq_n_f32(vld1q_f32(x + i), inv);
        vst1q_f32(x + i, v);
    }
#endif
    for (; i < n; ++i)
        x[i] *= inv;
}

/* ============================================================
 * Masked operations [C][T]
 * ============================================================ */

void mask_zero_f(float *VITS_RESTRICT x,
                 const int *VITS_RESTRICT mask,
                 int C,
                 int T)
{
    if (!mask)
        return;

    const size_t work = (size_t)C * T;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < C; ++c) {
        float *VITS_RESTRICT xc = x + (size_t)c * T;
        int t = 0;
#if VITS_HAS_NEON
        const int32x4_t zero_i = vdupq_n_s32(0);
        const float32x4_t zero_f = vdupq_n_f32(0.0f);
        for (; t + 3 < T; t += 4) {
            const int32x4_t m = vld1q_s32((const int32_t *)(mask + t));
            const uint32x4_t iszero = vceqq_s32(m, zero_i);
            const float32x4_t v = vld1q_f32(xc + t);
            vst1q_f32(xc + t, vbslq_f32(iszero, zero_f, v));
        }
#endif
        for (; t < T; ++t)
            if (!mask[t]) xc[t] = 0.0f;
    }
}

void masked_axpy_f(float *VITS_RESTRICT dst,
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
#pragma omp parallel for schedule(static) if(n >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < C; ++c) {
        float *VITS_RESTRICT d = dst + (size_t)c * T;
        const float *VITS_RESTRICT s = src + (size_t)c * T;
        int t = 0;
#if VITS_HAS_NEON
        const int32x4_t zero_i = vdupq_n_s32(0);
        const float32x4_t zero_f = vdupq_n_f32(0.0f);
        for (; t + 3 < T; t += 4) {
            const float32x4_t dv = vld1q_f32(d + t);
            const float32x4_t sv = vld1q_f32(s + t);
            const float32x4_t sum = vaddq_f32(dv, sv);
            const int32x4_t mv = vld1q_s32((const int32_t *)(mask + t));
            const uint32x4_t iszero = vceqq_s32(mv, zero_i);
            vst1q_f32(d + t, vbslq_f32(iszero, zero_f, sum));
        }
#endif
        for (; t < T; ++t)
            d[t] = mask[t] ? d[t] + s[t] : 0.0f;
    }
}

/* ============================================================
 * Element-wise vector operations
 * ============================================================ */

void axpy_f(float *VITS_RESTRICT dst,
            const float *VITS_RESTRICT src,
            int n)
{
    int i = 0;
#if VITS_HAS_NEON
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vaddq_f32(vld1q_f32(dst + i + 0),  vld1q_f32(src + i + 0));
        float32x4_t a1 = vaddq_f32(vld1q_f32(dst + i + 4),  vld1q_f32(src + i + 4));
        float32x4_t a2 = vaddq_f32(vld1q_f32(dst + i + 8),  vld1q_f32(src + i + 8));
        float32x4_t a3 = vaddq_f32(vld1q_f32(dst + i + 12), vld1q_f32(src + i + 12));
        vst1q_f32(dst + i + 0,  a0);
        vst1q_f32(dst + i + 4,  a1);
        vst1q_f32(dst + i + 8,  a2);
        vst1q_f32(dst + i + 12, a3);
    }
    for (; i + 3 < n; i += 4) {
        const float32x4_t v = vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i));
        vst1q_f32(dst + i, v);
    }
#endif
    for (; i < n; ++i)
        dst[i] += src[i];
}

void scal_f(float *VITS_RESTRICT x, int n, float alpha)
{
    int i = 0;
#if VITS_HAS_NEON
    for (; i + 15 < n; i += 16) {
        vst1q_f32(x + i + 0,  vmulq_n_f32(vld1q_f32(x + i + 0),  alpha));
        vst1q_f32(x + i + 4,  vmulq_n_f32(vld1q_f32(x + i + 4),  alpha));
        vst1q_f32(x + i + 8,  vmulq_n_f32(vld1q_f32(x + i + 8),  alpha));
        vst1q_f32(x + i + 12, vmulq_n_f32(vld1q_f32(x + i + 12), alpha));
    }
    for (; i + 3 < n; i += 4) {
        const float32x4_t v = vmulq_n_f32(vld1q_f32(x + i), alpha);
        vst1q_f32(x + i, v);
    }
#endif
    for (; i < n; ++i)
        x[i] *= alpha;
}

/* ============================================================
 * RNG
 * ============================================================ */

float randn_f(void)
{
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    static _Thread_local int has_spare = 0;
    static _Thread_local float spare = 0.0f;
#else
    static int has_spare = 0;
    static float spare = 0.0f;
#endif

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u1, u2;
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
