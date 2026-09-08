#include "vits.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
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
 * Quantize FP32 → Q16 (INT16 with power-of-2 scale) — AVX2
 * ============================================================ */
void quantize_f32_to_q16(const float *VITS_RESTRICT in, int n,
                         int exponent, int16_t *VITS_RESTRICT out)
{
    const float scale = (float)ldexp(1.0, (double)(-exponent));
    const __m256 vscale = _mm256_set1_ps(scale);

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 xq = _mm256_mul_ps(x, vscale);
        __m256i x32 = _mm256_cvtps_epi32(xq);

        /* Pack 8 × INT32 → 8 × INT16 (saturating) */
        __m128i lo4 = _mm256_castsi256_si128(x32);
        __m128i hi4 = _mm256_extracti128_si256(x32, 1);
        __m128i x16 = _mm_packs_epi32(lo4, hi4);

        _mm_storeu_si128((__m128i *)(out + i), x16);
    }
    for (; i < n; i++) {
        float v = in[i] * scale;
        int32_t q = (int32_t)lroundf(v);
        out[i] = saturate_i16(q);
    }
}

/* ============================================================
 * Conv1D: INT16 input × INT8 weights → INT32 acc → INT16 output — AVX2
 *
 * Inner t-loop vectorized 8-wide using INT32 SIMD accumulators.
 * Weights are scalar per (i,j) pair, broadcast to INT32 lanes.
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

        int32_t *VITS_RESTRICT acc =
            (int32_t *)malloc(sizeof(int32_t) * (size_t)T);
        if (!acc) return;

        memset(acc, 0, sizeof(int32_t) * (size_t)T);

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

                const __m128i vw = _mm_set1_epi32(wv);

                /* Process 8 t-positions at a time */
                int t = t0;
                for (; t + 7 < t1; t += 8) {
                    /* Load 8 × INT16 activations */
                    __m128i x =
                        _mm_loadu_si128((const __m128i *)(in_i + t + shift_t));

                    /* Expand to 8 × INT32 */
                    __m128i x_lo = _mm_cvtepi16_epi32(x);
                    __m128i x_hi =
                        _mm_cvtepi16_epi32(_mm_srli_si128(x, 8));

                    /* Load accumulators, multiply-add, store */
                    __m128i acc_lo =
                        _mm_loadu_si128((const __m128i *)(acc + t));
                    __m128i acc_hi =
                        _mm_loadu_si128((const __m128i *)(acc + t + 4));

                    acc_lo = _mm_add_epi32(acc_lo, _mm_mullo_epi32(x_lo, vw));
                    acc_hi = _mm_add_epi32(acc_hi, _mm_mullo_epi32(x_hi, vw));

                    _mm_storeu_si128((__m128i *)(acc + t), acc_lo);
                    _mm_storeu_si128((__m128i *)(acc + t + 4), acc_hi);
                }

                /* Scalar tail */
                for (; t < t1; t++) {
                    acc[t] += (int32_t)in_i[t + shift_t] * wv;
                }
            }
        }

        /* Requantize: (acc * alpha + 2^(A-1)) >> A + bias, saturate */
        for (int t = 0; t < T; t++) {
            int64_t scaled = (int64_t)acc[t] * alpha_o;
            int32_t result = (int32_t)((scaled + (1 << shift)) >> Q16_ALPHA_BITS);
            result += bias_o;
            out_o[t] = saturate_i16(result);
        }

        free(acc);
    }
}

/* ============================================================
 * ConvTranspose1D: INT16 input × INT8 weights → INT32 → INT16 — AVX2
 *
 * Scatter pattern (t*stride - pad + j) prevents wide SIMD on t.
 * Uses 4-way unroll for the inner t-loop.
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

        int32_t *VITS_RESTRICT acc =
            (int32_t *)malloc(sizeof(int32_t) * (size_t)oT);
        if (!acc) return;

        memset(acc, 0, sizeof(int32_t) * (size_t)oT);

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

                /* 4-way unrolled scatter */
                int t = t0;
                for (; t + 3 < t1; t += 4) {
                    acc[t * stride - pad + j]     += (int32_t)in_i[t]     * wv;
                    acc[(t+1) * stride - pad + j] += (int32_t)in_i[t+1] * wv;
                    acc[(t+2) * stride - pad + j] += (int32_t)in_i[t+2] * wv;
                    acc[(t+3) * stride - pad + j] += (int32_t)in_i[t+3] * wv;
                }
                for (; t < t1; t++) {
                    acc[t * stride - pad + j] += (int32_t)in_i[t] * wv;
                }
            }
        }

        for (int t = 0; t < oT; t++) {
            int64_t scaled = (int64_t)acc[t] * alpha_o;
            int32_t result = (int32_t)((scaled + (1 << shift)) >> Q16_ALPHA_BITS);
            result += bias_o;
            out_o[t] = saturate_i16(result);
        }

        free(acc);
    }
}

/* ============================================================
 * Integer LeakyReLU — AVX2
 *
 *   x >= 0:  y = x
 *   x < 0:   y = -( (-x) * num + den/2) / den   (rounded)
 *
 * SIMD path handles den that is a power of 2 (shift-based divide).
 * For den=128: 16-wide AVX2. For other pow2 den: 8-wide SSE.
 * Non-pow2 den falls back to scalar.
 * ============================================================ */
void leaky_relu_q16(int16_t *VITS_RESTRICT x, int n,
                    int slope_num, int slope_den)
{
    if (slope_num == slope_den)
        return;

    const int32_t num  = slope_num;
    const int32_t half = slope_den / 2;

    /* Determine if den is a power of 2 → can use shift */
    int shift_amt = -1;
    if ((slope_den & (slope_den - 1)) == 0) {
        for (int s = 0; s < 31; s++)
            if ((1 << s) == slope_den) { shift_amt = s; break; }
    }

    if (shift_amt < 0) {
        /* Non-power-of-2 denominator: scalar fallback */
        for (int i = 0; i < n; i++) {
            int32_t v = (int32_t)x[i];
            if (v < 0) {
                int32_t neg = -v;
                int32_t result = (neg * num + half) / slope_den;
                v = -result;
            }
            x[i] = (int16_t)v;
        }
        return;
    }

    /* ---- 8-wide path: expand to INT32 to avoid overflow in v*num ---- */
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v8 = _mm_loadu_si128((const __m128i *)(x + i));

        /* Split into two groups of 4, expand to INT32 */
        __m128i v_lo = _mm_cvtepi16_epi32(v8);
        __m128i v_hi = _mm_cvtepi16_epi32(_mm_srli_si128(v8, 8));

        const __m128i zero = _mm_setzero_si128();
        const __m128i vnum = _mm_set1_epi32(num);
        const __m128i vhalf = _mm_set1_epi32(half);

        /* Process each 4-lane group */
        __m128i mask_lo = _mm_cmpgt_epi32(zero, v_lo);
        __m128i mask_hi = _mm_cmpgt_epi32(zero, v_hi);

        /* neg_v = -v, masked to 0 where v >= 0 */
        __m128i neg_lo = _mm_and_si128(mask_lo, _mm_sub_epi32(zero, v_lo));
        __m128i neg_hi = _mm_and_si128(mask_hi, _mm_sub_epi32(zero, v_hi));

        /* scaled = neg * num + half, then shift (divide by power-of-2 den) */
        __m128i div_lo =
            _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(neg_lo, vnum), vhalf),
                           shift_amt);
        __m128i div_hi =
            _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(neg_hi, vnum), vhalf),
                           shift_amt);

        /* neg_result = -div (the actual negative-branch output) */
        __m128i neg_res_lo = _mm_sub_epi32(zero, div_lo);
        __m128i neg_res_hi = _mm_sub_epi32(zero, div_hi);

        /* Select in INT32: v>=0 → v, v<0 → neg_result */
        __m128i res_lo = _mm_blendv_epi8(v_lo, neg_res_lo, mask_lo);
        __m128i res_hi = _mm_blendv_epi8(v_hi, neg_res_hi, mask_hi);

        /* Pack 8 × INT32 → 8 × INT16 (saturating) and store */
        __m128i res16 = _mm_packs_epi32(res_lo, res_hi);
        _mm_storeu_si128((__m128i *)(x + i), res16);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int32_t v = (int32_t)x[i];
        if (v < 0) {
            int32_t neg = -v;
            int32_t result = (neg * num + half) >> shift_amt;
            v = -result;
        }
        x[i] = (int16_t)v;
    }
}

/* ============================================================
 * Residual add: dst += src (both INT16, saturating) — AVX2
 * ============================================================ */
void residual_add_q16(int16_t *VITS_RESTRICT dst,
                      const int16_t *VITS_RESTRICT src, int n)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i *)(dst + i));
        __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i r = _mm256_adds_epi16(d, s);
        _mm256_storeu_si256((__m256i *)(dst + i), r);
    }
    for (; i < n; i++) {
        int32_t sum = (int32_t)dst[i] + (int32_t)src[i];
        dst[i] = saturate_i16(sum);
    }
}

/* ============================================================
 * MRF divide-by-3: x = x / 3 (integer, rounded) — AVX2
 *
 *   1/3 ≈ 21845 / 65536
 * ============================================================ */
void mrf_div3_q16(int16_t *VITS_RESTRICT x, int n)
{
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i v16 = _mm256_loadu_si256((const __m256i *)(x + i));

        /* Expand 16 × INT16 → 16 × INT32 (4 per 128-bit quarter) */
        __m128i lo8  = _mm256_castsi256_si128(v16);
        __m128i hi8  = _mm256_extracti128_si256(v16, 1);

        __m128i q0 = _mm_cvtepi16_epi32(lo8);
        __m128i q1 = _mm_cvtepi16_epi32(_mm_srli_si128(lo8, 8));
        __m128i q2 = _mm_cvtepi16_epi32(hi8);
        __m128i q3 = _mm_cvtepi16_epi32(_mm_srli_si128(hi8, 8));

        /* Multiply by magic (1/3 in Q16), add rounding, shift right */
        const __m128i vmul = _mm_set1_epi32(21845);
        const __m128i vrnd = _mm_set1_epi32(32768);

        q0 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(q0, vmul), vrnd), 16);
        q1 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(q1, vmul), vrnd), 16);
        q2 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(q2, vmul), vrnd), 16);
        q3 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(q3, vmul), vrnd), 16);

        /* Pack 16 × INT32 → 16 × INT16 (saturating signed) */
        __m128i res_lo = _mm_packs_epi32(q0, q1);  /* 8 × INT16 */
        __m128i res_hi = _mm_packs_epi32(q2, q3);  /* 8 × INT16 */

        __m256i result = _mm256_inserti128_si256(
            _mm256_castsi128_si256(res_lo), res_hi, 1);

        _mm256_storeu_si256((__m256i *)(x + i), result);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int32_t v = (int32_t)x[i];
        int32_t result = (v * 21845 + 32768) >> 16;
        x[i] = (int16_t)result;
    }
}

/* ============================================================
 * Tanh LUT (scalar — LUT gather doesn't vectorize well)
 * ============================================================ */
void tanh_lut_init(int16_t lut[], int exp)
{
    /* LUT has 32768 entries (indices 0..32767) */
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
