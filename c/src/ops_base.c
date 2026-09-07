#include "vits.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * Optimization configuration
 * ============================================================ */

/*
 * Avoid creating OpenMP teams for tiny operations.
 * Tune these values according to the target CPU.
 */
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
 * Convolutions
 * ============================================================ */

/*
 * Layout:
 *
 * input  = [in_ch][T]
 * output = [out_ch][T]
 *
 * Original implementation:
 *
 *   out_ch -> T -> in_ch -> kernel
 *
 * This version:
 *
 *   out_ch -> in_ch -> kernel -> T
 *
 * The innermost loop therefore walks through memory linearly.
 * This is substantially better for cache and SIMD.
 */
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

        const float bias = c->bias ? c->bias[o] : 0.0f;

        /*
         * Initialize complete output row.
         */
        if (bias == 0.0f) {
            memset(out_o, 0, sizeof(float) * (size_t)T);
        } else {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < T; t++)
                out_o[t] = bias;
        }

        const float *VITS_RESTRICT w_o =
            c->weight + (size_t)o * in_ch * k;

        /*
         * Convolution:
         *
         * out[t] += w[j] * in[t + j*dilation - pad]
         *
         * Instead of checking bounds for every t,
         * calculate the valid temporal interval once
         * for each kernel position.
         */
        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i =
                in + (size_t)i * T;

            const float *VITS_RESTRICT w_i =
                w_o + (size_t)i * k;

            for (int j = 0; j < k; j++) {

                const float weight = w_i[j];

                /*
                 * Zero weights are common after pruning and
                 * this test is extremely cheap.
                 */
                if (weight == 0.0f)
                    continue;

                const int shift =
                    j * dil - pad;

                /*
                 * Need:
                 *
                 * 0 <= t + shift < T
                 */
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

#ifdef _OPENMP
#pragma omp simd
#endif
                for (int t = t0; t < t1; t++) {
                    out_o[t] +=
                        weight * in_i[t + shift];
                }
            }
        }
    }
}


/*
 * Depthwise Conv1D.
 *
 * Since every channel is independent we parallelize directly
 * by channel.
 */
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

    const size_t work =
        (size_t)ch *
        (size_t)k *
        (size_t)T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < ch; c++) {

        const float *VITS_RESTRICT in_c =
            in + (size_t)c * T;

        float *VITS_RESTRICT out_c =
            out + (size_t)c * T;

        const float *VITS_RESTRICT w_c =
            w + (size_t)c * k;

        const float bias = b[c];

        if (bias == 0.0f) {
            memset(out_c, 0, sizeof(float) * (size_t)T);
        } else {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < T; t++)
                out_c[t] = bias;
        }

        for (int j = 0; j < k; j++) {

            const float weight = w_c[j];

            if (weight == 0.0f)
                continue;

            const int shift =
                j * dilation - pad;

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

#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = t0; t < t1; t++) {
                out_c[t] +=
                    weight * in_c[t + shift];
            }
        }
    }
}


/*
 * Transposed convolution.
 *
 * Major change compared with the original:
 *
 *   - no malloc()
 *   - no temporary double buffer
 *   - output channel belongs entirely to one thread
 *
 * Therefore no atomic operations or synchronization are needed.
 */
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

    const int oT =
        (T - 1) * stride -
        2 * pad +
        k;

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

        const float bias =
            c->bias[o];

        /*
         * Initialize output channel.
         */
        if (bias == 0.0f) {
            memset(
                out_o,
                0,
                sizeof(float) * (size_t)oT);
        } else {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int t = 0; t < oT; t++)
                out_o[t] = bias;
        }

        /*
         * Weight layout:
         *
         * [out_ch][in_ch][kernel]
         */
        const float *VITS_RESTRICT w_o =
            c->weight +
            (size_t)o * in_ch * k;

        for (int i = 0; i < in_ch; i++) {

            const float *VITS_RESTRICT in_i =
                in + (size_t)i * T;

            const float *VITS_RESTRICT w_i =
                w_o + (size_t)i * k;

            /*
             * Loop kernel first.
             *
             * For a fixed j:
             *
             * pos = t*stride - pad + j
             */
            for (int j = 0; j < k; j++) {

                const float weight =
                    w_i[j];

                if (weight == 0.0f)
                    continue;

                /*
                 * Determine temporal range where:
                 *
                 * 0 <= t*stride - pad + j < oT
                 */
                int t0 = 0;
                int t1 = T;

                const int lower =
                    pad - j;

                if (lower > 0) {
                    t0 =
                        (lower + stride - 1) /
                        stride;
                }

                const int upper =
                    oT - 1 + pad - j;

                if (upper < 0)
                    continue;

                const int max_t =
                    upper / stride;

                t1 =
                    imin_i(t1, max_t + 1);

                t0 =
                    imax_i(t0, 0);

                if (t0 >= t1)
                    continue;

                for (int t = t0; t < t1; t++) {

                    const int pos =
                        t * stride -
                        pad +
                        j;

                    out_o[pos] +=
                        weight * in_i[t];
                }
            }
        }
    }
}


/* ============================================================
 * Normalization
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

    const float eps = LN_EPS;

    const size_t work =
        (size_t)T *
        (size_t)dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int t = 0; t < T; t++) {

        /*
         * FP32 accumulation is normally sufficient for
         * inference and substantially cheaper on ARM CPUs.
         */
        float mean = 0.0f;

        for (int i = 0; i < dim; i++)
            mean += in[(size_t)i * T + t];

        mean /= (float)dim;

        float var = 0.0f;

        for (int i = 0; i < dim; i++) {
            const float d =
                in[(size_t)i * T + t] -
                mean;

            var += d * d;
        }

        var /= (float)dim;

        const float inv =
            1.0f / sqrtf(var + eps);

        for (int i = 0; i < dim; i++) {

            const size_t idx =
                (size_t)i * T + t;

            out[idx] =
                ln->weight[i] *
                (in[idx] - mean) *
                inv +
                ln->bias[i];
        }
    }
}


/* ============================================================
 * Activations
 * ============================================================ */

void gelu(float *x, int n)
{
    /*
     * sqrt(2/pi)
     */
    const float alpha =
        0.7978845608028654f;

    const float coeff =
        0.044715f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float v = x[i];

        const float v3 =
            v * v * v;

        const float inner =
            alpha *
            (v + coeff * v3);

        x[i] =
            0.5f *
            v *
            (1.0f + tanhf(inner));
    }
}


void leaky_relu_f(
    float *VITS_RESTRICT x,
    int n,
    float slope)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float v = x[i];

        x[i] =
            v >= 0.0f
            ? v
            : v * slope;
    }
}


void tanh_f(float *VITS_RESTRICT x, int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++)
        x[i] = tanhf(x[i]);
}


void sigmoid_f(float *VITS_RESTRICT x, int n)
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

            const float z =
                expf(-v);

            x[i] =
                1.0f /
                (1.0f + z);

        } else {

            const float z =
                expf(v);

            x[i] =
                z /
                (1.0f + z);
        }
    }
}


void relu_f(
    float *VITS_RESTRICT x,
    int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float v = x[i];

        x[i] =
            v > 0.0f
            ? v
            : 0.0f;
    }
}


void softmax_f(
    float *VITS_RESTRICT x,
    int n)
{
    if (n <= 0)
        return;

    float maxv =
        -INFINITY;

#ifdef _OPENMP
#pragma omp parallel for reduction(max:maxv) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        if (x[i] > maxv)
            maxv = x[i];
    }

    float sum =
        0.0f;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++) {

        const float v =
            expf(x[i] - maxv);

        x[i] = v;

        sum += v;
    }

    const float inv_sum =
        1.0f / sum;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++)
        x[i] *= inv_sum;
}


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

        const float h =
            hi[i];

        float sigmoid;

        if (h >= 0.0f) {

            const float z =
                expf(-h);

            sigmoid =
                1.0f /
                (1.0f + z);

        } else {

            const float z =
                expf(h);

            sigmoid =
                z /
                (1.0f + z);
        }

        out[i] =
            tanhf(lo[i]) *
            sigmoid;
    }
}


/* ============================================================
 * Masked operations
 * channel-first [C][T]
 * ============================================================ */

void mask_zero_f(
    float *VITS_RESTRICT x,
    const int *VITS_RESTRICT mask,
    int C,
    int T)
{
    if (!mask)
        return;

    const size_t work =
        (size_t)C *
        (size_t)T;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(work >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < C; c++) {

        float *VITS_RESTRICT xc =
            x + (size_t)c * T;

#ifdef _OPENMP
#pragma omp simd
#endif
        for (int t = 0; t < T; t++) {

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
    const size_t n =
        (size_t)C *
        (size_t)T;

    if (!mask) {

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
        for (size_t i = 0; i < n; i++)
            dst[i] += src[i];

        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_WORK_THRESHOLD)
#endif
    for (int c = 0; c < C; c++) {

        float *VITS_RESTRICT dc =
            dst + (size_t)c * T;

        const float *VITS_RESTRICT sc =
            src + (size_t)c * T;

#ifdef _OPENMP
#pragma omp simd
#endif
        for (int t = 0; t < T; t++) {

            dc[t] =
                mask[t]
                ? dc[t] + sc[t]
                : 0.0f;
        }
    }
}


/* ============================================================
 * Element-wise vector operations
 * ============================================================ */

void axpy_f(
    float *VITS_RESTRICT dst,
    const float *VITS_RESTRICT src,
    int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++)
        dst[i] += src[i];
}


void scal_f(
    float *VITS_RESTRICT x,
    int n,
    float alpha)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if(n >= VITS_OMP_VECTOR_THRESHOLD)
#endif
    for (int i = 0; i < n; i++)
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

        u1 =
            (float)rand() /
            ((float)RAND_MAX + 1.0f);

    } while (u1 <= 0.0f);

    u2 =
        (float)rand() /
        ((float)RAND_MAX + 1.0f);

    const float r =
        sqrtf(
            -2.0f *
            logf(u1));

    const float angle =
        2.0f *
        (float)M_PI *
        u2;

    spare =
        r *
        sinf(angle);

    has_spare = 1;

    return
        r *
        cosf(angle);
}