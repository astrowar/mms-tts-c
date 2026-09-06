#include "vits.h"
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void conv1d(const float *in, int in_ch, int T,
            const Conv1d *c, float *out)
{
    int out_ch = c->out_ch;
    int k = c->k;
    int pad = c->pad;
    int dil = c->dilation;

    for (int o = 0; o < out_ch; o++) {
        const float *w = c->weight + o * in_ch * k;
        double bias = c->bias[o];
        for (int t = 0; t < T; t++) {
            double acc = bias;
            for (int i = 0; i < in_ch; i++) {
                const float *in_c = in + (size_t)i * T;
                for (int j = 0; j < k; j++) {
                    int idx = t + j * dil - pad;
                    if (idx >= 0 && idx < T)
                        acc += (double)w[i * k + j] * (double)in_c[idx];
                }
            }
            out[(size_t)o * T + t] = (float)acc;
        }
    }
}

void conv1d_depthwise(const float *in, int ch, int T,
                      const float *w, const float *b,
                      int k, int dilation, int pad,
                      float *out)
{
    for (int c = 0; c < ch; c++) {
        const float *in_c = in + (size_t)c * T;
        const float *w_c = w + (size_t)c * k;
        double bias = b[c];
        for (int t = 0; t < T; t++) {
            double acc = bias;
            for (int j = 0; j < k; j++) {
                int idx = t + j * dilation - pad;
                if (idx >= 0 && idx < T)
                    acc += (double)w_c[j] * (double)in_c[idx];
            }
            out[(size_t)c * T + t] = (float)acc;
        }
    }
}

void conv_transpose1d(const float *in, int in_ch, int T,
                      const ConvTranspose1d *c, float *out, int *out_T)
{
    int oc = c->out_ch;
    int k = c->k;
    int stride = c->stride;
    int pad = c->pad;
    int oT = (T - 1) * stride - 2 * pad + k;
    *out_T = oT;

    double *acc = (double *)malloc(sizeof(double) * (size_t)oc * oT);
    if (!acc) return;

    for (int o = 0; o < oc; o++)
        for (int t = 0; t < oT; t++)
            acc[(size_t)o * oT + t] = c->bias[o];

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < in_ch; i++) {
            float v = in[(size_t)i * T + t];
            if (v == 0.0f) continue;
            for (int o = 0; o < oc; o++) {
                const float *w = c->weight + ((size_t)o * in_ch + i) * k;
                for (int j = 0; j < k; j++) {
                    int pos = t * stride - pad + j;
                    if (pos >= 0 && pos < oT)
                        acc[(size_t)o * oT + pos] += (double)v * (double)w[j];
                }
            }
        }
    }

    for (int o = 0; o < oc; o++)
        for (int t = 0; t < oT; t++)
            out[(size_t)o * oT + t] = (float)acc[(size_t)o * oT + t];

    free(acc);
}

void layer_norm(const float *in, int T, int dim,
                const LayerNorm *ln, float *out)
{
    float eps = LN_EPS;

    for (int t = 0; t < T; t++) {
        double mean = 0.0;
        for (int i = 0; i < dim; i++)
            mean += (double)in[(size_t)i * T + t];
        mean /= (double)dim;

        double var = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = (double)in[(size_t)i * T + t] - mean;
            var += d * d;
        }
        var /= (double)dim;

        double inv = 1.0 / sqrt(var + eps);
        for (int i = 0; i < dim; i++)
            out[(size_t)i * T + t] =
                (float)((double)ln->weight[i] * ((double)in[(size_t)i * T + t] - mean) * inv
                + (double)ln->bias[i]);
    }
}

void gelu(float *x, int n)
{
    const float kAlpha = sqrtf(2.0f / M_PI);
    const float kCoeff = 0.044715f;

    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = kAlpha * (v + kCoeff * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void leaky_relu_f(float *x, int n, float slope)
{
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f)
            x[i] *= slope;
}

void tanh_f(float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = tanhf(x[i]);
}

void sigmoid_f(float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

float randn_f(void)
{
    static int has_spare = 0;
    static float spare = 0.0f;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u1, u2;
    do {
        u1 = (float)rand() / ((float)RAND_MAX + 1.0f);
    } while (u1 <= 0.0f);
    u2 = (float)rand() / ((float)RAND_MAX + 1.0f);

    float r = sqrtf(-2.0f * logf(u1));
    float angle = 2.0f * (float)M_PI * u2;

    spare = r * sinf(angle);
    has_spare = 1;

    return r * cosf(angle);
}
