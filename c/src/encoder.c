#include "vits.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================
 * Attention forward (all heads, single batch)
 *
 * x: [HIDDEN][T]  (channel-first)
 * out: [HIDDEN][T]
 * ============================================================ */
static void attention_forward(const Attn *a, const float *x, int T,
                              const int *mask, float *out)
{
    size_t hdT = (size_t)HEAD_DIM * T;
    size_t TT  = (size_t)T * T;

    float *q    = (float *)malloc(hdT * sizeof(float));
    float *k    = (float *)malloc(hdT * sizeof(float));
    float *v    = (float *)malloc(hdT * sizeof(float));
    float *attn = (float *)malloc(TT * sizeof(float));
    float *o    = (float *)malloc((size_t)HIDDEN * T * sizeof(float));

    for (int h = 0; h < NUM_HEADS; h++) {
        int hd = HEAD_DIM;
        int hd_off = h * hd;

        /* Compute Q, K, V (each [hd][T]) */
        for (int i = 0; i < hd; i++) {
            float bq = a->q_b[hd_off + i];
            float bk = a->k_b[hd_off + i];
            float bv = a->v_b[hd_off + i];
            for (int t = 0; t < T; t++) {
                float sq = bq, sk = bk, sv = bv;
                for (int c = 0; c < HIDDEN; c++) {
                    float xc = x[(size_t)c * T + t];
                    sq += a->q_w[(hd_off + i) * HIDDEN + c] * xc;
                    sk += a->k_w[(hd_off + i) * HIDDEN + c] * xc;
                    sv += a->v_w[(hd_off + i) * HIDDEN + c] * xc;
                }
                if (mask && !mask[t]) { sq = 0; sk = 0; sv = 0; }
                q[i * T + t] = sq * ATTEN_SCALE;
                k[i * T + t] = sk;
                v[i * T + t] = sv;
            }
        }

        /* Attention scores: [T][T] */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float s = 0.0f;
                for (int d = 0; d < hd; d++)
                    s += q[d * T + i] * k[d * T + j];
                attn[i * T + j] = s;
            }
        }

        /* Relative position bias on keys (shared across heads) */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                int rel_idx = j - i + WINDOW;
                if (rel_idx < 0 || rel_idx >= REL_SIZE) continue;
                float rb = 0.0f;
                for (int d = 0; d < hd; d++)
                    rb += q[d * T + i] * a->rel_k[rel_idx * HEAD_DIM + d];
                attn[i * T + j] += rb;
            }
        }

        /* Apply mask and softmax */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (mask && !mask[j])
                    attn[i * T + j] = -1e10f;
            }
            softmax_f(attn + i * T, T);
        }

        /* Output: attn @ V, accumulate into full o at head offset */
        for (int d = 0; d < hd; d++) {
            for (int t = 0; t < T; t++) {
                float s = 0.0f;
                for (int j = 0; j < T; j++)
                    s += attn[t * T + j] * v[d * T + j];
                o[(hd_off + d) * T + t] = s;
            }
        }

        /* Relative value bias (shared across heads) */
        for (int i = 0; i < T; i++) {
            for (int d = 0; d < hd; d++) {
                float rb = 0.0f;
                for (int j = 0; j < T; j++) {
                    int rel_idx = j - i + WINDOW;
                    if (rel_idx < 0 || rel_idx >= REL_SIZE) continue;
                    rb += attn[i * T + j] * a->rel_v[rel_idx * HEAD_DIM + d];
                }
                o[(hd_off + d) * T + i] += rb;
            }
        }
    }

    /* Output projection: [HIDDEN][HIDDEN] @ [HIDDEN][T] */
    for (int i = 0; i < HIDDEN; i++) {
        float bo = a->o_b[i];
        for (int t = 0; t < T; t++) {
            float s = bo;
            for (int c = 0; c < HIDDEN; c++)
                s += a->o_w[i * HIDDEN + c] * o[(size_t)c * T + t];
            out[(size_t)i * T + t] = s;
        }
    }

    free(q);
    free(k);
    free(v);
    free(attn);
    free(o);
}

/* ============================================================
 * Feed-forward (Conv1d x2 with ReLU)
 * ============================================================ */
static void ffn_forward(const EncLayer *layer, const float *x, int T,
                        const int *mask, float *out)
{
    size_t fT = (size_t)FFN_DIM * T;
    float *tmp = (float *)malloc(fT * sizeof(float));

    Conv1d c1 = {
        .in_ch = HIDDEN, .out_ch = FFN_DIM, .k = FFN_KERNEL,
        .pad = 1, .dilation = 1,
        .weight = (float *)layer->ffn1_w, .bias = (float *)layer->ffn1_b
    };
    conv1d(x, HIDDEN, T, &c1, tmp);
    relu_f(tmp, (int)fT);
    mask_zero_f(tmp, mask, FFN_DIM, T);

    Conv1d c2 = {
        .in_ch = FFN_DIM, .out_ch = HIDDEN, .k = FFN_KERNEL,
        .pad = 1, .dilation = 1,
        .weight = (float *)layer->ffn2_w, .bias = (float *)layer->ffn2_b
    };
    conv1d(tmp, FFN_DIM, T, &c2, out);
    mask_zero_f(out, mask, HIDDEN, T);

    free(tmp);
}

/* ============================================================
 * Full encoder forward
 * ============================================================ */
void encoder_forward(const VitsModel *m,
                     const int32_t *input_ids, int T,
                     const int *mask,
                     float *hidden,
                     float *prior_means,
                     float *prior_log_vars)
{
    size_t hT = (size_t)HIDDEN * T;
    float *x   = (float *)malloc(hT * sizeof(float));
    float *tmp = (float *)malloc(hT * sizeof(float));
    float *proj = (float *)malloc(2 * hT * sizeof(float));

    /* Embedding */
    for (int t = 0; t < T; t++) {
        int id = input_ids[t];
        for (int c = 0; c < HIDDEN; c++)
            x[(size_t)c * T + t] = m->embed_w[(size_t)id * HIDDEN + c] * EMBED_SCALE;
        if (mask && !mask[t])
            for (int c = 0; c < HIDDEN; c++)
                x[(size_t)c * T + t] = 0.0f;
    }

    /* Transformer layers */
    for (int l = 0; l < NUM_LAYERS; l++) {
        const EncLayer *layer = &m->layers[l];

        attention_forward(&layer->attn, x, T, mask, tmp);
        axpy_f(x, tmp, (int)hT);
        layer_norm(x, T, HIDDEN, &layer->ln1, tmp);
        memcpy(x, tmp, hT * sizeof(float));

        ffn_forward(layer, x, T, mask, tmp);
        axpy_f(x, tmp, (int)hT);
        layer_norm(x, T, HIDDEN, &layer->ln2, tmp);
        memcpy(x, tmp, hT * sizeof(float));
    }

    memcpy(hidden, x, hT * sizeof(float));

    /* Project to prior distribution */
    Conv1d pc = {
        .in_ch = HIDDEN, .out_ch = 2 * HIDDEN, .k = 1,
        .pad = 0, .dilation = 1,
        .weight = (float *)m->proj_w, .bias = (float *)m->proj_b
    };
    conv1d(hidden, HIDDEN, T, &pc, proj);

    for (int t = 0; t < T; t++)
        for (int c = 0; c < HIDDEN; c++) {
            prior_means[c * T + t] = proj[(size_t)c * T + t];
            prior_log_vars[c * T + t] = proj[(size_t)(HIDDEN + c) * T + t];
        }
    mask_zero_f(prior_means, mask, HIDDEN, T);
    mask_zero_f(prior_log_vars, mask, HIDDEN, T);

    free(x);
    free(tmp);
    free(proj);
}
