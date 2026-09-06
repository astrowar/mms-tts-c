#include "vits.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef ENABLE_DUMP
#include "../validate/dump.h"
static const char *flow_dump_dir = NULL;
void flow_set_dump_dir(const char *dir) { flow_dump_dir = dir; }
#endif

/* ============================================================
 * WaveNet forward
 *
 * input: [HIDDEN][T]
 * mask: [T]
 * output: [HIDDEN][T]
 *
 * Each layer:
 *   hidden = conv(input, 192->384, k=5, dil=1, pad=2)   [384][T]
 *   t = tanh(hidden[:192])                               [192][T]
 *   s = sigmoid(hidden[192:])                            [192][T]
 *   acts = t * s                                         [192][T]
 *
 *   res_skip = conv(acts, 192->384_or_192, k=1)
 *   if not last: input = (input + res_skip[:192]) * mask; output += res_skip[192:]
 *   if last:     output += res_skip[:192]
 * ============================================================ */
static void wavenet_forward(const WaveNet *wn, const float *input, int T,
                            const int *mask, float *output)
{
    int C = HIDDEN;   /* 192 */
    int HC = 2 * C;   /* 384 */
    size_t cT = (size_t)C * T;
    size_t hcT = (size_t)HC * T;

    /* Heap buffers sized to actual T */
    float *inp    = (float *)malloc(sizeof(float) * cT);
    float *hidden = (float *)malloc(sizeof(float) * hcT);
    float *acts   = (float *)malloc(sizeof(float) * cT);
    float *rs     = (float *)malloc(sizeof(float) * hcT);

    if (!inp || !hidden || !acts || !rs) {
        free(inp);
        free(hidden);
        free(acts);
        free(rs);
        return;
    }

    /* `wavenet_forward()` is used in-place, so copy the source first. */
    memcpy(inp, input, sizeof(float) * cT);

    /* Zero output after copying, otherwise an in-place call wipes the input. */
    memset(output, 0, sizeof(float) * cT);

#ifdef ENABLE_DUMP
    static int wn_call_count = 0;
    wn_call_count++;
    int dump_this_call = (flow_dump_dir && wn_call_count == 1);
    if (dump_this_call) {
        int dim2[2] = {C, T};
        dump_f32(flow_dump_dir, "05_wn_input", inp, 2, dim2);
    }
#endif

    for (int i = 0; i < wn->num_layers; i++) {
        /* in_layer: Conv1d(192, 384, 5, dil=1, pad=2) */
        Conv1d ic = {
            .in_ch = C, .out_ch = HC, .k = WAVE_KERNEL,
            .pad = 2, .dilation = 1,
            .weight = wn->in_w[i], .bias = wn->in_b[i]
        };
        conv1d(inp, C, T, &ic, hidden);

        /* Gating: tanh * sigmoid */
        gating_f(hidden, hidden + C * T, acts, (int)cT);

        /* res_skip: Conv1d(192, rs_out[i], 1) */
        Conv1d rsc = {
            .in_ch = C, .out_ch = wn->rs_out[i], .k = 1,
            .pad = 0, .dilation = 1,
            .weight = wn->rs_w[i], .bias = wn->rs_b[i]
        };
        conv1d(acts, C, T, &rsc, rs);

        if (i < wn->num_layers - 1) {
            masked_axpy_f(inp, rs, mask, C, T);
            axpy_f(output, rs + C * T, (int)cT);
        } else {
            axpy_f(output, rs, (int)cT);
        }

#ifdef ENABLE_DUMP
        if (dump_this_call) {
            char name[64];
            snprintf(name, sizeof(name), "05_wn_layer_%d_hidden", i);
            int dim2[2] = {HC, T};
            dump_f32(flow_dump_dir, name, hidden, 2, dim2);
            snprintf(name, sizeof(name), "05_wn_layer_%d_rs", i);
            dump_f32(flow_dump_dir, name, rs, 2, dim2);
            snprintf(name, sizeof(name), "05_wn_layer_%d_out", i);
            dump_f32(flow_dump_dir, name, output, 2, dim2);
        }
#endif
    }

    /* Apply mask to output */
    mask_zero_f(output, mask, C, T);

    free(inp);
    free(hidden);
    free(acts);
    free(rs);
}

/* ============================================================
 * Residual coupling layer reverse
 * ============================================================ */
static void coupling_reverse(const CouplingLayer *cl, float *latents, int T,
                             const int *mask)
{
    int half = HALF_FLOW;
    size_t hT = (size_t)HIDDEN * T;
    size_t halfT = (size_t)half * T;

    float *hidden = (float *)malloc(sizeof(float) * hT);
    float *mean   = (float *)malloc(sizeof(float) * halfT);

    /* conv_pre: Conv1d(96, 192, 1) */
    {
        Conv1d cp = {
            .in_ch = half, .out_ch = HIDDEN, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = cl->conv_pre_w, .bias = cl->conv_pre_b
        };
        conv1d(latents, half, T, &cp, hidden);
    }
    mask_zero_f(hidden, mask, HIDDEN, T);

    /* WaveNet (in-place: hidden -> hidden) */
    wavenet_forward(&cl->wavenet, hidden, T, mask, hidden);

    /* conv_post: Conv1d(192, 96, 1) */
    {
        Conv1d cp = {
            .in_ch = HIDDEN, .out_ch = half, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = cl->conv_post_w, .bias = cl->conv_post_b
        };
        conv1d(hidden, HIDDEN, T, &cp, mean);
    }
    mask_zero_f(mean, mask, half, T);

    /* Reverse: second_half -= mean */
    float *second = latents + halfT;
    for (int c = 0; c < half; c++)
        for (int t = 0; t < T; t++)
            second[(size_t)c * T + t] -= mean[(size_t)c * T + t];

    free(hidden);
    free(mean);
}

/* ============================================================
 * flow_reverse: Full prior encoder flow (4 coupling layers)
 * ============================================================ */
void flow_reverse(const CouplingBlock *block,
                  float *latents, int T,
                  const int *mask)
{
    size_t fullT = (size_t)FLOW_SIZE * T;
    float *tmp = (float *)malloc(sizeof(float) * fullT);

    for (int f = block->num_flows - 1; f >= 0; f--) {
        /* Flip the full channel axis, matching torch.flip(inputs, [1]). */
        memcpy(tmp, latents, sizeof(float) * fullT);
        for (int c = 0; c < FLOW_SIZE; c++) {
            memcpy(latents + (size_t)c * T,
                   tmp + (size_t)(FLOW_SIZE - 1 - c) * T,
                   sizeof(float) * T);
        }

        coupling_reverse(&block->flows[f], latents, T, mask);

#ifdef ENABLE_DUMP
        if (flow_dump_dir) {
            char name[64];
            snprintf(name, sizeof(name), "05_flow_after_%d", f);
            int dim2[2] = {FLOW_SIZE, T};
            dump_f32(flow_dump_dir, name, latents, 2, dim2);
        }
#endif
    }

    free(tmp);
}
