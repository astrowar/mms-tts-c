#include "vits.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef ENABLE_DUMP
#include "../validate/dump.h"
static const char *dp_dump_dir = NULL;
void dp_set_dump_dir(const char *dir) { dp_dump_dir = dir; }
#endif

/* ============================================================
 * DDS forward: [dim][T] -> [dim][T]
 * 3 layers: depthwise(dilated) -> LN -> GELU -> pointwise -> LN -> GELU -> residual
 * ============================================================ */
static void dds_forward(const DDS *d, const float *in, int T,
                        const int *mask, float *out)
{
    int dim = d->dim;
    size_t n = (size_t)dim * T;
    float *buf = (float *)malloc(n * sizeof(float));
    float *buf2 = (float *)malloc(n * sizeof(float));
    if (!buf || !buf2) { free(buf); free(buf2); return; }

    memcpy(out, in, n * sizeof(float));

    for (int i = 0; i < d->num_layers; i++) {
        int dilation = 1;
        for (int p = 0; p < i; p++) dilation *= d->kernel;
        int pad = dilation; /* (k*d - d)/2 = d for k=3 */

        conv1d_depthwise(out, dim, T, d->dw_w[i], d->dw_b[i],
                         d->kernel, dilation, pad, buf);

        layer_norm(buf, T, dim, &d->norm1[i], buf);
        gelu(buf, (int)n);

        Conv1d pc = {
            .in_ch = dim, .out_ch = dim, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = d->pw_w[i], .bias = d->pw_b[i]
        };
        conv1d(buf, dim, T, &pc, buf2);

        layer_norm(buf2, T, dim, &d->norm2[i], buf2);
        gelu(buf2, (int)n);

        masked_axpy_f(out, buf2, mask, dim, T);
    }
    free(buf);
    free(buf2);
}

/* ============================================================
 * Rational Quadratic Spline (RQS)
 * ============================================================ */

#define RQS_BINS  10
#define RQS_MIN_W 1e-3f
/* Boundary derivative value = min_deriv + softplus(log(exp(1-min_deriv)-1)) = 1.0 */
#define RQS_BOUNDARY_DERIV 1.0f

static float softplus_f(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

/*
 * RQS reverse for a single element y given pre-computed parameters.
 * Matches _rational_quadratic_spline(reverse=True) from transformers.
 *
 * Bin is found via cumheights (inverse of forward which uses cumwidths).
 * delta = height/width (local slope), not a derivative-based quantity.
 */
static float rqs_reverse_one(float y,
                             const float *widths, const float *cumwidths,
                             const float *heights, const float *cumheights,
                             const float *derivatives)
{
    float tail = DP_TAIL;
    if (y <= -tail || y >= tail)
        return y;

    /* Find bin: cumheights[bin] <= y < cumheights[bin+1] */
    int bin = 0;
    for (int b = RQS_BINS - 1; b >= 0; b--) {
        if (y >= cumheights[b]) { bin = b; break; }
    }

    float in_cumw = cumwidths[bin];
    float bin_w   = widths[bin];
    float in_cumh = cumheights[bin];
    float bin_h   = heights[bin];
    float deriv   = derivatives[bin];
    float deriv_n = derivatives[bin + 1];

    /* delta = height / width (the "slope" of the bin) */
    float delta = bin_h / bin_w;

    /* intermediate1 = u[bin] + u[bin+1] - 2*delta */
    float inter1 = deriv + deriv_n - 2.0f * delta;

    float inter2 = y - in_cumh;
    float inter3 = inter2 * inter1;

    float a = bin_h * (delta - deriv) + inter3;
    float b = bin_h * deriv - inter3;
    float c = -delta * inter2;

    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) disc = 0.0f;

    float root = (2.0f * c) / (-b - sqrtf(disc));
    return root * bin_w + in_cumw;
}

/*
 * Prepare RQS parameters from conv_proj output [29][T] (channel-first)
 * Output: widths[10][T], cumw[11][T], heights[10][T], cumh[11][T], derivs[11][T]
 *
 * conv_proj outputs are in [out_ch][T] layout: element (c, t) is at proj[c*T + t].
 * Widths: channels 0..9, Heights: channels 10..19, Derivatives: channels 20..28.
 *
 * Conv_proj outputs are divided by sqrt(HIDDEN) before softmax/softplus
 * (matches PyTorch VitsConvFlow which scales by 1/sqrt(filter_channels)).
 */
static void rqs_prepare(const float *proj, int T,
                        float *widths, float *cumw,
                        float *heights, float *cumh,
                        float *derivs)
{
    const float inv_sqrt_hidden = 1.0f / sqrtf((float)HIDDEN);

    for (int t = 0; t < T; t++) {
        float *w  = widths  + (size_t)t * RQS_BINS;
        float *cw = cumw    + (size_t)t * (RQS_BINS + 1);
        float *h  = heights + (size_t)t * RQS_BINS;
        float *ch = cumh    + (size_t)t * (RQS_BINS + 1);
        float *d  = derivs  + (size_t)t * (RQS_BINS + 1);

        /* Widths: proj[b][t] = proj[b*T + t], scale, softmax, min scaling */
        for (int b = 0; b < RQS_BINS; b++)
            w[b] = proj[(size_t)b * T + t] * inv_sqrt_hidden;
        softmax_f(w, RQS_BINS);
        for (int b = 0; b < RQS_BINS; b++)
            w[b] = RQS_MIN_W + (1.0f - RQS_MIN_W * RQS_BINS) * w[b];

        cw[0] = -DP_TAIL;
        for (int b = 0; b < RQS_BINS; b++)
            cw[b + 1] = cw[b] + w[b] * 2.0f * DP_TAIL;
        cw[RQS_BINS] = DP_TAIL;
        for (int b = 0; b < RQS_BINS; b++)
            w[b] = cw[b + 1] - cw[b];

        /* Heights: proj[10+b][t] = proj[(10+b)*T + t] */
        for (int b = 0; b < RQS_BINS; b++)
            h[b] = proj[(size_t)(RQS_BINS + b) * T + t] * inv_sqrt_hidden;
        softmax_f(h, RQS_BINS);
        for (int b = 0; b < RQS_BINS; b++)
            h[b] = RQS_MIN_W + (1.0f - RQS_MIN_W * RQS_BINS) * h[b];

        ch[0] = -DP_TAIL;
        for (int b = 0; b < RQS_BINS; b++)
            ch[b + 1] = ch[b] + h[b] * 2.0f * DP_TAIL;
        ch[RQS_BINS] = DP_TAIL;
        for (int b = 0; b < RQS_BINS; b++)
            h[b] = ch[b + 1] - ch[b];

        /* Derivatives: proj[20+b][t], min_deriv + softplus(raw), boundaries=1.0 */
        d[0] = RQS_BOUNDARY_DERIV;
        for (int b = 0; b < RQS_BINS - 1; b++)
            d[b + 1] = RQS_MIN_W + softplus_f(proj[(size_t)(2 * RQS_BINS + b) * T + t]);
        d[RQS_BINS] = RQS_BOUNDARY_DERIV;
    }
}

/* ============================================================
 * ConvFlow reverse (single coupling layer)
 *
 * x: [2][T] (first_half conditions second_half via RQS)
 * ============================================================ */
static void convflow_reverse(const ConvFlow *cf, float *x, int T,
                             const int *mask, const float *global_cond)
{
    int half = DP_CHANNELS / 2;
    int out_ch = 3 * RQS_BINS - 1; /* 29 */
    size_t hT = (size_t)HIDDEN * T;
    size_t pT = (size_t)out_ch * T;

    float *cond_buf = (float *)malloc(hT * sizeof(float));
    float *proj_buf = (float *)malloc(pT * sizeof(float));
    float *widths   = (float *)malloc((size_t)RQS_BINS * T * sizeof(float));
    float *cumw     = (float *)malloc((size_t)(RQS_BINS + 1) * T * sizeof(float));
    float *heights  = (float *)malloc((size_t)RQS_BINS * T * sizeof(float));
    float *cumh     = (float *)malloc((size_t)(RQS_BINS + 1) * T * sizeof(float));
    float *derivs   = (float *)malloc((size_t)(RQS_BINS + 1) * T * sizeof(float));

    if (!cond_buf || !proj_buf || !widths || !cumw || !heights || !cumh || !derivs) {
        free(cond_buf); free(proj_buf); free(widths); free(cumw);
        free(heights); free(cumh); free(derivs);
        return;
    }

    const float *first_half = x; /* [1][T] = x[0..T-1] */

    /* conv_pre: Conv1d(1, 192, 1), then add global conditioning */
    {
        Conv1d cp = {
            .in_ch = 1, .out_ch = HIDDEN, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = cf->conv_pre_w, .bias = cf->conv_pre_b
        };
        conv1d(first_half, 1, T, &cp, cond_buf);
    }
    axpy_f(cond_buf, global_cond, (int)(HIDDEN * (size_t)T));
    mask_zero_f(cond_buf, mask, HIDDEN, T);

    /* DDS on conditioning (in-place) */
    dds_forward(&cf->dds, cond_buf, T, mask, cond_buf);

    /* conv_proj: Conv1d(192, 29, 1) */
    {
        Conv1d cp = {
            .in_ch = HIDDEN, .out_ch = out_ch, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = cf->conv_proj_w, .bias = cf->conv_proj_b
        };
        conv1d(cond_buf, HIDDEN, T, &cp, proj_buf);
    }
    mask_zero_f(proj_buf, mask, out_ch, T);

    /* Prepare RQS params */
    rqs_prepare(proj_buf, T, widths, cumw, heights, cumh, derivs);

#ifdef ENABLE_DUMP
    /* Dump RQS params for first convflow call (CF4) */
    static int cf_call_count = 0;
    cf_call_count++;
    if (dp_dump_dir && cf_call_count == 1) {
        int dim2[2] = {out_ch, T};
        dump_f32(dp_dump_dir, "03_cf4_conv_proj", proj_buf, 2, dim2);
        int dim2w[2] = {RQS_BINS, T};
        dump_f32(dp_dump_dir, "03_cf4_rqs_widths", widths, 2, dim2w);
        dump_f32(dp_dump_dir, "03_cf4_rqs_heights", heights, 2, dim2w);
        int dim2d[2] = {RQS_BINS + 1, T};
        dump_f32(dp_dump_dir, "03_cf4_rqs_derivs", derivs, 2, dim2d);
    }
#endif

    /* Apply RQS reverse to second_half */
    float *second_half = x + (size_t)half * T;
    for (int t = 0; t < T; t++) {
        if (mask && !mask[t]) continue;
        const float *w  = widths  + (size_t)t * RQS_BINS;
        const float *cw = cumw    + (size_t)t * (RQS_BINS + 1);
        const float *h  = heights + (size_t)t * RQS_BINS;
        const float *ch = cumh    + (size_t)t * (RQS_BINS + 1);
        const float *d  = derivs  + (size_t)t * (RQS_BINS + 1);
        second_half[t] = rqs_reverse_one(second_half[t], w, cw, h, ch, d);
    }

    free(cond_buf); free(proj_buf); free(widths); free(cumw);
    free(heights); free(cumh); free(derivs);
}

/* ============================================================
 * ElementwiseAffine reverse
 * ============================================================ */
static void elem_affine_reverse(const ElemAffine *ea, float *x, int T)
{
    for (int c = 0; c < DP_CHANNELS; c++) {
        float scale = expf(ea->log_scale[c]);
        float trans = ea->translate[c];
        for (int t = 0; t < T; t++)
            x[(size_t)c * T + t] = (x[(size_t)c * T + t] - trans) / scale;
    }
}

/* ============================================================
 * dp_reverse: Stochastic DP inference
 * inject_latents: NULL = generate with randn; else [2][T] pre-computed
 * ============================================================ */
void dp_reverse(const StochDP *dp,
                const float *cond, int T,
                const int *mask,
                const float *inject_latents,
                float *log_duration)
{
    size_t hT = (size_t)HIDDEN * T;
    size_t latT = (size_t)DP_CHANNELS * T;

    float *cond_buf = (float *)malloc(hT * sizeof(float));
    float *dds_out  = (float *)malloc(hT * sizeof(float));
    float *latents  = (float *)malloc(latT * sizeof(float));

    if (!cond_buf || !dds_out || !latents) {
        free(cond_buf); free(dds_out); free(latents);
        return;
    }

    /* Step 1: conv_pre -> conv_dds -> conv_proj */
    {
        Conv1d pc = {
            .in_ch = HIDDEN, .out_ch = HIDDEN, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = dp->conv_pre_w, .bias = dp->conv_pre_b
        };
        conv1d(cond, HIDDEN, T, &pc, cond_buf);
    }

#ifdef ENABLE_DUMP
    if (dp_dump_dir) {
        int dim2[2] = {HIDDEN, T};
        dump_f32(dp_dump_dir, "03_dp_conv_pre", cond_buf, 2, dim2);
    }
#endif

    dds_forward(&dp->conv_dds, cond_buf, T, mask, dds_out);

#ifdef ENABLE_DUMP
    if (dp_dump_dir) {
        int dim2[2] = {HIDDEN, T};
        dump_f32(dp_dump_dir, "03_dp_dds_out", dds_out, 2, dim2);
    }
#endif

    {
        Conv1d pc = {
            .in_ch = HIDDEN, .out_ch = HIDDEN, .k = 1,
            .pad = 0, .dilation = 1,
            .weight = dp->conv_proj_w, .bias = dp->conv_proj_b
        };
        conv1d(dds_out, HIDDEN, T, &pc, cond_buf);
    }

    mask_zero_f(cond_buf, mask, HIDDEN, T);

#ifdef ENABLE_DUMP
    if (dp_dump_dir) {
        int dim2[2] = {HIDDEN, T};
        dump_f32(dp_dump_dir, "03_dp_condition", cond_buf, 2, dim2);
    }
#endif

    /* Step 2: Generate random latents [2][T] (or use injected) */
    if (inject_latents) {
        memcpy(latents, inject_latents, latT * sizeof(float));
    } else {
        for (int i = 0; i < DP_CHANNELS * T; i++)
            latents[i] = randn_f() * NOISE_SCALE_DUR;
    }

    /* Step 3: Flows in reverse [CF4, CF3, CF2, EA] — skip CF1 ("useless vflow") */
    int flow_idx = 0;
    for (int fi = DP_NUM_FLOWS - 1; fi >= 1; fi--) {
        for (int t = 0; t < T; t++) {
            float tmp = latents[t];
            latents[t] = latents[T + t];
            latents[T + t] = tmp;
        }
        convflow_reverse(&dp->flows[fi], latents, T, mask, cond_buf);
#ifdef ENABLE_DUMP
        if (dp_dump_dir) {
            char name[64];
            snprintf(name, sizeof(name), "03_dp_flow_%d_out", flow_idx);
            int dim2[2] = {DP_CHANNELS, T};
            dump_f32(dp_dump_dir, name, latents, 2, dim2);
        }
#endif
        flow_idx++;
    }

    /* EA (with flip) */
    for (int t = 0; t < T; t++) {
        float tmp = latents[t];
        latents[t] = latents[T + t];
        latents[T + t] = tmp;
    }
    elem_affine_reverse(&dp->flow_0, latents, T);
#ifdef ENABLE_DUMP
    if (dp_dump_dir) {
        char name[64];
        snprintf(name, sizeof(name), "03_dp_flow_%d_out", flow_idx);
        int dim2[2] = {DP_CHANNELS, T};
        dump_f32(dp_dump_dir, name, latents, 2, dim2);
    }
#endif

    /* Step 4: Output = first channel */
    for (int t = 0; t < T; t++)
        log_duration[t] = (mask && !mask[t]) ? 0.0f : latents[t];

    free(cond_buf); free(dds_out); free(latents);
}
