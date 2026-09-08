#include "vits.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Optional dump support (compile with -DENABLE_DUMP) */
#ifdef ENABLE_DUMP
#include "../validate/dump.h"
static const char *g_dump_dir = NULL;

static void maybe_dump_f32(const char *name, const float *data, int ndim, int *dims)
{
    if (g_dump_dir)
        dump_f32(g_dump_dir, name, data, ndim, dims);
}

static void maybe_dump_i32(const char *name, const int32_t *data, int ndim, int *dims)
{
    if (g_dump_dir)
        dump_i32(g_dump_dir, name, data, ndim, dims);
}
#else
static void maybe_dump_f32(const char *name, const float *data, int ndim, int *dims) { (void)name; (void)data; (void)ndim; (void)dims; }
static void maybe_dump_i32(const char *name, const int32_t *data, int ndim, int *dims) { (void)name; (void)data; (void)ndim; (void)dims; }
#endif

void set_dump_dir(const char *dir)
{
#ifdef ENABLE_DUMP
    g_dump_dir = dir;
#endif
}

static int vits_synthesize_impl(const VitsModel *m, const Vocab *vocab,
                                const char *text, int seed,
                                const char *inject_dir,
                                float *waveform, int *wave_len,
                                int use_q16);

/* ============================================================
 * vits_synthesize: Full inference pipeline
 *
 * All large buffers are heap-allocated to avoid stack overflow.
 *
 * text: input text
 * seed: RNG seed (-1 for random)
 * inject_dir: if non-NULL, load pre-generated latents from this dir
 *             (e.g. ref_out/) instead of using randn
 * waveform: output buffer (must be at least MAX_WAV_LEN floats)
 * wave_len: output, number of valid samples
 *
 * Returns 0 on success, -1 on error.
 * ============================================================ */
int vits_synthesize(const VitsModel *m, const Vocab *vocab,
                    const char *text, int seed,
                    const char *inject_dir,
                    float *waveform, int *wave_len)
{
    return vits_synthesize_impl(m, vocab, text, seed, inject_dir,
                                waveform, wave_len, 0);
}

/* Extended version with pipeline selection */
int vits_synthesize_pipeline(const VitsModel *m, const Vocab *vocab,
                             const char *text, int seed,
                             const char *inject_dir,
                             float *waveform, int *wave_len,
                             int use_q16)
{
    return vits_synthesize_impl(m, vocab, text, seed, inject_dir,
                                waveform, wave_len, use_q16);
}

int vits_synthesize_impl(const VitsModel *m, const Vocab *vocab,
                         const char *text, int seed,
                         const char *inject_dir,
                         float *waveform, int *wave_len,
                         int use_q16)
{
    if (seed >= 0)
        srand(seed);

    /* Timing */
    double t_start = now_ms(), t_tok, t_enc, t_dp, t_samp, t_flow, t_hifi;

    /* ── Stage 1: Tokenize ─────────────────────────────────────────── */
    int32_t ids[MAX_TOK_LEN];
    int T = 0;
    tokenize(vocab, text, ids, &T);
    t_tok = now_ms();

    /* Attention mask: all 1s (no batch padding for single sequence).
     * Blank tokens (id=0) from add_blank are NOT padding — they participate
     * fully in attention and conditioning. */
    int mask[MAX_TOK_LEN];
    for (int t = 0; t < T; t++)
        mask[t] = 1;

    maybe_dump_i32("01_token_ids", ids, 1, &T);
    {
        float *mask_f = (float *)malloc(sizeof(float) * T);
        for (int t = 0; t < T; t++) mask_f[t] = mask[t];
        int d1 = T;
        maybe_dump_f32("01_attention_mask", mask_f, 1, &d1);
        free(mask_f);
    }

    /* ── Stage 2: Text Encoder ─────────────────────────────────────── */
    size_t hT = (size_t)HIDDEN * T;
    float *hidden         = (float *)malloc(hT * sizeof(float));
    float *prior_means    = (float *)malloc(hT * sizeof(float));
    float *prior_log_vars = (float *)malloc(hT * sizeof(float));

    if (!hidden || !prior_means || !prior_log_vars) {
        fprintf(stderr, "Error: OOM in encoder stage\n");
        free(hidden); free(prior_means); free(prior_log_vars);
        return -1;
    }

    encoder_forward(m, ids, T, mask, hidden, prior_means, prior_log_vars);

    t_enc = now_ms();

    {
        int dim2[2] = {HIDDEN, T};
        maybe_dump_f32("02_hidden_cf", hidden, 2, dim2);
        maybe_dump_f32("02_prior_means", prior_means, 2, dim2);
        maybe_dump_f32("02_prior_log_vars", prior_log_vars, 2, dim2);
    }

    /* ── Stage 3: Duration Predictor (reverse) ─────────────────────── */
    float *log_duration = (float *)malloc(sizeof(float) * T);
    if (!log_duration) {
        fprintf(stderr, "Error: OOM in DP stage\n");
        free(hidden); free(prior_means); free(prior_log_vars);
        return -1;
    }

    /* Load injected DP latents if available */
    const float *inject_dp_lat = NULL;
    float *inject_dp_buf = NULL;
    if (inject_dir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/03_dp_latents_init.npy", inject_dir);
        int n = 0;
        inject_dp_buf = load_npy_f32(path, &n, DP_CHANNELS * T);
        if (inject_dp_buf) {
            inject_dp_lat = inject_dp_buf;
            printf("  [inject] loaded DP latents from %s (%d elements)\n", path, n);
        } else {
            fprintf(stderr, "  [inject] WARNING: could not load %s, using randn\n", path);
        }
    }

    dp_reverse(&m->dp, hidden, T, mask, inject_dp_lat, log_duration);
    t_dp = now_ms();

    {
        int d1 = T;
        maybe_dump_f32("03_log_duration", log_duration, 1, &d1);
    }

    /* Compute integer durations */
    int *durations = (int *)malloc(sizeof(int) * T);
    if (!durations) {
        fprintf(stderr, "Error: OOM\n");
        free(inject_dp_buf); free(log_duration);
        free(hidden); free(prior_means); free(prior_log_vars);
        return -1;
    }
    int mel_T = 0;
    for (int t = 0; t < T; t++) {
        if (mask[t]) {
            durations[t] = (int)ceilf(expf(log_duration[t]));
            if (durations[t] < 1) durations[t] = 1;
        } else {
            durations[t] = 0;
        }
        mel_T += durations[t];
    }

    if (mel_T > MAX_MEL_LEN) {
        fprintf(stderr, "Error: mel length %d exceeds max %d\n", mel_T, MAX_MEL_LEN);
        free(durations); free(log_duration);
        free(hidden); free(prior_means); free(prior_log_vars);
        return -1;
    }

    {
        float *dur_f = (float *)malloc(sizeof(float) * T);
        for (int t = 0; t < T; t++) dur_f[t] = (float)durations[t];
        int d1 = T;
        maybe_dump_f32("03_duration", dur_f, 1, &d1);
        free(dur_f);
    }

    /* ── Stage 4: Prior Expansion + Sampling ───────────────────────── */
    size_t mel_size = (size_t)FLOW_SIZE * mel_T;
    float *latents = (float *)malloc(mel_size * sizeof(float));
    if (!latents) {
        fprintf(stderr, "Error: OOM in sampling stage\n");
        free(inject_dp_buf);
        free(durations); free(log_duration);
        free(hidden); free(prior_means); free(prior_log_vars);
        return -1;
    }

    if (inject_dir) {
        /* Try to load pre-sampled latents (bypasses randn) */
        char path[512];
        snprintf(path, sizeof(path), "%s/04_latents_sampled.npy", inject_dir);
        int n = 0;
        float *injected = load_npy_f32(path, &n, (int)mel_size);
        if (injected) {
            memcpy(latents, injected, mel_size * sizeof(float));
            free(injected);
            printf("  [inject] loaded prior latents from %s (%d elements)\n", path, n);
        } else {
            fprintf(stderr, "  [inject] WARNING: could not load %s, using randn\n", path);
            /* Fall through to normal sampling */
            int mel_idx = 0;
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < durations[t]; d++) {
                    for (int c = 0; c < HIDDEN; c++) {
                        float sampled = prior_means[c * T + t] +
                            randn_f() * expf(prior_log_vars[c * T + t]) * NOISE_SCALE;
                        latents[(size_t)c * mel_T + mel_idx] = sampled;
                    }
                    mel_idx++;
                }
            }
        }
    } else {
        int mel_idx = 0;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < durations[t]; d++) {
                for (int c = 0; c < HIDDEN; c++) {
                    float sampled = prior_means[c * T + t] +
                        randn_f() * expf(prior_log_vars[c * T + t]) * NOISE_SCALE;
                    latents[(size_t)c * mel_T + mel_idx] = sampled;
                }
                mel_idx++;
            }
        }
    }

    t_samp = now_ms();

    {
        int dim_mel2[2] = {FLOW_SIZE, mel_T};
        maybe_dump_f32("04_latents_sampled", latents, 2, dim_mel2);
    }

    /* ── Stage 5: Flow (reverse: prior → spectrogram) ──────────────── */
    int *out_mask = (int *)malloc(sizeof(int) * mel_T);
    if (!out_mask) {
        fprintf(stderr, "Error: OOM\n");
        free(inject_dp_buf); free(latents); free(durations); free(log_duration);
        free(hidden); free(prior_means); free(prior_log_vars);
        return -1;
    }
    for (int t = 0; t < mel_T; t++) out_mask[t] = 1;

    flow_reverse(&m->flow, latents, mel_T, out_mask);
    free(out_mask);
    t_flow = now_ms();

    {
        int dim_mel2[2] = {FLOW_SIZE, mel_T};
        maybe_dump_f32("05_flow_output", latents, 2, dim_mel2);
    }

    if (inject_dir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/05_flow_output.npy", inject_dir);
        int n = 0;
        float *injected_flow = load_npy_f32(path, &n, (int)mel_size);
        if (injected_flow) {
            memcpy(latents, injected_flow, mel_size * sizeof(float));
            free(injected_flow);
            printf("  [inject] loaded flow output from %s (%d elements)\n", path, n);
        } else {
            fprintf(stderr, "  [inject] WARNING: could not load %s, using computed flow output\n", path);
        }
    }

    /* ── Stage 6: HiFi-GAN Decoder ─────────────────────────────────── */
    if (use_q16 && m->decoder_q16.params) {
        /* Q16 fixed-point pipeline: outputs INT16 PCM directly */
        int16_t *pcm = (int16_t *)malloc(sizeof(int16_t) *
                                          (size_t)mel_T * TOTAL_UP);
        if (!pcm) {
            fprintf(stderr, "Error: OOM in Q16 HiFi-GAN\n");
            free(inject_dp_buf); free(latents); free(durations);
            free(log_duration); free(hidden);
            free(prior_means); free(prior_log_vars);
            return -1;
        }
        hifigan_forward_q16(&m->decoder_q16, latents, mel_T, pcm, wave_len);
        /* Convert INT16 → float for the existing output pipeline */
        for (int i = 0; i < *wave_len; i++)
            waveform[i] = (float)pcm[i] / 32768.0f;
        free(pcm);
    } else {
        hifigan_forward_q(&m->decoder_q, latents, mel_T, waveform, wave_len);
    }
    t_hifi = now_ms();

    {
        int d1 = *wave_len;
        maybe_dump_f32("06_waveform", waveform, 1, &d1);
    }

    /* Print timing */
    double total = t_hifi - t_start;
    printf("  [profile] T=%d mel_T=%d wave=%.2fs total=%.1fms\n",
           T, mel_T, (double)*wave_len / m->sampling_rate, total);
    printf("    tokenize:  %7.1fms (%5.1f%%)\n", t_tok - t_start, 100.0 * (t_tok - t_start) / total);
    printf("    encoder:   %7.1fms (%5.1f%%)\n", t_enc - t_tok, 100.0 * (t_enc - t_tok) / total);
    printf("    duration:  %7.1fms (%5.1f%%)\n", t_dp - t_enc, 100.0 * (t_dp - t_enc) / total);
    printf("    sampling:  %7.1fms (%5.1f%%)\n", t_samp - t_dp, 100.0 * (t_samp - t_dp) / total);
    printf("    flow:      %7.1fms (%5.1f%%)\n", t_flow - t_samp, 100.0 * (t_flow - t_samp) / total);
    printf("    hifi-gan:  %7.1fms (%5.1f%%)\n", t_hifi - t_flow, 100.0 * (t_hifi - t_flow) / total);

    /* ── Stage 7: Normalize + WAV (validation) ─────────────────────── */
#ifdef ENABLE_DUMP
    if (g_dump_dir) {
        float peak = 0.0f;
        for (int i = 0; i < *wave_len; i++) {
            float a = fabsf(waveform[i]);
            if (a > peak) peak = a;
        }
        float scale = 1.0f / (peak > 1e-8f ? peak : 1e-8f);

        float *norm = (float *)malloc(sizeof(float) * (*wave_len));
        float *int16_f = (float *)malloc(sizeof(float) * (*wave_len));
        if (norm && int16_f) {
            for (int i = 0; i < *wave_len; i++) {
                float n = waveform[i] * scale;
                if (n > 1.0f) n = 1.0f;
                if (n < -1.0f) n = -1.0f;
                norm[i] = n;
                int16_f[i] = (float)(int16_t)(n * 32767.0f);
            }
            int d1 = *wave_len;
            maybe_dump_f32("07_audio_normalized", norm, 1, &d1);
            maybe_dump_f32("07_audio_int16", int16_f, 1, &d1);

            char wav_path[1024];
            snprintf(wav_path, sizeof(wav_path), "%s/07_output.wav", g_dump_dir);
            if (write_wav16(wav_path, waveform, *wave_len, m->sampling_rate) == 0)
                printf("  [dump] wrote %s\n", wav_path);
            free(norm);
            free(int16_f);
        }
    }
#endif

    /* Cleanup */
    free(inject_dp_buf);
    free(latents);
    free(durations);
    free(log_duration);
    free(hidden);
    free(prior_means);
    free(prior_log_vars);

    return 0;
}
