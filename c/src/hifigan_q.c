#include "vits.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_DUMP
#include "../validate/dump.h"
static const char *hifi_q_dump_dir = NULL;
void hifigan_q_set_dump_dir(const char *dir) { hifi_q_dump_dir = dir; }
#endif

/* ============================================================
 * Quantized HiFi-GAN forward pass
 *
 * All conv layers use int8 weights with per-channel dequantization.
 * Activations remain float32 throughout.
 * Dumps prefixed "06q_" to distinguish from F32 dumps.
 * ============================================================ */
void hifigan_forward_q(const HiFiGanQ *dec,
                       const float *mel, int mel_T,
                       float *wave, int *wave_T)
{
    int wave_len = mel_T * TOTAL_UP;

    /* Buffer size: largest intermediate is at the last upstage output */
    int last_ch = dec->up[NUM_UP - 1].out_ch;
    size_t max_size = (size_t)last_ch * wave_len + 8192;

    float *A = (float *)malloc(max_size * sizeof(float));
    float *B = (float *)malloc(max_size * sizeof(float));
    float *C = (float *)malloc(max_size * sizeof(float));
    float *D = (float *)malloc(max_size * sizeof(float));
    float *E = (float *)malloc(max_size * sizeof(float));

#ifdef ENABLE_DUMP
    static int hifi_q_call_count = 0;
    hifi_q_call_count++;
    int dump_this_call = (hifi_q_call_count == 1);
#endif

    /* ---- 1. conv_pre ---- */
    conv1d_q(mel, dec->conv_pre.in_ch, mel_T, &dec->conv_pre, A);

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q_dump_dir) {
        int dim2[2] = {dec->conv_pre.out_ch, mel_T};
        dump_f32(hifi_q_dump_dir, "06q_hifi_conv_pre", A, 2, dim2);
    }
#endif

    int cur_ch = dec->conv_pre.out_ch;
    int cur_T = mel_T;

    /* ---- 2. Four upsampling + MRF stages ---- */
    for (int i = 0; i < NUM_UP; i++) {
        size_t cur_size = (size_t)cur_ch * cur_T;

        leaky_relu_f(A, cur_size, LEAKY_SLOPE);

        /* Upsampler */
        int new_T;
        conv_transpose1d_q(A, cur_ch, cur_T, &dec->up[i], B, &new_T);
        int new_ch = dec->up[i].out_ch;
        size_t new_size = (size_t)new_ch * new_T;

        /* MRF: 3 ResBlocks in parallel on B */
        memset(A, 0, new_size * sizeof(float));

        for (int j = 0; j < RF_DILS; j++) {
            const ResBlockQ *rb = &dec->rb[i * RF_DILS + j];

            memcpy(C, B, new_size * sizeof(float));

            for (int d = 0; d < RF_DILS; d++) {
                memcpy(D, C, new_size * sizeof(float));

                leaky_relu_f(C, new_size, LEAKY_SLOPE);

                conv1d_q(C, new_ch, new_T, &rb->c1[d], E);

                leaky_relu_f(E, new_size, LEAKY_SLOPE);

                conv1d_q(E, new_ch, new_T, &rb->c2[d], C);

                axpy_f(C, D, (int)new_size);
            }

            axpy_f(A, C, (int)new_size);
        }

        scal_f(A, (int)new_size, 1.0f / (float)RF_DILS);

#ifdef ENABLE_DUMP
        if (dump_this_call && hifi_q_dump_dir) {
            char name[64];
            snprintf(name, sizeof(name), "06q_hifi_stage_%d", i);
            int dims[2] = {new_ch, new_T};
            dump_f32(hifi_q_dump_dir, name, A, 2, dims);
        }
#endif

        cur_ch = new_ch;
        cur_T = new_T;
    }

    /* ---- 3. Output: LeakyReLU(0.01) -> conv_post -> tanh ---- */
    size_t final_size = (size_t)cur_ch * cur_T;
    leaky_relu_f(A, final_size, 0.01f);

    conv1d_q(A, cur_ch, cur_T, &dec->conv_post, wave);

    *wave_T = wave_len;

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_q_dump_dir) {
        int dim1[1] = {wave_len};
        dump_f32(hifi_q_dump_dir, "06q_hifi_pre_tanh", wave, 1, dim1);
    }
#endif

    tanh_f(wave, wave_len);

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
}
