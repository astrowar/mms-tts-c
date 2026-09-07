#include "vits.h"
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_DUMP
#include "../validate/dump.h"
#include <stdio.h>
static const char *hifi_dump_dir = NULL;
void hifigan_set_dump_dir(const char *dir) { hifi_dump_dir = dir; }
#endif

/*
 * HiFi-GAN vocoder forward pass for mms-tts-por.
 *
 * Architecture:
 *   conv_pre : Conv1d(192, 512, k=7, pad=3)
 *   4 stages, each:
 *       LeakyReLU(0.1)
 *       ConvTranspose1d (upsample)
 *       MRF: sum of 3 parallel ResBlocks / 3
 *   LeakyReLU(0.1)
 *   conv_post: Conv1d(32, 1, k=7, pad=3, no bias)
 *   tanh
 *
 * ResBlock (kernel K, dilations [1,3,5]) — sub-blocks are SEQUENTIAL:
 *   for each dilation d:
 *       residual = x
 *       x = LeakyReLU(x)
 *       x = conv1_d(x)       // Conv1d(C, C, K, dil=d, pad=d*(K-1)/2)
 *       x = LeakyReLU(x)
 *       x = conv2_d(x)       // Conv1d(C, C, K, dil=1, pad=(K-1)/2)
 *       x = x + residual
 *
 * MRF at each stage — the 3 ResBlocks (different K) are applied IN PARALLEL
 * to the same upsampled input, outputs are summed and divided by 3.
 */

void hifigan_forward(const HiFiGan *dec,
                     const float *mel, int mel_T,
                     float *wave, int *wave_T)
{
    int wave_len = mel_T * TOTAL_UP;

    /*
     * Maximum intermediate buffer size:
     *   stage 0: 256 * (mel_T*8 + 17)   ~ 2048 * mel_T + 4352
     *   stage 1: 128 * (mel_T*64 + 17)  ~ 8192 * mel_T + 2176
     *   stage 2:  64 * (mel_T*128 + 9)  ~ 8192 * mel_T + 576
     *   stage 3:  32 * (mel_T*256 + 2)  ~ 8192 * mel_T + 64
     * So max is 32 * wave_len + 8192 (safe margin for kernel overshoot).
     */
    size_t max_size = (size_t)32 * wave_len + 8192;

    /*
     * Working buffers:
     *   A — current state / MRF accumulator
     *   B — upsample output (constant during MRF)
     *   C — ResBlock working buffer
     *   D — ResBlock residual
     *   E — ResBlock conv temp
     */
    float *A = (float *)malloc(max_size * sizeof(float));
    float *B = (float *)malloc(max_size * sizeof(float));
    float *C = (float *)malloc(max_size * sizeof(float));
    float *D = (float *)malloc(max_size * sizeof(float));
    float *E = (float *)malloc(max_size * sizeof(float));

    /* ---- 1. conv_pre: [192][mel_T] -> [512][mel_T] ---- */
    conv1d(mel, 192, mel_T, &dec->conv_pre, A);

#ifdef ENABLE_DUMP
    static int hifi_call_count = 0;
    hifi_call_count++;
    int dump_this_call = (hifi_call_count == 1);
    if (dump_this_call && hifi_dump_dir) {
        int dim2[2] = {HIFI_INIT_CH, mel_T};
        dump_f32(hifi_dump_dir, "06_hifi_conv_pre", A, 2, dim2);
    }
#endif

    int cur_ch = HIFI_INIT_CH;
    int cur_T = mel_T;

    /* ---- 2. Four upsampling + MRF stages ---- */
    for (int i = 0; i < NUM_UP; i++) {
        size_t cur_size = (size_t)cur_ch * cur_T;

        /* LeakyReLU before upsampling */
        leaky_relu_f(A, cur_size, LEAKY_SLOPE);

        /* Upsample: A -> B */
        int new_T;
        conv_transpose1d(A, cur_ch, cur_T, &dec->up[i], B, &new_T);
        int new_ch = dec->up[i].out_ch;
        size_t new_size = (size_t)new_ch * new_T;

        /* MRF: 3 ResBlocks in parallel on B, sum into A, divide by 3 */
        memset(A, 0, new_size * sizeof(float));

        for (int j = 0; j < 3; j++) {
            const ResBlock *rb = &dec->rb[i * 3 + j];

            /* C = copy of upsampled input */
            memcpy(C, B, new_size * sizeof(float));

            /* Process 3 dilation sub-blocks sequentially */
            for (int d = 0; d < RF_DILS; d++) {
                /* D = residual (snapshot of C before modification) */
                memcpy(D, C, new_size * sizeof(float));

                /* C = LeakyReLU(C) */
                leaky_relu_f(C, new_size, LEAKY_SLOPE);

                /* E = conv1_d(C) */
                conv1d(C, new_ch, new_T, &rb->c1[d], E);

                /* E = LeakyReLU(E) */
                leaky_relu_f(E, new_size, LEAKY_SLOPE);

                /* C = conv2_d(E) */
                conv1d(E, new_ch, new_T, &rb->c2[d], C);

                /* C = C + D (residual connection) */
                axpy_f(C, D, (int)new_size);
            }

            /* Accumulate: A += C */
            axpy_f(A, C, (int)new_size);
        }

        /* Divide by 3 */
        scal_f(A, (int)new_size, 1.0f / 3.0f);

#ifdef ENABLE_DUMP
        if (dump_this_call && hifi_dump_dir) {
            char name[64];
            snprintf(name, sizeof(name), "06_hifi_stage_%d", i);
            int dims[2] = {new_ch, new_T};
            dump_f32(hifi_dump_dir, name, A, 2, dims);
        }
#endif

        cur_ch = new_ch;
        cur_T = new_T;
    }

    /* ---- 3. Output: LeakyReLU -> conv_post -> tanh ----
     * NOTE: Python uses default slope (0.01) here, not config's 0.1.
     *   hidden_states = nn.functional.leaky_relu(hidden_states)  # default 0.01
     */
    size_t final_size = (size_t)cur_ch * cur_T;
    leaky_relu_f(A, final_size, 0.01f);

    /* conv_post (metadata from struct, no bias) */
    conv1d(A, dec->conv_post.in_ch, cur_T, &dec->conv_post, wave);

    *wave_T = wave_len;

#ifdef ENABLE_DUMP
    if (dump_this_call && hifi_dump_dir) {
        int dim1[1] = {wave_len};
        dump_f32(hifi_dump_dir, "06_hifi_pre_tanh", wave, 1, dim1);
    }
#endif

    /* tanh squashes to [-1, 1] */
    tanh_f(wave, wave_len);

    /* Cleanup */
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
}
