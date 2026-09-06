#include "vits.h"
#include <stdio.h>
#include <string.h>

/*
 * Write a 16-bit PCM mono WAV file.
 * Returns 0 on success, -1 on error.
 */
int write_wav16(const char *path, const float *samples,
                int n_samples, int sample_rate)
{
    if (path == NULL || samples == NULL || n_samples <= 0)
        return -1;

    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    /* Compute peak amplitude */
    float peak = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float a = fabsf(samples[i]);
        if (a > peak)
            peak = a;
    }

    float scale = 1.0f / (peak > 1e-8f ? peak : 1e-8f);

    /* --- Build WAV header (44 bytes, little-endian) --- */

    uint32_t data_size = (uint32_t)n_samples * 2;
    uint32_t riff_size = 36 + data_size;
    uint32_t byte_rate = (uint32_t)sample_rate * 2;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;
    uint16_t num_channels = 1;
    uint16_t audio_format = 1;  /* PCM */
    uint32_t fmt_size = 16;

    uint8_t header[44];
    int pos = 0;

    /* "RIFF" */
    memcpy(&header[pos], "RIFF", 4); pos += 4;
    /* RIFF chunk size (little-endian u32) */
    header[pos++] = riff_size & 0xFF;
    header[pos++] = (riff_size >> 8) & 0xFF;
    header[pos++] = (riff_size >> 16) & 0xFF;
    header[pos++] = (riff_size >> 24) & 0xFF;
    /* "WAVE" */
    memcpy(&header[pos], "WAVE", 4); pos += 4;

    /* "fmt " */
    memcpy(&header[pos], "fmt ", 4); pos += 4;
    /* fmt chunk size = 16 */
    header[pos++] = fmt_size & 0xFF;
    header[pos++] = (fmt_size >> 8) & 0xFF;
    header[pos++] = (fmt_size >> 16) & 0xFF;
    header[pos++] = (fmt_size >> 24) & 0xFF;
    /* audio format = 1 (PCM) */
    header[pos++] = audio_format & 0xFF;
    header[pos++] = (audio_format >> 8) & 0xFF;
    /* num channels = 1 (mono) */
    header[pos++] = num_channels & 0xFF;
    header[pos++] = (num_channels >> 8) & 0xFF;
    /* sample rate */
    header[pos++] = (uint8_t)(sample_rate & 0xFF);
    header[pos++] = (uint8_t)((sample_rate >> 8) & 0xFF);
    header[pos++] = (uint8_t)((sample_rate >> 16) & 0xFF);
    header[pos++] = (uint8_t)((sample_rate >> 24) & 0xFF);
    /* byte rate */
    header[pos++] = byte_rate & 0xFF;
    header[pos++] = (byte_rate >> 8) & 0xFF;
    header[pos++] = (byte_rate >> 16) & 0xFF;
    header[pos++] = (byte_rate >> 24) & 0xFF;
    /* block align */
    header[pos++] = block_align & 0xFF;
    header[pos++] = (block_align >> 8) & 0xFF;
    /* bits per sample */
    header[pos++] = bits_per_sample & 0xFF;
    header[pos++] = (bits_per_sample >> 8) & 0xFF;

    /* "data" */
    memcpy(&header[pos], "data", 4); pos += 4;
    /* data chunk size */
    header[pos++] = data_size & 0xFF;
    header[pos++] = (data_size >> 8) & 0xFF;
    header[pos++] = (data_size >> 16) & 0xFF;
    header[pos++] = (data_size >> 24) & 0xFF;

    if (pos != 44) { /* sanity check */
        fclose(f);
        return -1;
    }

    if (fwrite(header, 1, 44, f) != 44) {
        fclose(f);
        return -1;
    }

    /* --- Write audio data --- */
    for (int i = 0; i < n_samples; i++) {
        float normalized = samples[i] * scale;
        /* Clamp to [-1, 1] before conversion */
        if (normalized > 1.0f)   normalized = 1.0f;
        if (normalized < -1.0f)  normalized = -1.0f;

        int16_t s = (int16_t)(normalized * 32767.0f);

        uint8_t bytes[2];
        bytes[0] = (uint8_t)(s & 0xFF);
        bytes[1] = (uint8_t)((s >> 8) & 0xFF);

        if (fwrite(bytes, 1, 2, f) != 2) {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}
