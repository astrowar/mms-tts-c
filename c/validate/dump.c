#include "dump.h"
#include <stdio.h>
#include <stdlib.h>

static size_t total_size(int ndim, const int *dims)
{
    size_t n = 1;
    for (int i = 0; i < ndim; i++)
        n *= (size_t)dims[i];
    return n;
}

void dump_f32(const char *dir, const char *name,
              const float *data, int ndim, const int *dims)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);

    FILE *f = fopen(path, "wb");
    if (!f) return;

    int32_t ndim_i = ndim;
    fwrite(&ndim_i, sizeof(int32_t), 1, f);
    for (int i = 0; i < ndim; i++) {
        int32_t d = dims[i];
        fwrite(&d, sizeof(int32_t), 1, f);
    }

    size_t n = total_size(ndim, dims);
    fwrite(data, sizeof(float), n, f);

    fclose(f);
}

void dump_i32(const char *dir, const char *name,
              const int32_t *data, int ndim, const int *dims)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);

    FILE *f = fopen(path, "wb");
    if (!f) return;

    int32_t ndim_i = ndim;
    fwrite(&ndim_i, sizeof(int32_t), 1, f);
    for (int i = 0; i < ndim; i++) {
        int32_t d = dims[i];
        fwrite(&d, sizeof(int32_t), 1, f);
    }

    size_t n = total_size(ndim, dims);
    fwrite(data, sizeof(int32_t), n, f);

    fclose(f);
}
