#ifndef DUMP_H
#define DUMP_H

/*
 * Simple binary dump for intermediate tensors.
 * Format: [int32 ndim][int32 dims...][float32 data (row-major)]
 *
 * Use compare.py to load and diff against Python .npy reference.
 */

#include <stdint.h>

void dump_f32(const char *dir, const char *name,
              const float *data, int ndim, const int *dims);

void dump_i32(const char *dir, const char *name,
              const int32_t *data, int ndim, const int *dims);

#endif /* DUMP_H */
