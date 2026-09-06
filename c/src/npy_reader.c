#include "vits.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Minimal .npy file reader for float32 arrays.
 * Supports only fortran_order=False, descr='<f4' (little-endian float32).
 *
 * .npy format:
 *   [7 bytes]  magic: \x93NUMPY
 *   [2 bytes]  major version (uint16 LE)
 *   [2 bytes]  minor version (uint16 LE)
 *   [2 bytes]  header length (uint16 LE)
 *   [N bytes]  header (Python dict as string, NUL-padded to even length)
 *   [data]     raw array data
 */

/*
 * Load a .npy file containing float32 data.
 * Returns a malloc'd buffer of n_elements floats, or NULL on error.
 * *n_elements is set to the total number of elements.
 * If expected_n > 0, the file must have exactly that many elements.
 */
float *load_npy_f32(const char *path, int *n_elements, int expected_n)
{
    *n_elements = 0;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "  [inject] cannot open %s\n", path);
        return NULL;
    }

    /* Read magic: 6 bytes: \x93NUMPY */
    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "  [inject] not a .npy file: %s\n", path);
        fclose(f);
        return NULL;
    }

    /* Read version (major, minor) */
    unsigned char ver[2];
    if (fread(ver, 1, 2, f) != 2) {
        fprintf(stderr, "  [inject] short header in %s\n", path);
        fclose(f);
        return NULL;
    }
    int major = ver[0];
    if (major > 1) {
        fprintf(stderr, "  [inject] unsupported .npy version %d in %s\n", major, path);
        fclose(f);
        return NULL;
    }

    /* Read header length */
    unsigned char hdr_len_buf[2];
    if (fread(hdr_len_buf, 1, 2, f) != 2) {
        fprintf(stderr, "  [inject] short header length in %s\n", path);
        fclose(f);
        return NULL;
    }
    int hdr_len = hdr_len_buf[0] | (hdr_len_buf[1] << 8);
    if (hdr_len < 4 || hdr_len > 1024) {
        fprintf(stderr, "  [inject] implausible header length %d in %s\n", hdr_len, path);
        fclose(f);
        return NULL;
    }

    /* Read header string */
    char *hdr = (char *)malloc(hdr_len + 1);
    if (!hdr) { fclose(f); return NULL; }
    if (fread(hdr, 1, hdr_len, f) != (size_t)hdr_len) {
        fprintf(stderr, "  [inject] failed to read header in %s\n", path);
        free(hdr);
        fclose(f);
        return NULL;
    }
    hdr[hdr_len] = '\0';

    /*
     * Parse shape from header.
     * Header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 37), }
     */
    int total = 0;
    int shape0 = 0, shape1 = 0;
    int fortran_order = 0;
    if (strstr(hdr, "'fortran_order': True"))
        fortran_order = 1;

    const char *shape_str = strstr(hdr, "'shape': ");
    if (shape_str) {
        shape_str += 9; /* skip "'shape': " → now at '(' or whitespace */
        while (*shape_str == ' ') shape_str++;
        if (*shape_str == '(') {
            shape_str++;
            total = 1;
            int val = 0;
            int in_num = 0;
            int dim_idx = 0;
            while (*shape_str && *shape_str != ')') {
                if (*shape_str >= '0' && *shape_str <= '9') {
                    val = val * 10 + (*shape_str - '0');
                    in_num = 1;
                } else if (in_num) {
                    if (dim_idx == 0) shape0 = val;
                    else if (dim_idx == 1) shape1 = val;
                    total *= val;
                    val = 0;
                    in_num = 0;
                    dim_idx++;
                }
                shape_str++;
            }
            if (in_num) {
                if (dim_idx == 0) shape0 = val;
                else if (dim_idx == 1) shape1 = val;
                total *= val;
            }
        }
    }

    /* For 0-d arrays, shape is () meaning 1 element */
    if (total == 0 && strstr(hdr, "'shape': ()"))
        total = 1;

    free(hdr);

    if (expected_n > 0 && total != expected_n) {
        fprintf(stderr, "  [inject] size mismatch in %s: got %d, expected %d\n",
                path, total, expected_n);
        fclose(f);
        return NULL;
    }

    if (total <= 0) {
        fprintf(stderr, "  [inject] could not determine shape from %s\n", path);
        fclose(f);
        return NULL;
    }

    /* Read data */
    float *data = (float *)malloc(total * sizeof(float));
    if (!data) { fclose(f); return NULL; }

    size_t nread = fread(data, sizeof(float), (size_t)total, f);
    if ((int)nread != total) {
        fprintf(stderr, "  [inject] read only %d/%d elements from %s\n",
                (int)nread, total, path);
        free(data);
        fclose(f);
        return NULL;
    }

    fclose(f);

    /* If Fortran-ordered 2D array, transpose to row-major (C-order) */
    if (fortran_order && shape0 > 1 && shape1 > 1) {
        float *transposed = (float *)malloc(total * sizeof(float));
        if (!transposed) { free(data); return NULL; }
        for (int i = 0; i < shape0; i++)
            for (int j = 0; j < shape1; j++)
                transposed[i * shape1 + j] = data[j * shape0 + i];
        free(data);
        data = transposed;
    }

    *n_elements = total;
    return data;
}
