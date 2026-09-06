#include "vits.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================
 * UTF-8 helpers
 * ============================================================ */

/* Return the number of bytes in a UTF-8 sequence given its first byte */
static int utf8_char_len(unsigned char c)
{
    if ((c & 0x80) == 0x00) return 1;   /* 0xxxxxxx */
    if ((c & 0xE0) == 0xC0) return 2;   /* 110xxxxx */
    if ((c & 0xF0) == 0xE0) return 3;   /* 1110xxxx */
    if ((c & 0xF8) == 0xF0) return 4;   /* 11110xxx */
    return 1; /* invalid, treat as single byte */
}

/* Lowercase a single UTF-8 character in-place.
 * Handles ASCII A-Z and 2-byte UTF-8 Latin-1 uppercase (U+00C0-U+00DE).
 * For 2-byte sequences in the 0xC3 range (U+0080-U+00BF), uppercase
 * occupies second-byte 0x80-0x9F and lowercase 0xA0-0xBF (offset +32). */
static void utf8_tolower_inplace(unsigned char *p, int len)
{
    if (len == 1) {
        if (p[0] >= 'A' && p[0] <= 'Z')
            p[0] += 32;
    } else if (len == 2) {
        if (p[0] == 0xC3 && p[1] >= 0x80 && p[1] <= 0x9F)
            p[1] += 32;
    }
    /* 3- and 4-byte characters: no lowercasing needed for this vocab */
}

/* ============================================================
 * vocab_init
 * ============================================================ */

void vocab_init(Vocab *v)
{
    v->vocab_size = VOCAB_SIZE;

    /* Initialize id_map to -1 (no mapping) */
    for (int i = 0; i < 128; i++)
        v->id_map[i] = -1;

    /* Vocabulary for mms-tts-por: id -> UTF-8 character */
    static const char *chars[VOCAB_SIZE] = {
        [0]  = "\xC3\xA0",   /* à  */
        [1]  = "\xC3\xBA",   /* ú  */
        [2]  = "1",
        [3]  = "u",
        [4]  = "l",
        [5]  = "2",
        [6]  = "h",
        [7]  = "\xC3\xA9",   /* é  */
        [8]  = "p",
        [9]  = "\xC3\xA3",   /* ã  */
        [10] = "x",
        [11] = "'",
        [12] = "\xC3\xA8",   /* ê  */
        [13] = "_",
        [14] = "s",
        [15] = "\xC3\xA7",   /* ç  */
        [16] = "4",
        [17] = "v",
        [18] = "m",
        [19] = "-",
        [20] = "g",
        [21] = "q",
        [22] = "c",
        [23] = "z",
        [24] = "\xC3\xA2",   /* â  */
        [25] = "\xC3\xAD",   /* í  */
        [26] = "t",
        [27] = "e",
        [28] = "o",
        [29] = "i",
        [30] = "f",
        [31] = "b",
        [32] = " ",
        [33] = "r",
        [34] = "\xC3\xB4",   /* ô  */
        [35] = "n",
        [36] = "\xE2\x80\x94", /* — (em dash) */
        [37] = "\xC3\xB3",   /* ó  */
        [38] = "a",
        [39] = "j",
        [40] = "d",
        [41] = "\xC3\xB5",   /* õ  */
        [42] = "\xC3\xA1",   /* á  */
    };

    for (int i = 0; i < VOCAB_SIZE; i++) {
        strncpy(v->chars[i], chars[i], 7);
        v->chars[i][7] = '\0';

        /* Build fast ASCII lookup: if the char is a single ASCII byte,
         * record the mapping in id_map */
        if ((unsigned char)chars[i][0] < 128)
            v->id_map[(unsigned char)chars[i][0]] = i;
    }
}

/* ============================================================
 * vocab_char_to_id
 * ============================================================ */

int vocab_char_to_id(const Vocab *v, const char *utf8_char)
{
    for (int i = 0; i < v->vocab_size; i++) {
        if (strcmp(v->chars[i], utf8_char) == 0)
            return i;
    }
    return -1;
}

/* ============================================================
 * tokenize
 *
 * Pipeline (matches HuggingFace VitsTokenizer):
 *   1. Lowercase the input text (in a temp buffer)
 *   2. Filter: collect IDs of characters in vocab (skip others)
 *   3. Strip: remove leading/trailing spaces (id 32)
 *   4. Emit: [pad, id0, pad, id1, ..., pad, idN, pad]
 *
 * Output: [pad, id0, pad, id1, pad, id2, ..., pad, idN, pad]
 * Max length: 2 * text_len + 1
 * ============================================================ */

void tokenize(const Vocab *v, const char *text, int32_t *ids, int *n_ids)
{
    int text_len = (int)strlen(text);

    /* Copy text into a mutable buffer for in-place lowercasing */
    unsigned char *buf = (unsigned char *)malloc(text_len + 1);
    memcpy(buf, text, text_len + 1);

    /* Pass 1: lowercase all UTF-8 characters */
    int pos = 0;
    while (pos < text_len) {
        int clen = utf8_char_len(buf[pos]);
        utf8_tolower_inplace(&buf[pos], clen);
        pos += clen;
    }

    /* Pass 2: collect in-vocab character IDs */
    int32_t *char_ids = (int32_t *)malloc(sizeof(int32_t) * (text_len + 1));
    int n_chars = 0;

    pos = 0;
    while (pos < text_len) {
        int clen = utf8_char_len(buf[pos]);

        /* Extract this character as a null-terminated string */
        char cbuf[8];
        memcpy(cbuf, &buf[pos], (size_t)clen);
        cbuf[clen] = '\0';

        int id = vocab_char_to_id(v, cbuf);
        if (id >= 0)
            char_ids[n_chars++] = id;
        /* else: character not in vocab, skip it */

        pos += clen;
    }

    /* Pass 3: strip leading/trailing spaces (matches Python .strip()) */
    const int space_id = 32;
    int start = 0, end = n_chars;
    while (start < end && char_ids[start] == space_id) start++;
    while (end > start && char_ids[end - 1] == space_id) end--;
    n_chars = end - start;

    /* Pass 4: emit [pad, id0, pad, id1, ..., pad, idN, pad] */
    int n = 0;
    const int pad_id = 0;
    ids[n++] = pad_id;
    for (int i = 0; i < n_chars; i++) {
        ids[n++] = char_ids[start + i];
        ids[n++] = pad_id;
    }

    *n_ids = n;
    free(char_ids);
    free(buf);
}
