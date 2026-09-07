#include "vits.h"
#include "../model_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/* ============================================================
 * VTSM binary loader
 *
 * Reads a flat binary file produced by export_weights.py.
 * All tensors are pre-processed:
 *   - weight_norm is already fused
 *   - ConvTranspose weights are already transposed to [out,in,k]
 *
 * File layout:
 *   [0..3]   magic "VTSM"
 *   [4..7]   version (uint32 LE)
 *   [8..11]  vocab_size (uint32 LE)
 *   [12..15] hidden_size (uint32 LE)
 *   [16..23] total_data_size bytes (uint64 LE)
 *   [24..31] reserved (zero)
 *   [32..]   sequential float32 data
 * ============================================================ */

/* Get a direct pointer into the mmap'd region (zero-copy) */
static inline float *ptr_at(const unsigned char *base, uint64_t offset)
{
    return (float *)(base + offset);
}

int load_vtsm(const char *path, VitsModel *model, Vocab *vocab)
{
    (void)vocab;

    /* Open and mmap the file (file-backed, reclaimable by OS) */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[ERROR] cannot open '%s'\n", path);
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "[ERROR] cannot stat '%s'\n", path);
        close(fd);
        return -1;
    }

    size_t fsize = (size_t)st.st_size;
    if (fsize < WTS_FILE_HEADER_SIZE) {
        fprintf(stderr, "[ERROR] file too small (%zu bytes)\n", fsize);
        close(fd);
        return -1;
    }

    const unsigned char *buf = (const unsigned char *)mmap(
        NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "[ERROR] mmap failed for '%s'\n", path);
        return -1;
    }

    /* Validate header */
    uint32_t magic, version, vocab_size, hidden_size;
    uint64_t total_bytes;
    memcpy(&magic, buf + 0, 4);
    memcpy(&version, buf + 4, 4);
    memcpy(&vocab_size, buf + 8, 4);
    memcpy(&hidden_size, buf + 12, 4);
    memcpy(&total_bytes, buf + 16, 8);

    if (magic != WTS_MAGIC) {
        fprintf(stderr, "[ERROR] bad magic: 0x%08x (expected 0x%08x)\n",
                magic, WTS_MAGIC);
        munmap((void *)buf, fsize);
        return -1;
    }
    if (version != WTS_VERSION) {
        fprintf(stderr, "[ERROR] unsupported version: %u\n", version);
        munmap((void *)buf, fsize);
        return -1;
    }
    if (vocab_size != WTS_VOCAB_SIZE) {
        fprintf(stderr, "[ERROR] vocab size mismatch: %u (expected %d)\n",
                vocab_size, WTS_VOCAB_SIZE);
        munmap((void *)buf, fsize);
        return -1;
    }
    if (hidden_size != WTS_HIDDEN_SIZE) {
        fprintf(stderr, "[ERROR] hidden size mismatch: %u (expected %d)\n",
                hidden_size, WTS_HIDDEN_SIZE);
        munmap((void *)buf, fsize);
        return -1;
    }
    if (total_bytes != WTS_TOTAL_DATA_BYTES) {
        fprintf(stderr, "[ERROR] data size mismatch: %lu (expected %lu)\n",
                (unsigned long)total_bytes,
                (unsigned long)WTS_TOTAL_DATA_BYTES);
        munmap((void *)buf, fsize);
        return -1;
    }
    if (fsize != WTS_FILE_HEADER_SIZE + (size_t)WTS_TOTAL_DATA_BYTES) {
        fprintf(stderr, "[ERROR] file size %zu != expected %lu\n", fsize,
                (unsigned long)(WTS_FILE_HEADER_SIZE + WTS_TOTAL_DATA_BYTES));
        munmap((void *)buf, fsize);
        return -1;
    }

    model->sampling_rate = SAMPLE_RATE;
    model->vtsm_map = buf;
    model->vtsm_map_size = fsize;

    /* ============================================================
     * Text Encoder
     * ============================================================ */

    /* Embedding */
    model->embed_w = ptr_at(buf, wts_tensors[0].offset);

    /* Encoder layers */
    for (int l = 0; l < NUM_LAYERS; l++) {
        EncLayer *el = &model->layers[l];
        int base = 1 + l * 18;  /* tensor index in wts_tensors */

        /* Attention projections */
        el->attn.q_w = ptr_at(buf, wts_tensors[base + 0].offset);
        el->attn.q_b = ptr_at(buf, wts_tensors[base + 1].offset);
        el->attn.k_w = ptr_at(buf, wts_tensors[base + 2].offset);
        el->attn.k_b = ptr_at(buf, wts_tensors[base + 3].offset);
        el->attn.v_w = ptr_at(buf, wts_tensors[base + 4].offset);
        el->attn.v_b = ptr_at(buf, wts_tensors[base + 5].offset);
        el->attn.o_w = ptr_at(buf, wts_tensors[base + 6].offset);
        el->attn.o_b = ptr_at(buf, wts_tensors[base + 7].offset);

        /* Relative positions */
        el->attn.rel_k = ptr_at(buf, wts_tensors[base + 8].offset);
        el->attn.rel_v = ptr_at(buf, wts_tensors[base + 9].offset);

        /* FFN */
        el->ffn1_w = ptr_at(buf, wts_tensors[base + 10].offset);
        el->ffn1_b = ptr_at(buf, wts_tensors[base + 11].offset);
        el->ffn2_w = ptr_at(buf, wts_tensors[base + 12].offset);
        el->ffn2_b = ptr_at(buf, wts_tensors[base + 13].offset);

        /* Layer norms */
        el->ln1.dim = HIDDEN;
        el->ln2.dim = HIDDEN;
        el->ln1.weight = ptr_at(buf, wts_tensors[base + 14].offset);
        el->ln1.bias = ptr_at(buf, wts_tensors[base + 15].offset);
        el->ln2.weight = ptr_at(buf, wts_tensors[base + 16].offset);
        el->ln2.bias = ptr_at(buf, wts_tensors[base + 17].offset);
    }

    /* Encoder projection */
    {
        int base = 1 + NUM_LAYERS * 18;  /* tensor index 109 */
        model->proj_w = ptr_at(buf, wts_tensors[base + 0].offset);
        model->proj_b = ptr_at(buf, wts_tensors[base + 1].offset);
    }

    /* ============================================================
     * Duration Predictor
     * ============================================================ */
    {
        int base = 111;  /* dp.main section starts at tensor 111 */

        /* conv_pre */
        model->dp.conv_pre_w = ptr_at(buf, wts_tensors[base + 0].offset);
        model->dp.conv_pre_b = ptr_at(buf, wts_tensors[base + 1].offset);

        /* DDS (3 layers × 8 tensors = 24) */
        DDS *dds = &model->dp.conv_dds;
        dds->num_layers = DP_DDS_LAYERS;
        dds->dim = HIDDEN;
        dds->kernel = DP_KERNEL;
        for (int i = 0; i < DP_DDS_LAYERS; i++) {
            int o = base + 2 + i * 8;
            dds->dw_w[i] = ptr_at(buf, wts_tensors[o + 0].offset);
            dds->dw_b[i] = ptr_at(buf, wts_tensors[o + 1].offset);
            dds->pw_w[i] = ptr_at(buf, wts_tensors[o + 2].offset);
            dds->pw_b[i] = ptr_at(buf, wts_tensors[o + 3].offset);
            dds->norm1[i].dim = HIDDEN;
            dds->norm1[i].weight = ptr_at(buf, wts_tensors[o + 4].offset);
            dds->norm1[i].bias = ptr_at(buf, wts_tensors[o + 5].offset);
            dds->norm2[i].dim = HIDDEN;
            dds->norm2[i].weight = ptr_at(buf, wts_tensors[o + 6].offset);
            dds->norm2[i].bias = ptr_at(buf, wts_tensors[o + 7].offset);
        }

        /* conv_proj */
        int proj_idx = base + 2 + DP_DDS_LAYERS * 8;  /* +24 */
        model->dp.conv_proj_w = ptr_at(buf, wts_tensors[proj_idx + 0].offset);
        model->dp.conv_proj_b = ptr_at(buf, wts_tensors[proj_idx + 1].offset);

        /* Flow 0: ElementwiseAffine */
        int ea_idx = proj_idx + 2;  /* +26 */
        model->dp.flow_0.translate = ptr_at(buf, wts_tensors[ea_idx + 0].offset);
        model->dp.flow_0.log_scale = ptr_at(buf, wts_tensors[ea_idx + 1].offset);
    }

    /* DP Flows 1..4 (ConvFlow) */
    for (int fi = 1; fi <= DP_NUM_FLOWS; fi++) {
        ConvFlow *cf = &model->dp.flows[fi - 1];
        int base = 141 + (fi - 1) * 28;  /* dp.flow.1 starts at 141 */

        /* conv_pre: Conv1d(1, 192, 1) → [192, 1, 1] */
        cf->conv_pre_w = ptr_at(buf, wts_tensors[base + 0].offset);
        cf->conv_pre_b = ptr_at(buf, wts_tensors[base + 1].offset);

        /* DDS */
        DDS *dds = &cf->dds;
        dds->num_layers = DP_DDS_LAYERS;
        dds->dim = HIDDEN;
        dds->kernel = DP_KERNEL;
        for (int i = 0; i < DP_DDS_LAYERS; i++) {
            int o = base + 2 + i * 8;
            dds->dw_w[i] = ptr_at(buf, wts_tensors[o + 0].offset);
            dds->dw_b[i] = ptr_at(buf, wts_tensors[o + 1].offset);
            dds->pw_w[i] = ptr_at(buf, wts_tensors[o + 2].offset);
            dds->pw_b[i] = ptr_at(buf, wts_tensors[o + 3].offset);
            dds->norm1[i].dim = HIDDEN;
            dds->norm1[i].weight = ptr_at(buf, wts_tensors[o + 4].offset);
            dds->norm1[i].bias = ptr_at(buf, wts_tensors[o + 5].offset);
            dds->norm2[i].dim = HIDDEN;
            dds->norm2[i].weight = ptr_at(buf, wts_tensors[o + 6].offset);
            dds->norm2[i].bias = ptr_at(buf, wts_tensors[o + 7].offset);
        }

        /* conv_proj: Conv1d(192, 29, 1) */
        int proj_idx = base + 2 + DP_DDS_LAYERS * 8;  /* +26 */
        cf->conv_proj_w = ptr_at(buf, wts_tensors[proj_idx + 0].offset);
        cf->conv_proj_b = ptr_at(buf, wts_tensors[proj_idx + 1].offset);
    }

    /* ============================================================
     * Flow (4 coupling layers)
     * ============================================================ */
    model->flow.num_flows = NUM_PRIOR_FLOWS;

    for (int fi = 0; fi < NUM_PRIOR_FLOWS; fi++) {
        CouplingLayer *cl = &model->flow.flows[fi];
        int base = 253 + fi * 20;  /* flow.0 starts at 253 */

        /* conv_pre */
        cl->conv_pre_w = ptr_at(buf, wts_tensors[base + 0].offset);
        cl->conv_pre_b = ptr_at(buf, wts_tensors[base + 1].offset);

        /* WaveNet */
        WaveNet *wn = &cl->wavenet;
        wn->num_layers = NUM_WAVE;
        for (int i = 0; i < NUM_WAVE; i++) {
            int rs = (i == NUM_WAVE - 1) ? HIDDEN : (2 * HIDDEN);
            wn->rs_out[i] = rs;

            int o = base + 2 + i * 4;
            /* in_layers: weight already fused */
            wn->in_w[i] = ptr_at(buf, wts_tensors[o + 0].offset);
            wn->in_b[i] = ptr_at(buf, wts_tensors[o + 1].offset);
            /* res_skip_layers */
            wn->rs_w[i] = ptr_at(buf, wts_tensors[o + 2].offset);
            wn->rs_b[i] = ptr_at(buf, wts_tensors[o + 3].offset);
        }

        /* conv_post */
        int post_idx = base + 2 + NUM_WAVE * 4;  /* +18 */
        cl->conv_post_w = ptr_at(buf, wts_tensors[post_idx + 0].offset);
        cl->conv_post_b = ptr_at(buf, wts_tensors[post_idx + 1].offset);
    }

    /* ============================================================
     * HiFi-GAN Decoder
     * ============================================================ */
    {
        int base = 333;  /* decoder section starts at 333 */

        /* conv_pre: Conv1d(192, 512, k=7) */
        model->decoder.conv_pre.in_ch = HIDDEN;
        model->decoder.conv_pre.out_ch = HIFI_INIT_CH;
        model->decoder.conv_pre.k = 7;
        model->decoder.conv_pre.pad = 3;
        model->decoder.conv_pre.dilation = 1;
        model->decoder.conv_pre.weight = ptr_at(buf, wts_tensors[base + 0].offset);
        model->decoder.conv_pre.bias = ptr_at(buf, wts_tensors[base + 1].offset);

        /* Upsamplers: ConvTranspose1d (already transposed in binary) */
        static const int up_in[]  = {512, 256, 128, 64};
        static const int up_out[] = {256, 128, 64, 32};
        static const int up_k[]   = {16, 16, 4, 4};
        static const int up_s[]   = {8, 8, 2, 2};
        static const int up_p[]   = {4, 4, 1, 1};

        for (int i = 0; i < NUM_UP; i++) {
            int o = base + 2 + i * 2;
            model->decoder.up[i].in_ch = up_in[i];
            model->decoder.up[i].out_ch = up_out[i];
            model->decoder.up[i].k = up_k[i];
            model->decoder.up[i].stride = up_s[i];
            model->decoder.up[i].pad = up_p[i];
            model->decoder.up[i].weight = ptr_at(buf, wts_tensors[o + 0].offset);
            model->decoder.up[i].bias = ptr_at(buf, wts_tensors[o + 1].offset);
        }

        /* ResBlocks: 4 stages × 3 kernels */
        static const int rb_ch[] = {256, 128, 64, 32};
        static const int rb_k[]  = {3, 7, 11};
        static const int rb_dil[] = {1, 3, 5};

        for (int stage = 0; stage < 4; stage++) {
            for (int kb = 0; kb < 3; kb++) {
                ResBlock *rb = &model->decoder.rb[stage * 3 + kb];
                rb->ch = rb_ch[stage];
                rb->kernel = rb_k[kb];
                for (int d = 0; d < RF_DILS; d++)
                    rb->dil[d] = rb_dil[d];

                /* Each ResBlock has convs1[3] and convs2[3], each with weight+bias
                 * In the export plan: c1.0.w, c1.0.b, c1.1.w, c1.1.b, c1.2.w, c1.2.b,
                 *                    c2.0.w, c2.0.b, c2.1.w, c2.1.b, c2.2.w, c2.2.b
                 * = 12 tensors per ResBlock
                 */
                int rb_idx = stage * 3 + kb;
                int o = base + 2 + NUM_UP * 2 + rb_idx * 12;

                for (int d = 0; d < RF_DILS; d++) {
                    int k = rb_k[kb];
                    int ch = rb_ch[stage];
                    int dil = rb_dil[d];

                    /* Layout per dilation: c1.w, c1.b, c2.w, c2.b */
                    rb->c1[d].in_ch = ch;
                    rb->c1[d].out_ch = ch;
                    rb->c1[d].k = k;
                    rb->c1[d].pad = dil * (k - 1) / 2;
                    rb->c1[d].dilation = dil;
                    rb->c1[d].weight = ptr_at(buf, wts_tensors[o + d * 4 + 0].offset);
                    rb->c1[d].bias = ptr_at(buf, wts_tensors[o + d * 4 + 1].offset);

                    rb->c2[d].in_ch = ch;
                    rb->c2[d].out_ch = ch;
                    rb->c2[d].k = k;
                    rb->c2[d].pad = (k - 1) / 2;
                    rb->c2[d].dilation = 1;
                    rb->c2[d].weight = ptr_at(buf, wts_tensors[o + d * 4 + 2].offset);
                    rb->c2[d].bias = ptr_at(buf, wts_tensors[o + d * 4 + 3].offset);
                }
            }
        }

        /* conv_post: Conv1d(32, 1, k=7, no bias) */
        int cp_idx = base + 2 + NUM_UP * 2 + 12 * 12;  /* last tensor */
        {
            const int cp_k = 7;
            model->decoder.conv_post.in_ch = rb_ch[3];
            model->decoder.conv_post.out_ch = 1;
            model->decoder.conv_post.k = cp_k;
            model->decoder.conv_post.pad = (cp_k - 1) / 2;
            model->decoder.conv_post.dilation = 1;
            model->decoder.conv_post.weight = ptr_at(buf, wts_tensors[cp_idx].offset);
            model->decoder.conv_post.bias = NULL;
        }
    }

    return 0;
}

/* ============================================================
 * Release the mmap'd weight file. All pointer fields point into
 * this mapping, so a single munmap frees all weight data.
 * Call before free(model) to release the file-backed pages.
 * ============================================================ */
void free_model(VitsModel *model)
{
    if (!model) return;
    if (model->use_int8_hifi)
        hifigan_free_q(&model->decoder_q);
    if (model->vtsm_map) {
        munmap((void *)model->vtsm_map, model->vtsm_map_size);
        model->vtsm_map = NULL;
        model->vtsm_map_size = 0;
    }
}
