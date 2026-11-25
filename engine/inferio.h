#ifndef INFERIO_H
#define INFERIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERIO_VERSION "v0.1.0"

typedef enum {
    INFERIO_OK = 0,
    INFERIO_ERR_OOM = 1,
    INFERIO_ERR_CUDA = 2,
    INFERIO_ERR_ARG = 3,
    INFERIO_ERR_IO = 4
} inferio_rc;

typedef enum {
    INFERIO_PHASE_EMBED = 0,
    INFERIO_PHASE_LN = 1,
    INFERIO_PHASE_QKV = 2,
    INFERIO_PHASE_ATTN = 3,
    INFERIO_PHASE_PROJ = 4,
    INFERIO_PHASE_FC1 = 5,
    INFERIO_PHASE_FC2 = 6,
    INFERIO_PHASE_HEAD = 7,
    INFERIO_PHASE_ARGMAX = 8,
    INFERIO_PHASE_COUNT = 9
} inferio_phase;

typedef struct {
    double prefill_ms;   /* CUDA-event time: ids upload -> last-position fp32 logits */
    long long peak_bytes;   /* engine-tracked allocation high-water mark */
    long long weight_bytes; /* total fp16 weights on device */
    int64_t batch;
    int64_t seqlen;
    int64_t group_seqs; /* internal sequence-group size used */
    int64_t n_groups;
    int rc;
    double phase_ms[INFERIO_PHASE_COUNT]; /* filled only when INFERIO_PROFILE=1 */
} inferio_stats;

/* gpu_index selects the device within CUDA_VISIBLE_DEVICES. */
int inferio_init(const char *model_dir, int gpu_index);
int inferio_prefill(const int32_t *host_ids, int batch, int seqlen, inferio_stats *st);
int inferio_last_argmax(int32_t *host_out, int batch);
int inferio_copy_last_logits(float *host_out, int batch, int vocab);
const char *inferio_last_error(void);
const char *inferio_version(void);
void inferio_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
