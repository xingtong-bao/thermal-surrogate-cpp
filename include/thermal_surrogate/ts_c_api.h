/* thermal_surrogate/ts_c_api.h -- C ABI for callers that cannot use the C++ headers
 *
 * Why a C layer: tool integrations usually load this library at run time
 * from a different compiler or language (Python ctypes, a C host, a plugin
 * loader). A C ABI has no name mangling, no std:: types across the boundary
 * and no exceptions. The C++ API stays the richer one; this file is a thin
 * opaque-handle wrapper over it (see docs/integration_design.md).
 *
 * Rules
 *   - every function returns a ts_status code; 0 is TS_OK;
 *   - nothing here throws; a C++ exception inside is converted to a code;
 *   - the caller owns every buffer it passes; the library never frees them;
 *   - handles must be destroyed by the thread that created them, after all
 *     predictions have returned; ts_predict itself is thread-safe per handle.
 */
#ifndef THERMAL_SURROGATE_TS_C_API_H
#define THERMAL_SURROGATE_TS_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TS_API_VERSION 1

typedef struct ts_engine ts_engine;      /* opaque: validator + predictor + reference + dispatcher */

typedef enum ts_status {
  TS_OK = 0,
  TS_INVALID_INPUT = 1,
  TS_SHAPE_MISMATCH = 2,
  TS_CHANNEL_MISMATCH = 3,
  TS_OUT_OF_DISTRIBUTION = 4,
  TS_MODEL_NOT_LOADED = 5,
  TS_INFERENCE_FAILED = 6,
  TS_REFERENCE_FAILED = 7,
  TS_TIMEOUT = 8,
  TS_INVALID_ARGUMENT = 100                 /* null handle or buffer passed to the C API */
} ts_status;

typedef enum ts_source { TS_SOURCE_NONE = 0, TS_SOURCE_SURROGATE = 1, TS_SOURCE_REFERENCE_INPROCESS = 2, TS_SOURCE_REFERENCE_EXTERNAL = 3 } ts_source;

/* Scalar results of one prediction; the field goes to the caller's buffer. */
typedef struct ts_result {
  float tmax;                /* K */
  size_t tmax_x, tmax_y;     /* informative only (contract section 3) */
  int oob_flag;              /* 1 if I4/I5 fired and a reference answered */
  int source;                /* ts_source */
  double latency_ms;
  char model_version[32];    /* empty unless source == surrogate */
  char message[160];         /* status detail, empty on TS_OK */
} ts_result;

/* Creates an engine from the repository layout: <model_dir>/model.onnx,
 * metadata.json, config_bank.npy, fd_solver_params.json and
 * <contract_dir>/stats.json. Returns NULL on failure (details in *message
 * if given). Backends: surrogate, then the in-process FD reference. */
ts_engine* ts_engine_create(const char* model_dir, const char* contract_dir, char* message, size_t message_len);
void ts_engine_destroy(ts_engine* engine);

/* input: 64*64*1*7 floats, C order, physical units. field: 64*64 floats, K. */
ts_status ts_predict(const ts_engine* engine, const float* input, float* field, ts_result* result);

const char* ts_version(void);              /* library version string */
const char* ts_status_name(ts_status code);

#ifdef __cplusplus
}
#endif
#endif
