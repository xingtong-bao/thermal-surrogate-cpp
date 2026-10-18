/* examples/c_caller.c -- a plain C program using the library through the C ABI
 *
 *   c_caller <model_dir> <contract_dir> <input.npy>
 *
 * Reads a contract tensor from a .npy file (minimal parser, float32 only),
 * predicts, prints Tmax, the source backend and the status. Built by CMake
 * as target `c_caller` with the C compiler, to prove the header is C-clean. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "thermal_surrogate/ts_c_api.h"

static float* load_npy_f32(const char* path, size_t expect) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char pre[10];
  if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
  size_t header_len = pre[8] | (pre[9] << 8);
  if (fseek(f, (long)header_len, SEEK_CUR) != 0) { fclose(f); return NULL; }
  float* buf = (float*)malloc(expect * sizeof(float));
  if (!buf || fread(buf, sizeof(float), expect, f) != expect) { free(buf); fclose(f); return NULL; }
  fclose(f);
  return buf;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: c_caller <model_dir> <contract_dir> <input.npy>\n"); return 2; }
  char msg[256] = {0};
  ts_engine* eng = ts_engine_create(argv[1], argv[2], msg, sizeof msg);
  if (!eng) { fprintf(stderr, "engine: %s\n", msg); return 1; }
  float* input = load_npy_f32(argv[3], 64 * 64 * 7);
  if (!input) { fprintf(stderr, "cannot read %s\n", argv[3]); ts_engine_destroy(eng); return 1; }
  float* field = (float*)malloc(64 * 64 * sizeof(float));
  ts_result r;
  ts_status st = ts_predict(eng, input, field, &r);
  printf("thermal_surrogate %s: status=%s source=%d oob=%d Tmax=%.2f K at (%zu,%zu) in %.1f ms %s\n",
         ts_version(), ts_status_name(st), r.source, r.oob_flag, r.tmax, r.tmax_x, r.tmax_y, r.latency_ms, r.model_version);
  free(field);
  free(input);
  ts_engine_destroy(eng);
  return st == TS_OK ? 0 : 1;
}
