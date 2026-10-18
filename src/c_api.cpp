// src/c_api.cpp -- the C ABI over the C++ objects (see ts_c_api.h)
#include "thermal_surrogate/ts_c_api.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"

namespace ts = thermal_surrogate;

// The handle bundles everything a prediction needs; declaration order is
// construction order and the reverse is destruction order (dispatcher holds
// references into the others, so it goes last and dies first).
struct ts_engine {
  ts::Validator validator;
  ts::Predictor predictor;
  std::unique_ptr<ts::Dispatcher> dispatcher;
  ts_engine(const std::string& models, const std::string& contract)
      : validator(contract + "/stats.json", models + "/config_bank.npy"),
        predictor(models + "/model.onnx", models + "/metadata.json") {
    std::vector<std::unique_ptr<ts::ISolverBackend>> chain;
    chain.push_back(std::make_unique<ts::SurrogateBackend>(predictor));
    chain.push_back(std::make_unique<ts::FdSolverBackend>(ts::load_fd_params(models + "/fd_solver_params.json")));
    dispatcher = std::make_unique<ts::Dispatcher>(validator, std::move(chain));
  }
};

namespace {
void copy_str(char* dst, std::size_t len, const std::string& s) {
  if (!dst || len == 0) return;
  std::strncpy(dst, s.c_str(), len - 1);
  dst[len - 1] = '\0';
}
ts_status map_code(ts::StatusCode c) { return static_cast<ts_status>(static_cast<int>(c)); }   // enums are aligned by design
}  // namespace

extern "C" {

ts_engine* ts_engine_create(const char* model_dir, const char* contract_dir, char* message, std::size_t message_len) {
  if (!model_dir || !contract_dir) { copy_str(message, message_len, "null directory"); return nullptr; }
  try {
    return new ts_engine(model_dir, contract_dir);
  } catch (const std::exception& e) {
    copy_str(message, message_len, e.what());
    return nullptr;
  }
}

void ts_engine_destroy(ts_engine* engine) { delete engine; }

ts_status ts_predict(const ts_engine* engine, const float* input, float* field, ts_result* result) {
  if (!engine || !input || !field) return TS_INVALID_ARGUMENT;
  ts::Output out;
  engine->dispatcher->solve(ts::Input{input, ts::kInputShape}, out);
  if (result) {
    *result = ts_result{};
    result->tmax = out.tmax;
    result->tmax_x = out.tmax_index[0];
    result->tmax_y = out.tmax_index[1];
    result->oob_flag = out.oob_flag ? 1 : 0;
    result->source = static_cast<int>(out.source) + (out.source == ts::Source::None ? 0 : 1);   // None=0, Surrogate=1, ...
    if (out.source == ts::Source::None) result->source = TS_SOURCE_NONE;
    result->latency_ms = out.latency_ms;
    copy_str(result->model_version, sizeof result->model_version, out.model_version);
    copy_str(result->message, sizeof result->message, out.status.message);
  }
  if (out.status.ok()) std::memcpy(field, out.field.data(), ts::kCells * sizeof(float));
  return map_code(out.status.code);
}

const char* ts_version(void) { return ts::version().library; }

const char* ts_status_name(ts_status code) {
  if (code == TS_INVALID_ARGUMENT) return "InvalidArgument";
  return ts::to_string(static_cast<ts::StatusCode>(static_cast<int>(code)));
}

}  // extern "C"
