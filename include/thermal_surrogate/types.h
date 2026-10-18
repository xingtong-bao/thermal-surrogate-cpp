// thermal_surrogate/types.h -- the C++ face of contract/contract.md
//
// Everything a caller needs to hand a sample to the library and read the
// answer back. The shapes and field names mirror the contract one to one, so
// a reader can hold the two side by side. No ONNX Runtime types leak out of
// this header: callers never include onnxruntime headers.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace thermal_surrogate {

// ---- fixed geometry of the v1 model (contract section 2) -------------------
// One sample is a dense float32 tensor (X, Y, Z, P) in C order: P varies
// fastest, then Z, then Y, then X.
constexpr std::size_t kX = 64;
constexpr std::size_t kY = 64;
constexpr std::size_t kZ = 1;
constexpr std::size_t kP = 7;                                   // channels
constexpr std::size_t kCells = kX * kY * kZ;                    // 4096
constexpr std::size_t kInputSize = kCells * kP;                 // 28672 floats
constexpr std::array<std::size_t, 4> kInputShape{kX, kY, kZ, kP};
constexpr std::array<std::size_t, 3> kOutputShape{kX, kY, kZ};

// Channel order is part of the contract; the index is the P coordinate.
enum class Channel : std::uint8_t {
  ChipletPower = 0,   // W, one value per chiplet painted over its footprint
  GridX = 1,          // mm
  GridY = 2,          // mm
  LocalThermalK = 3,  // W/(m*K)
  AmbientK = 4,       // K
  HW_m2K = 5,         // W/(m^2*K)
  RConvec_K_per_W = 6 // K/W
};
constexpr const char* kChannelNames[kP] = {
    "chiplet_power", "grid_x", "grid_y", "local_thermal_k",
    "ambient_K",     "h_w_m2k", "r_convec_k_per_w"};

// ---- Input: a non-owning view of the caller's buffer -------------------------
// The library never copies the input on the caller's behalf and never frees
// it. `data` must point to kInputSize floats, C-contiguous, physical units.
// `shape` is carried so the validator can report a wrong shape instead of
// reading out of bounds.
struct Input {
  const float* data = nullptr;
  std::array<std::size_t, 4> shape = kInputShape;   // (X, Y, Z, P)
};

// ---- Status: every failure mode of the contract (section 6) ------------------
enum class StatusCode : std::uint8_t {
  Ok = 0,
  InvalidInput,       // I1: null pointer, NaN/Inf, wrong dtype assumptions
  ShapeMismatch,      // I2
  ChannelMismatch,    // I3
  OutOfDistribution,  // I4 / I5: not an error; the dispatcher falls back
  ModelNotLoaded,     // schema/model version mismatch or missing file
  InferenceFailed,    // the runtime threw
  ReferenceFailed,    // the reference solver did not converge / crashed
  Timeout             // external reference backend exceeded its budget
};
const char* to_string(StatusCode code);                 // stable spellings = schema enum

struct Status {
  StatusCode code = StatusCode::Ok;
  std::string message;                                    // human-readable detail, may be empty
  bool ok() const { return code == StatusCode::Ok; }
};

// ---- Output: contract section 3 ----------------------------------------------
enum class Source : std::uint8_t { Surrogate = 0, ReferenceInProcess, ReferenceExternal, None };
const char* to_string(Source source);

struct Output {
  std::vector<float> field;              // kCells values, K, same (X, Y, Z) order as the input
  float tmax = 0.0f;                     // max(field)
  std::array<std::size_t, 2> tmax_index{0, 0};   // (x, y) of the max; informative only (contract)
  bool oob_flag = false;                 // any of I4 / I5 fired
  std::vector<std::string> oob_channels; // channel names, or "layout" for I5
  Source source = Source::None;          // which backend produced `field`
  double latency_ms = 0.0;               // wall time of the backend call
  std::string model_version;             // models/metadata.json -> model_version (surrogate only)
  // Conformal error bounds (contract section 3, S8.2): with probability 0.9 the
  // field MAE / |Tmax error| of a surrogate answer on an in-support input is
  // below these values. 0 when no table is loaded or the answer is not from
  // the surrogate. Calibrated on the validation split, see contract/conformal.json.
  float mae_bound_90_K = 0.0f;
  float tmax_err_bound_90_K = 0.0f;
  Status status;
};

// ---- library identity ---------------------------------------------------------
struct Version {
  const char* library;                   // this library, semantic version
  const char* onnxruntime;               // runtime the library was built against
};
Version version();

}  // namespace thermal_surrogate
