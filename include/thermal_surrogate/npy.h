// thermal_surrogate/npy.h -- minimal reader/writer for NumPy .npy files
//
// The fixtures (tests/fixtures/parity/*.npy) are plain .npy version 1.0
// files: little-endian float32, C order. Supporting exactly that subset keeps
// the reader at ~80 lines and avoids a dependency. Anything else (fortran
// order, other dtypes, v2/v3 headers) is rejected with a clear message
// rather than silently misread.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace thermal_surrogate::npy {

struct Array {
  std::vector<std::size_t> shape;
  std::vector<float> data;          // C order
  std::size_t size() const { return data.size(); }
};

// Throws std::runtime_error on I/O or format problems.
Array load_f32(const std::string& path);
void save_f32(const std::string& path, const std::vector<std::size_t>& shape, const float* data);

}  // namespace thermal_surrogate::npy
