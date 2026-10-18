// apps/ts_demo.cpp -- command-line front end
//
//   ts_demo --info [model.onnx]   print library/runtime versions and the model's IO
//
// Later stories add `--predict <input.npy>` and `--bench`. Keeping the demo a
// thin shell over the library is deliberate: anything worth doing twice goes
// into the library, not into main().
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "thermal_surrogate/model_info.h"
#include "thermal_surrogate/types.h"

namespace {

void print_tensors(const char* heading, const std::vector<thermal_surrogate::TensorInfo>& tensors) {
  std::cout << heading << "\n";
  for (const auto& t : tensors) {
    std::cout << "  " << t.name << "  " << t.element_type << "  (";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i) std::cout << ", ";
      if (t.shape[i] < 0) std::cout << "dyn"; else std::cout << t.shape[i];
    }
    std::cout << ")\n";
  }
}

int usage() {
  std::cerr << "usage: ts_demo --info [model.onnx]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--info") != 0) return usage();
  const std::string model = argc >= 3 ? argv[2] : TS_DEFAULT_MODEL_PATH;

  const auto v = thermal_surrogate::version();
  std::cout << "thermal_surrogate " << v.library << " (built against ONNX Runtime " << v.onnxruntime << ")\n";
  try {
    const auto info = thermal_surrogate::inspect_model(model);
    std::cout << "ONNX Runtime loaded: " << info.onnxruntime_version << "\n"
              << "model: " << info.path << "\n";
    print_tensors("inputs:", info.inputs);
    print_tensors("outputs:", info.outputs);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
