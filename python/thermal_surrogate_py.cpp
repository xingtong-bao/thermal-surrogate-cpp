// python/thermal_surrogate_py.cpp -- S7.2: a thin Python binding over the C++ engine
//
//   import thermal_surrogate as ts
//   eng = ts.Engine("models", "contract")
//   out = eng.predict(x)        # x: numpy float32 (64, 64, 1, 7), C-contiguous
//   out["field"], out["tmax"], out["source"], out["oob_flag"], out["status"]
//
// The binding owns the same objects the C ABI owns (validator, predictor,
// FD fallback, dispatcher) and exposes one call. NumPy arrays are passed by
// buffer protocol: no copy on the way in, one copy on the way out.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"

namespace py = pybind11;
namespace ts = thermal_surrogate;

class Engine {
 public:
  Engine(const std::string& models, const std::string& contract)
      : validator_(contract + "/stats.json", models + "/config_bank.npy"),
        predictor_(models + "/model.onnx", models + "/metadata.json") {
    std::vector<std::unique_ptr<ts::ISolverBackend>> chain;
    chain.push_back(std::make_unique<ts::SurrogateBackend>(predictor_));
    chain.push_back(std::make_unique<ts::FdSolverBackend>(ts::load_fd_params(models + "/fd_solver_params.json")));
    dispatcher_ = std::make_unique<ts::Dispatcher>(validator_, std::move(chain));
    try { dispatcher_->set_conformal(ts::ConformalTable(contract + "/conformal.json")); } catch (const std::exception&) {}
  }

  py::dict predict(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    if (x.ndim() != 4) throw py::value_error("input must have shape (64, 64, 1, 7)");
    ts::Input in;
    in.data = x.data();
    for (int i = 0; i < 4; ++i) in.shape[static_cast<std::size_t>(i)] = static_cast<std::size_t>(x.shape(i));
    ts::Output out;
    {
      py::gil_scoped_release release;            // the runtime does not need the GIL
      dispatcher_->solve(in, out);
    }
    return to_dict(out);
  }

  static py::dict to_dict(const ts::Output& out) {
    py::dict d;
    d["status"] = ts::to_string(out.status.code);
    d["message"] = out.status.message;
    d["source"] = ts::to_string(out.source);
    d["oob_flag"] = out.oob_flag;
    d["oob_channels"] = out.oob_channels;
    d["latency_ms"] = out.latency_ms;
    d["mae_bound_90_K"] = out.mae_bound_90_K;
    d["tmax_err_bound_90_K"] = out.tmax_err_bound_90_K;
    d["model_version"] = out.model_version;
    if (out.status.ok()) {
      py::array_t<float> field({static_cast<py::ssize_t>(ts::kX), static_cast<py::ssize_t>(ts::kY), static_cast<py::ssize_t>(ts::kZ)});
      std::copy(out.field.begin(), out.field.end(), field.mutable_data());
      d["field"] = field;
      d["tmax"] = out.tmax;
      d["tmax_index"] = py::make_tuple(out.tmax_index[0], out.tmax_index[1]);
    }
    return d;
  }

  py::dict predict_power_scaled(py::array_t<float, py::array::c_style | py::array::forcecast> x, double factor) {
    if (x.ndim() != 4) throw py::value_error("input must have shape (64, 64, 1, 7)");
    ts::Input in;
    in.data = x.data();
    for (int i = 0; i < 4; ++i) in.shape[static_cast<std::size_t>(i)] = static_cast<std::size_t>(x.shape(i));
    ts::Output out;
    {
      py::gil_scoped_release release;
      dispatcher_->solve_power_scaled(in, factor, out);
    }
    return to_dict(out);
  }

  std::string model_version() const { return predictor_.model_version(); }

 private:
  ts::Validator validator_;
  ts::Predictor predictor_;
  std::unique_ptr<ts::Dispatcher> dispatcher_;
};

PYBIND11_MODULE(thermal_surrogate, m) {
  m.doc() = "thermal-surrogate-cpp: validated surrogate inference with physics fallback";
  m.attr("__version__") = ts::version().library;
  m.attr("onnxruntime_version") = ts::version().onnxruntime;
  py::class_<Engine>(m, "Engine")
      .def(py::init<const std::string&, const std::string&>(), py::arg("models_dir"), py::arg("contract_dir"))
      .def("predict", &Engine::predict, py::arg("input"), "Validate, predict or fall back; returns a dict")
      .def("predict_power_scaled", &Engine::predict_power_scaled, py::arg("input"), py::arg("factor"),
           "Power what-if by physics: exact field for all chiplet powers scaled by factor")
      .def_property_readonly("model_version", &Engine::model_version);
}
