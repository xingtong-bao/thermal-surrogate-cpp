// bench/machine_info.h -- what machine produced a benchmark number
//
// Every results file under docs/results that contains a timing carries this
// block, because a latency without the CPU it ran on is not a measurement.
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace bench {

inline std::string sysctl_str(const char* name) {
#if defined(__APPLE__)
  char buf[256];
  std::size_t len = sizeof buf;
  if (sysctlbyname(name, buf, &len, nullptr, 0) == 0) return std::string(buf, len > 0 ? len - 1 : 0);
#else
  (void)name;
#endif
  return "unknown";
}

inline long long sysctl_int(const char* name) {
#if defined(__APPLE__)
  long long v = 0;
  std::size_t len = sizeof v;
  if (sysctlbyname(name, &v, &len, nullptr, 0) == 0) return v;
#else
  (void)name;
#endif
  return 0;
}

inline nlohmann::json machine_info() {
  nlohmann::json m;
#if defined(__APPLE__)
  m["os"] = "macOS";
  m["model"] = sysctl_str("hw.model");
  m["cpu"] = sysctl_str("machdep.cpu.brand_string");
  m["memory_gb"] = static_cast<double>(sysctl_int("hw.memsize")) / (1024.0 * 1024.0 * 1024.0);
  m["performance_cores"] = sysctl_int("hw.perflevel0.physicalcpu");
  m["efficiency_cores"] = sysctl_int("hw.perflevel1.physicalcpu");
#elif defined(__linux__)
  m["os"] = "Linux";
#else
  m["os"] = "other";
#endif
  m["hardware_threads"] = std::thread::hardware_concurrency();
  m["compiler"] =
#if defined(__clang__)
      std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
      std::string("gcc ") + __VERSION__;
#else
      "unknown";
#endif
  m["build_type"] =
#ifdef NDEBUG
      "Release";
#else
      "Debug";
#endif
  return m;
}

}  // namespace bench
