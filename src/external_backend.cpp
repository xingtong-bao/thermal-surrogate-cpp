// src/external_backend.cpp -- see external_backend.h. POSIX only (fork/exec + waitpid polling).
#include "thermal_surrogate/external_backend.h"

#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "thermal_surrogate/npy.h"

namespace thermal_surrogate {
namespace {

// Runs argv with a deadline. Returns {exit_code, timed_out}. -1 exit = could not start.
std::pair<int, bool> run_with_timeout(const std::vector<std::string>& argv, double timeout_s) {
  std::vector<char*> cargv;
  for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
  cargv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) return {-1, false};
  if (pid == 0) {
    execvp(cargv[0], cargv.data());
    _exit(127);                                       // exec failed
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  int status = 0;
  while (true) {
    const pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) break;
    if (r < 0) return {-1, false};
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return {-1, true};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, false};
}

}  // namespace

void ExternalReferenceBackend::solve(const Input& in, Output& out) noexcept {
  out = Output{};
  out.source = Source::ReferenceExternal;
  if (in.data == nullptr || in.shape != kInputShape) { out.status = {StatusCode::InvalidInput, "bad input"}; return; }
  if (opt_.command.empty()) { out.status = {StatusCode::ReferenceFailed, "no command configured"}; return; }
  const auto t0 = std::chrono::steady_clock::now();
  std::string dir;
  try {
    std::string tmpl = (opt_.workdir.empty() ? std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") : opt_.workdir) + "/ts_ext_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (mkdtemp(buf.data()) == nullptr) { out.status = {StatusCode::ReferenceFailed, "cannot create temp dir"}; return; }
    dir = buf.data();
    const std::string in_path = dir + "/input.npy", out_path = dir + "/output.npy";
    npy::save_f32(in_path, {kX, kY, kZ, kP}, in.data);

    auto argv = opt_.command;
    argv.push_back(in_path);
    argv.push_back(out_path);
    const auto [code, timed_out] = run_with_timeout(argv, opt_.timeout_s);
    if (timed_out) { out.status = {StatusCode::Timeout, "external reference exceeded " + std::to_string(opt_.timeout_s) + " s"}; }
    else if (code != 0) { out.status = {StatusCode::ReferenceFailed, "external reference exited with code " + std::to_string(code)}; }
    else {
      auto field = npy::load_f32(out_path);
      if (field.size() != kCells) { out.status = {StatusCode::ReferenceFailed, "external reference wrote " + std::to_string(field.size()) + " values, expected 4096"}; }
      else {
        out.field = std::move(field.data);
        std::size_t best = 0;
        for (std::size_t i = 1; i < kCells; ++i) if (out.field[i] > out.field[best]) best = i;
        out.tmax = out.field[best];
        out.tmax_index = {best / kY, best % kY};
        out.status = {StatusCode::Ok, ""};
      }
    }
  } catch (const std::exception& e) {
    out.status = {StatusCode::ReferenceFailed, e.what()};
  }
  if (!opt_.keep_files && !dir.empty()) {
    unlink((dir + "/input.npy").c_str());
    unlink((dir + "/output.npy").c_str());
    rmdir(dir.c_str());
  }
  out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!out.status.ok()) out.field.clear();
}

}  // namespace thermal_surrogate
