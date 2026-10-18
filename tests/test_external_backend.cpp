// tests/test_external_backend.cpp -- S5.3: the subprocess protocol, exit codes and the timeout
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "thermal_surrogate/external_backend.h"
#include "thermal_surrogate/npy.h"

using namespace thermal_surrogate;

namespace {
const std::string kStub = std::string(TS_FIXTURE_DIR) + "/external_stub.py";
ExternalReferenceBackend make(double timeout_s = 10.0) {
  ExternalBackendOptions o;
  o.command = {"python3", kStub};
  o.timeout_s = timeout_s;
  return ExternalReferenceBackend(o);
}
npy::Array fixture() { return npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input.npy"); }
}  // namespace

TEST(ExternalBackend, RoundTripThroughFiles) {
  setenv("EXT_STUB", "ok", 1);
  auto b = make();
  const auto x = fixture();
  Output out;
  b.solve(Input{x.data.data(), kInputShape}, out);
  ASSERT_TRUE(out.status.ok()) << out.status.message;
  EXPECT_EQ(out.source, Source::ReferenceExternal);
  EXPECT_EQ(out.field.size(), kCells);
  EXPECT_FLOAT_EQ(out.field[123], 300.0f);
  EXPECT_FLOAT_EQ(out.tmax, 300.0f);
  EXPECT_GT(out.latency_ms, 0.0);
}

TEST(ExternalBackend, NonZeroExitIsReferenceFailed) {
  setenv("EXT_STUB", "fail", 1);
  auto b = make();
  const auto x = fixture();
  Output out;
  b.solve(Input{x.data.data(), kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::ReferenceFailed);
  EXPECT_TRUE(out.field.empty());
}

TEST(ExternalBackend, WrongSizeOutputIsReferenceFailed) {
  setenv("EXT_STUB", "short", 1);
  auto b = make();
  const auto x = fixture();
  Output out;
  b.solve(Input{x.data.data(), kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::ReferenceFailed);
}

TEST(ExternalBackend, HungProcessIsKilledOnTimeout) {
  setenv("EXT_STUB", "hang", 1);
  auto b = make(0.5);                               // the stub sleeps 30 s
  const auto x = fixture();
  Output out;
  b.solve(Input{x.data.data(), kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::Timeout);
  EXPECT_LT(out.latency_ms, 5000.0);                // came back at the deadline, not after 30 s
  setenv("EXT_STUB", "ok", 1);
}

TEST(ExternalBackend, MissingCommandIsReferenceFailed) {
  ExternalBackendOptions o;
  o.command = {"/nonexistent/solver"};
  ExternalReferenceBackend b(o);
  const auto x = fixture();
  Output out;
  b.solve(Input{x.data.data(), kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::ReferenceFailed);
}
