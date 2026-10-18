// tests/test_skeleton.cpp -- S4.1: the build links, the runtime loads, the model is the one we exported
#include <gtest/gtest.h>

#include <cstring>

#include "thermal_surrogate/model_info.h"
#include "thermal_surrogate/types.h"

using namespace thermal_surrogate;

TEST(Skeleton, VersionStringsArePresent) {
  const auto v = version();
  EXPECT_GT(std::strlen(v.library), 0u);
  EXPECT_STREQ(v.onnxruntime, "1.23.2");       // pinned in cmake/onnxruntime.cmake
}

TEST(Skeleton, StatusSpellingsMatchSchema) {
  // contract/schema.json $defs/status enum; a typo here would desynchronise C++ and JSON.
  EXPECT_STREQ(to_string(StatusCode::Ok), "Ok");
  EXPECT_STREQ(to_string(StatusCode::OutOfDistribution), "OutOfDistribution");
  EXPECT_STREQ(to_string(StatusCode::Timeout), "Timeout");
  EXPECT_STREQ(to_string(Source::Surrogate), "surrogate");
  EXPECT_STREQ(to_string(Source::ReferenceExternal), "reference_external");
}

TEST(Skeleton, ContractGeometry) {
  EXPECT_EQ(kInputSize, 64u * 64u * 1u * 7u);
  EXPECT_EQ(kCells, 4096u);
  EXPECT_STREQ(kChannelNames[static_cast<int>(Channel::AmbientK)], "ambient_K");
}

TEST(Skeleton, ModelLoadsAndMatchesContract) {
  // The loaded runtime must be the one we built against, and the exported
  // graph must expose exactly the contract tensors (models/metadata.json -> io).
  const auto info = inspect_model(TS_MODEL_PATH);
  EXPECT_EQ(info.onnxruntime_version, std::string(version().onnxruntime));
  ASSERT_EQ(info.inputs.size(), 1u);
  ASSERT_EQ(info.outputs.size(), 1u);
  EXPECT_EQ(info.inputs[0].name, "input");
  EXPECT_EQ(info.inputs[0].element_type, "float32");
  EXPECT_EQ(info.inputs[0].shape, (std::vector<std::int64_t>{1, 64, 64, 1, 7}));
  EXPECT_EQ(info.outputs[0].name, "field");
  EXPECT_EQ(info.outputs[0].shape, (std::vector<std::int64_t>{1, 64, 64, 1}));
}

TEST(Skeleton, MissingModelThrows) {
  EXPECT_THROW(inspect_model("/nonexistent/model.onnx"), std::runtime_error);
}
