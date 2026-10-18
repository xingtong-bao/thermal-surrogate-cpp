// src/status.cpp -- stable spellings for the enums (they must match contract/schema.json)
#include "thermal_surrogate/types.h"

namespace thermal_surrogate {

const char* to_string(StatusCode code) {
  switch (code) {
    case StatusCode::Ok: return "Ok";
    case StatusCode::InvalidInput: return "InvalidInput";
    case StatusCode::ShapeMismatch: return "ShapeMismatch";
    case StatusCode::ChannelMismatch: return "ChannelMismatch";
    case StatusCode::OutOfDistribution: return "OutOfDistribution";
    case StatusCode::ModelNotLoaded: return "ModelNotLoaded";
    case StatusCode::InferenceFailed: return "InferenceFailed";
    case StatusCode::ReferenceFailed: return "ReferenceFailed";
    case StatusCode::Timeout: return "Timeout";
  }
  return "Unknown";
}

const char* to_string(Source source) {
  switch (source) {
    case Source::Surrogate: return "surrogate";
    case Source::ReferenceInProcess: return "reference_inprocess";
    case Source::ReferenceExternal: return "reference_external";
    case Source::None: return "none";
  }
  return "unknown";
}

}  // namespace thermal_surrogate
