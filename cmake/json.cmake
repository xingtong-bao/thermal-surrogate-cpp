# cmake/json.cmake -- nlohmann/json (single header, MIT), pinned by tag
# Used to read models/metadata.json and contract/stats.json. Header-only, so
# it adds nothing to the link line and nothing to the public ABI.
include(FetchContent)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)
