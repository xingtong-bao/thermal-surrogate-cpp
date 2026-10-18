// src/npy.cpp -- see npy.h. Format reference: numpy/lib/format.py.
//   magic "\x93NUMPY", u8 major, u8 minor, u16 LE header_len (v1.0),
//   header_len bytes of ASCII dict, padded with spaces, ending in '\n',
//   then raw data.
#include "thermal_surrogate/npy.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace thermal_surrogate::npy {
namespace {

[[noreturn]] void fail(const std::string& path, const std::string& why) {
  throw std::runtime_error("npy: " + path + ": " + why);
}

// Pull the value of a key out of the header dict without a real parser:
// the dict is produced by NumPy itself and always has the form
//   {'descr': '<f4', 'fortran_order': False, 'shape': (64, 64, 1, 7), }
std::string field(const std::string& header, const std::string& key) {
  const auto k = header.find("'" + key + "'");
  if (k == std::string::npos) return {};
  const auto colon = header.find(':', k);
  auto start = header.find_first_not_of(" ", colon + 1);
  std::size_t end;
  if (header[start] == '(') {
    end = header.find(')', start) + 1;
  } else if (header[start] == '\'') {
    end = header.find('\'', start + 1) + 1;
  } else {
    end = header.find_first_of(",}", start);
  }
  return header.substr(start, end - start);
}

std::vector<std::size_t> parse_shape(std::string tuple) {
  // "(64, 64, 1, 7)" or "(4096,)" or "()"
  std::vector<std::size_t> shape;
  for (auto& c : tuple) if (c == '(' || c == ')' || c == ',') c = ' ';
  std::istringstream ss(tuple);
  std::size_t v;
  while (ss >> v) shape.push_back(v);
  return shape;
}

}  // namespace

Array load_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail(path, "cannot open");
  char magic[6];
  f.read(magic, 6);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) fail(path, "not a .npy file");
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  if (ver[0] != 1) fail(path, "only .npy version 1.x is supported");
  unsigned char len[2];
  f.read(reinterpret_cast<char*>(len), 2);
  const std::size_t header_len = static_cast<std::size_t>(len[0]) | (static_cast<std::size_t>(len[1]) << 8);
  std::string header(header_len, '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!f) fail(path, "truncated header");

  if (field(header, "descr") != "'<f4'") fail(path, "dtype must be little-endian float32 ('<f4'), got " + field(header, "descr"));
  if (field(header, "fortran_order") != "False") fail(path, "fortran_order must be False");
  Array a;
  a.shape = parse_shape(field(header, "shape"));
  std::size_t n = 1;
  for (auto d : a.shape) n *= d;
  a.data.resize(n);
  f.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) fail(path, "truncated data");
  return a;
}

void save_f32(const std::string& path, const std::vector<std::size_t>& shape, const float* data) {
  std::ostringstream dict;
  dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (std::size_t i = 0; i < shape.size(); ++i) dict << shape[i] << (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
  dict << "), }";
  std::string header = dict.str();
  // Total header (magic 6 + ver 2 + len 2 + dict + '\n') is padded to a multiple of 64.
  const std::size_t base = 10;
  std::size_t total = base + header.size() + 1;
  const std::size_t pad = (64 - total % 64) % 64;
  header.append(pad, ' ');
  header.push_back('\n');
  std::ofstream f(path, std::ios::binary);
  if (!f) fail(path, "cannot create");
  f.write("\x93NUMPY\x01\x00", 8);
  const auto hl = static_cast<std::uint16_t>(header.size());
  const unsigned char len[2] = {static_cast<unsigned char>(hl & 0xff), static_cast<unsigned char>(hl >> 8)};
  f.write(reinterpret_cast<const char*>(len), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::size_t n = 1;
  for (auto d : shape) n *= d;
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

}  // namespace thermal_surrogate::npy
