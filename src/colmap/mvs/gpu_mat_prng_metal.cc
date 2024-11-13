#include "colmap/mvs/gpu_mat_prng_metal.h"

#include <limits>
#include <random>

namespace colmap {
namespace mvs {

GpuMatPRNG::GpuMatPRNG(int width, int height) : GpuMat(width, height) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<uint32_t> dist(
      0, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> random_numbers;
  int n = width * height;
  random_numbers.reserve(n);
  for (int i = 0; i < n; ++i) {
    random_numbers.push_back(dist(rng));
  }
  CopyToDevice(random_numbers.data(), width);
}

}  // namespace mvs
}  // namespace colmap
