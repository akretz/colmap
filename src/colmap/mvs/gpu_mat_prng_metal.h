#pragma once

#include "colmap/mvs/gpu_mat_metal.h"

namespace colmap {
namespace mvs {

class GpuMatPRNG : public GpuMat<uint32_t> {
 public:
  GpuMatPRNG(int width, int height);
};

}  // namespace mvs
}  // namespace colmap
