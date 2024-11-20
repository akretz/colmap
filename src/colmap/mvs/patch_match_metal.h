#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/gpu_mat_metal.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"

namespace colmap {
namespace mvs {

class PatchMatchMetal {
 public:
  PatchMatchMetal(const PatchMatchOptions& options,
                  const PatchMatch::Problem& problem);

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Data for reference image.
  std::unique_ptr<GpuMat<float>> depth_map_;
  std::unique_ptr<GpuMat<float>> normal_map_;
  std::unique_ptr<GpuMat<float>> prev_sel_prob_map_;
  std::unique_ptr<GpuMat<uint8_t>> consistency_mask_;
};

}  // namespace mvs
}  // namespace colmap