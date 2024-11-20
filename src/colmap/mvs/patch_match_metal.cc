#include "colmap/mvs/patch_match_metal.h"

namespace colmap {
namespace mvs {

PatchMatchMetal::PatchMatchMetal(const PatchMatchOptions& options,
                                 const PatchMatch::Problem& problem)
    : options_(options), problem_(problem) {}

void PatchMatchMetal::Run() {}

DepthMap PatchMatchMetal::GetDepthMap() const {
  return DepthMap(
      depth_map_->CopyToMat(), options_.depth_min, options_.depth_max);
}

NormalMap PatchMatchMetal::GetNormalMap() const {
  return NormalMap(normal_map_->CopyToMat());
}

Mat<float> PatchMatchMetal::GetSelProbMap() const {
  return prev_sel_prob_map_->CopyToMat();
}

std::vector<int> PatchMatchMetal::GetConsistentImageIdxs() const {
  const Mat<uint8_t> mask = consistency_mask_->CopyToMat();
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(mask.GetDepth());
  for (size_t r = 0; r < mask.GetHeight(); ++r) {
    for (size_t c = 0; c < mask.GetWidth(); ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t d = 0; d < mask.GetDepth(); ++d) {
        if (mask.Get(r, c, d)) {
          pixel_consistent_image_idxs.push_back(problem_.src_image_idxs[d]);
        }
      }
      if (pixel_consistent_image_idxs.size() > 0) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

}  // namespace mvs
}  // namespace colmap