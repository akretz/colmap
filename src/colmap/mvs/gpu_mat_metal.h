#pragma once

#include <filesystem>

#include <Metal/Metal.hpp>
#include <dlfcn.h>

namespace colmap {
namespace mvs {

MTL::Device* GetDevice();
MTL::Library* GetMetalLib();

template <typename T>
std::string TypeToString();

// from mlx/backend/metal/device.cpp in the mlx repo
inline std::string get_colocated_mtllib_path(const std::string& lib_name) {
  Dl_info info;
  std::string mtllib_path;
  std::string lib_ext = lib_name + ".metallib";

  int success = dladdr((void*)get_colocated_mtllib_path, &info);
  if (success) {
    auto mtllib =
        std::filesystem::path(info.dli_fname).remove_filename() / lib_ext;
    mtllib_path = mtllib.c_str();
  }

  return mtllib_path;
}

template <typename T>
class GpuMat {
 public:
  GpuMat(uint32_t width, uint32_t height, uint32_t depth = 1);
  ~GpuMat();

  MTL::Buffer* GetBuffer() const;

  void FillWithScalar(T value);
  void FillWithVector(const T* values);
  void FillWithRandomNumbers(T min_value,
                             T max_value,
                             const GpuMat<uint32_t>& random_state);

  void CopyToDevice(const T* data, uint32_t pitch);
  void CopyToHost(T* data, uint32_t pitch) const;
  // Mat<T> CopyToMat() const;

  // Transpose array by swapping x and y coordinates.
  void Transpose(GpuMat<T>* output);

  // Flip array along vertical axis.
  void FlipHorizontal(GpuMat<T>* output);

  // Rotate array in counter-clockwise direction.
  void Rotate(GpuMat<T>* output);

 private:
  MTL::Buffer* buf_;
  uint32_t width_;
  uint32_t height_;
  uint32_t depth_;
};

class MetalKernelRunner {
 public:
  explicit MetalKernelRunner(const std::string& kernel_name);
  ~MetalKernelRunner();

  template <typename T>
  void SetGpuMat(const GpuMat<T>& gpu_mat, int index);

  template <typename T>
  void SetParameter(const T& val, int index, int num_elements = 1);

  void Run(const std::array<size_t, 3>& threads);

 private:
  MTL::CommandQueue* queue_;
  MTL::CommandBuffer* buffer_;
  MTL::ComputeCommandEncoder* enc_;
  int threads_per_thread_group_;
};

template <typename T>
void MetalKernelRunner::SetGpuMat(const GpuMat<T>& gpu_mat, int index) {
  enc_->setBuffer(gpu_mat.GetBuffer(), 0, index);
}

template <typename T>
void MetalKernelRunner::SetParameter(const T& val,
                                     int index,
                                     int num_elements) {
  enc_->setBytes(&val, num_elements * sizeof(T), index);
}

template <typename T>
GpuMat<T>::GpuMat(uint32_t width, uint32_t height, uint32_t depth)
    : width_(width), height_(height), depth_(depth) {
  size_t res_opt =
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;
  buf_ = GetDevice()->newBuffer(width * height * depth * sizeof(T), res_opt);
}

template <typename T>
GpuMat<T>::~GpuMat() {
  buf_->release();
}

template <typename T>
MTL::Buffer* GpuMat<T>::GetBuffer() const {
  return buf_;
}

template <typename T>
void GpuMat<T>::FillWithScalar(T value) {}

template <typename T>
void GpuMat<T>::FillWithVector(const T* values) {
  MetalKernelRunner runner("fill_with_vector_" + TypeToString<T>());
  const uint32_t dims[3] = {width_, height_, depth_};
  runner.SetGpuMat(*this, 0);
  GpuMat<T> vec(depth_, 1);
  vec.CopyToDevice(values, 0);
  runner.SetGpuMat(vec, 1);
  runner.SetParameter(dims, 2);
  runner.Run({width_, height_, depth_});
}

template <typename T, typename D>
std::vector<T> GenerateRandomNumbers(D dist, uint32_t width, uint32_t height) {}

template <typename T>
void GpuMat<T>::FillWithRandomNumbers(T min_value,
                                      T max_value,
                                      const GpuMat<uint32_t>& random_state) {
  MetalKernelRunner runner("uniform_random_" + TypeToString<T>());
  const uint32_t dims[3] = {width_, height_, depth_};
  runner.SetGpuMat(*this, 0);
  runner.SetGpuMat(random_state, 1);
  runner.SetParameter(dims, 2);
  runner.SetParameter(min_value, 3);
  runner.SetParameter(max_value, 4);
  runner.Run({width_, height_, 1});
}

template <typename T>
void GpuMat<T>::CopyToDevice(const T* data, uint32_t pitch) {
  T* ptr = static_cast<T*>(buf_->contents());
  for (size_t i = 0; i < depth_; ++i) {
    for (size_t j = 0; j < height_; ++j) {
      std::copy(data, data + width_, ptr);
      ptr += width_;
      data += pitch / sizeof(T);
    }
  }
}

template <typename T>
void GpuMat<T>::CopyToHost(T* data, uint32_t pitch) const {
  const T* ptr = static_cast<T*>(buf_->contents());
  for (size_t i = 0; i < depth_; ++i) {
    for (size_t j = 0; j < height_; ++j) {
      std::copy(ptr, ptr + width_, data);
      ptr += width_;
      data += pitch / sizeof(T);
    }
  }
}

template <typename T>
void GpuMat<T>::Transpose(GpuMat<T>* output) {
  MetalKernelRunner runner("transpose_" + TypeToString<T>());
  const uint32_t dims[3] = {height_, width_, depth_};
  runner.SetGpuMat(*this, 0);
  runner.SetGpuMat(*output, 1);
  runner.SetParameter(dims, 2);
  runner.Run({height_, width_, depth_});
}

template <typename T>
void GpuMat<T>::FlipHorizontal(GpuMat<T>* output) {
  MetalKernelRunner runner("flip_horizontal_" + TypeToString<T>());
  const uint32_t dims[3] = {width_, height_, depth_};
  runner.SetGpuMat(*this, 0);
  runner.SetGpuMat(*output, 1);
  runner.SetParameter(dims, 2);
  runner.Run({width_, height_, depth_});
}

template <typename T>
void GpuMat<T>::Rotate(GpuMat<T>* output) {
  MetalKernelRunner runner("rotate_" + TypeToString<T>());
  const uint32_t dims[3] = {height_, width_, depth_};
  runner.SetGpuMat(*this, 0);
  runner.SetGpuMat(*output, 1);
  runner.SetParameter(dims, 2);
  runner.Run({height_, width_, depth_});
}

}  // namespace mvs
}  // namespace colmap
