#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "colmap/mvs/gpu_mat_metal.h"

namespace colmap {
namespace mvs {

template <>
std::string TypeToString<int8_t>() {
  return "int8";
}

template <>
std::string TypeToString<int16_t>() {
  return "int16";
}

template <>
std::string TypeToString<int32_t>() {
  return "int32";
}

template <>
std::string TypeToString<int64_t>() {
  return "int64";
}

template <>
std::string TypeToString<float>() {
  return "float";
}

template <>
std::string TypeToString<double>() {
  return "double";
}

MetalKernelRunner::MetalKernelRunner(const std::string& kernel_name) {
  auto ns_name =
      NS::String::string(kernel_name.c_str(), NS::ASCIIStringEncoding);
  auto mtl_function = GetMetalLib()->newFunction(ns_name);
  MTL::ComputePipelineState* kernel;
  if (mtl_function) {
    NS::Error* error;
    kernel = GetDevice()->newComputePipelineState(mtl_function, &error);
  }

  if (!mtl_function || !kernel) {
    throw std::runtime_error("Failed to load kernel");
  }

  threads_per_thread_group_ = kernel->maxTotalThreadsPerThreadgroup();
  queue_ = GetDevice()->newCommandQueue();
  buffer_ = queue_->commandBuffer();
  enc_ = buffer_->computeCommandEncoder();
  enc_->setComputePipelineState(kernel);
}

MetalKernelRunner::~MetalKernelRunner() {
  enc_->release();
  buffer_->release();
  queue_->release();
}

void MetalKernelRunner::Run(const std::array<size_t, 3>& threads) {
  enc_->dispatchThreads(MTL::Size(threads[0], threads[1], threads[2]),
                        MTL::Size(threads_per_thread_group_, 1, 1));
  enc_->endEncoding();
  buffer_->commit();
  buffer_->waitUntilCompleted();
}

MTL::Device* GetDevice() {
  static MTL::Device* device;
  if (!device) {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("Failed to load Metal device");
    }
  }
  return device;
}

MTL::Library* GetMetalLib() {
  static MTL::Library* library;
  if (!library) {
    NS::Error* error;
    NS::String* path =
        NS::String::string(get_colocated_mtllib_path("metal_kernels").c_str(),
                           NS::UTF8StringEncoding);
    library = GetDevice()->newLibrary(path, &error);
    if (!library) {
      throw std::runtime_error("Failed to load metallib");
    }
  }
  return library;
}
}  // namespace mvs
}  // namespace colmap