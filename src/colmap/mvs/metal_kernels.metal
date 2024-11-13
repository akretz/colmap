// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define instantiate_all_kernels(name, func)       \
  instantiate_kernel(name "_int8", func, int8_t)  \
  instantiate_kernel(name "_int16", func, int16_t) \
  instantiate_kernel(name "_int32", func, int32_t) \
  instantiate_kernel(name "_int64", func, int64_t) \
  instantiate_kernel(name "_float", func, float)


template <typename T>
[[kernel]] void fill_with_vector(
    device T* src [[buffer(0)]],
    const constant T* vec [[buffer(1)]],
    const constant uint3& dims [[buffer(2)]],
    uint3 index [[thread_position_in_grid]]) {
  src[index.z * dims.x * dims.y + index.y * dims.x + index.x] = vec[index.z];
}

// Generate pseudo-random 32-bit integers using LCG from Numerical Recipes
template <typename T>
[[kernel]] void uniform_random(
    device T* src [[buffer(0)]],
    device uint32_t* rand_state [[buffer(1)]],
    const constant uint3& dims [[buffer(2)]],
    const constant T& min [[buffer(3)]],
    const constant T& max [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]) {
  uint32_t rand = rand_state[index.y * dims.x + index.x];
  // float a = float(max) - float(min);
  for (uint i = 0; i < dims.z; i++) {
    // IEEE 754 float has 23 fraction bits
    // float x = float(rand >> 9) / float((1 << 23) - 1);
    float x = float(rand & (1 << 23) - 1) / float((1 << 23) - 1);
    src[i * dims.x * dims.y + index.y * dims.x + index.x] = x * (max - min) + min; //T(a * x - float(min));
    rand *= 1664525U;
    rand += 1013904223U;
  }
  rand_state[index.y * dims.x + index.x] = rand;
}

template <typename T>
[[kernel]] void transpose(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint3& dims [[buffer(2)]],
    uint3 index [[thread_position_in_grid]]) {
  dst[index.z * dims.x * dims.y + index.y * dims.x + index.x] = src[index.z * dims.x * dims.y + index.x * dims.y + index.y];
}

template <typename T>
[[kernel]] void flip_horizontal(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint3& dims [[buffer(2)]],
    uint3 index [[thread_position_in_grid]]) {
  dst[index.z * dims.x * dims.y + index.y * dims.x + index.x] = src[index.z * dims.x * dims.y + index.y * dims.x + (dims.x - 1 - index.x)];
}

template <typename T>
[[kernel]] void rotate(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const constant uint3& dims [[buffer(2)]],
    uint3 index [[thread_position_in_grid]]) {
  dst[index.z * dims.x * dims.y + index.y * dims.x + index.x] = src[index.z * dims.x * dims.y + index.x * dims.y + (dims.y - 1 - index.y)];
}

instantiate_all_kernels("fill_with_vector", fill_with_vector)
instantiate_all_kernels("uniform_random", uniform_random)
instantiate_all_kernels("transpose", transpose)
instantiate_all_kernels("flip_horizontal", flip_horizontal)
instantiate_all_kernels("rotate", rotate)
