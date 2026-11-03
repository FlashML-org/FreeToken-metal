// Row kernels behind ftmetal.kernel.inplace. One thread per element (or per
// 4-vector), rows along y. Buffer slots: 0 pool, 1 rows, 2 src_ids, 3 dst_ids,
// 4 lanes per row. The grid is exact (dispatch_threads), so no bounds checks.
#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void scatter_rows(
    device T* pool [[buffer(0)]],
    device const T* rows [[buffer(1)]],
    device const int* dst_ids [[buffer(3)]],
    constant const int& lanes [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  pool[size_t(dst_ids[tid.y]) * lanes + tid.x] = rows[size_t(tid.y) * lanes + tid.x];
}

template <typename T>
[[kernel]] void copy_rows(
    device T* pool [[buffer(0)]],
    device const int* src_ids [[buffer(2)]],
    device const int* dst_ids [[buffer(3)]],
    constant const int& lanes [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  pool[size_t(dst_ids[tid.y]) * lanes + tid.x] = pool[size_t(src_ids[tid.y]) * lanes + tid.x];
}

template <typename T>
[[kernel]] void zero_rows(
    device T* pool [[buffer(0)]],
    device const int* ids [[buffer(3)]],
    constant const int& lanes [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  pool[size_t(ids[tid.y]) * lanes + tid.x] = T(0);
}

#define FTMETAL_ROWS_INSTANTIATE(T, NAME)                                         \
  template [[host_name("scatter_rows_" #NAME)]] [[kernel]] void scatter_rows<T>( \
      device T*, device const T*, device const int*, constant const int&, uint2); \
  template [[host_name("scatter_rows_vec_" #NAME)]] [[kernel]] void               \
  scatter_rows<vec<T, 4>>(                                                        \
      device vec<T, 4>*, device const vec<T, 4>*, device const int*,              \
      constant const int&, uint2);                                                \
  template [[host_name("copy_rows_" #NAME)]] [[kernel]] void copy_rows<T>(       \
      device T*, device const int*, device const int*, constant const int&,       \
      uint2);                                                                     \
  template [[host_name("copy_rows_vec_" #NAME)]] [[kernel]] void                 \
  copy_rows<vec<T, 4>>(                                                           \
      device vec<T, 4>*, device const int*, device const int*,                    \
      constant const int&, uint2);                                                \
  template [[host_name("zero_rows_" #NAME)]] [[kernel]] void zero_rows<T>(       \
      device T*, device const int*, constant const int&, uint2);                  \
  template [[host_name("zero_rows_vec_" #NAME)]] [[kernel]] void                 \
  zero_rows<vec<T, 4>>(                                                           \
      device vec<T, 4>*, device const int*, constant const int&, uint2);

FTMETAL_ROWS_INSTANTIATE(half, half)
FTMETAL_ROWS_INSTANTIATE(bfloat, bfloat)
FTMETAL_ROWS_INSTANTIATE(float, float)
FTMETAL_ROWS_INSTANTIATE(int, int)
FTMETAL_ROWS_INSTANTIATE(uint, uint)
