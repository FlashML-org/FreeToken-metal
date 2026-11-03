using frag = simdgroup_matrix<float, 8, 8>;

// An 8x8 fp32 fragment from a row-major device matrix of any dtype; transpose loads
// the block's transpose.
template <typename T>
inline frag load_frag(const device T* src, int stride, bool transpose) {
  simdgroup_matrix<T, 8, 8> raw;
  simdgroup_load(raw, src, stride, ulong2(0, 0), transpose);
  frag out;
  out.thread_elements()[0] = static_cast<float>(raw.thread_elements()[0]);
  out.thread_elements()[1] = static_cast<float>(raw.thread_elements()[1]);
  return out;
}
