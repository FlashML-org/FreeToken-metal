// One thread per (token, column): the window's keys summed in order in fp32, times 1 / WINDOW,
// rounded to the keys' dtype.

const uint column = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
if (column >= D) {
  return;
}
float total = 0.0f;
for (uint i = 0; i < WINDOW; ++i) {
  const size_t slot = size_t(window_slots[token * WINDOW + i]);
  total += static_cast<float>(index_keys[slot * D + column]);
}
means[token * D + column] = static_cast<T>(total * (1.0f / float(WINDOW)));
