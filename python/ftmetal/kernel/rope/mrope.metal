const uint elem = thread_position_in_grid.x;
const int num_tokens = x_shape[0];
const int num_heads = x_shape[1];
const int head_dim = x_shape[2];
const int half_dim = ROTARY_DIM / 2;
// Each (token, head) row has half_dim rotating pairs, then its pass-through dims.
const int slots = head_dim - half_dim;
if (elem >= uint(num_tokens * num_heads * slots)) {
  return;
}
const int slot = int(elem) % slots;
const int row = int(elem) / slots;
const int token = row / num_heads;
const int base = row * head_dim;
if (slot >= half_dim) {
  const int d = ROTARY_DIM + slot - half_dim;
  out[base + d] = x[base + d];
  return;
}
const int axis = section[slot];
const float angle = static_cast<float>(positions[axis * num_tokens + token]) * inv_freq[slot];
const float c = metal::precise::cos(angle);
const float s = metal::precise::sin(angle);
const float x1 = static_cast<float>(x[base + slot]);
const float x2 = static_cast<float>(x[base + slot + half_dim]);
out[base + slot] = static_cast<T>(x1 * c - x2 * s);
out[base + slot + half_dim] = static_cast<T>(x2 * c + x1 * s);
