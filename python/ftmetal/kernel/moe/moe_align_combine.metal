// Grid (H / V, T): each thread gathers V columns of a token's K expert rows from their
// sorted positions and accumulates the routing-weighted sum in fp32.

const uint token = thread_position_in_grid.y;
const uint column = thread_position_in_grid.x * V;
float acc[V] = {};
for (int j = 0; j < K; ++j) {
  const size_t row = pair_positions[token * K + j];
  const float weight = static_cast<float>(topk_weights[token * K + j]);
  auto source = expert_out + row * H + column;
  for (int v = 0; v < V; ++v) {
    acc[v] += weight * static_cast<float>(source[v]);
  }
}
auto target = out + (size_t)token * H + column;
for (int v = 0; v < V; ++v) {
  target[v] = static_cast<T>(acc[v]);
}
