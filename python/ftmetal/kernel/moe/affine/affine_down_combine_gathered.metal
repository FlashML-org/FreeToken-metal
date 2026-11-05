// affine_down_combine's tail over gathered down rows y [T, top_k, 1, N]: thread (token, row)
// multiplies each pair's row by its routing weight in the dtype, as (y * weights[..., None]) does,
// and sums the products as (...).sum(axis=-2) sums them (routed_sum); {{OUTPUT}} is the sum or its
// sigmoid_gated_add.
const uint index = thread_position_in_grid.x;
const int token = index / N;
const int row = index % N;
T products[TOP_K];
for (int j = 0; j < TOP_K; j++) {
  products[j] = y[size_t(token * TOP_K + j) * N + row] * weights[token * TOP_K + j];
}
const T total = routed_sum<T, TOP_K, 1>(products);
out[index] = {{OUTPUT}};
