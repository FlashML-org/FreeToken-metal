// qsa_indexer_topk's order-preserving key of a score: uint keys compare as the scores do, -0.0
// taken as +0.0 (MLX's comparisons tie them; the raw bits would order -0.0 below +0.0).
inline uint score_order_key(float score) {
  const uint raw = as_type<uint>(score);
  const uint bits = raw == 0x80000000u ? 0u : raw;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
