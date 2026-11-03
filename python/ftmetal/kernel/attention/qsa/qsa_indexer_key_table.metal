// One thread per table entry: a request's selected blocks' token slots ascending, then its
// incomplete tail block's; entries past its count hold its last key's slot (never read).

const uint entry = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint topk = selected_shape[1];
const uint width = (topk + 1) * BLOCK_SIZE;
if (entry >= width) {
  return;
}
const uint seq_len = uint(seq_lens[row]);
const uint complete = seq_len / BLOCK_SIZE;
const uint num_selected = metal::min(complete, topk) * BLOCK_SIZE;
uint position;
if (entry < num_selected) {
  position = uint(selected[row * topk + entry / BLOCK_SIZE]) * BLOCK_SIZE + entry % BLOCK_SIZE;
} else {
  const uint last = metal::max(seq_len, 1u) - 1;
  position = metal::min(complete * BLOCK_SIZE + entry - num_selected, last);
}
const uint pages = page_table_shape[1];
key_slots[row * width + entry] =
    int(uint(page_table[row * pages + position / page_size]) * page_size +
        position % page_size);
if (entry == 0) {
  counts[row] = int(num_selected + seq_len - complete * BLOCK_SIZE);
}
