// One thread per (token, h): the mean over the streams of sigmoid(up) * normed.

const uint element = thread_position_in_grid.x;
const uint token = element / HIDDEN;
const uint h = element % HIDDEN;
float total = 0;
for (int stream = 0; stream < HC_COUNT; ++stream) {
  #pragma METAL fp contract(off)
  const size_t index = (size_t(token) * HC_COUNT + stream) * HIDDEN + h;
  total += precise_sigmoid(static_cast<float>(up[index])) *
      static_cast<float>(normed[index]);
}
mixed[element] = static_cast<T>(total / static_cast<float>(HC_COUNT));
