#pragma once

#include <array>
#include <cstdint>

namespace ac_rsvd::math {

// Apply the ten-round Philox4x32 permutation to one counter.
std::array<std::uint32_t, 4> philox4x32(
    std::array<std::uint32_t, 4> counter,
    std::array<std::uint32_t, 2> key);

}  // namespace ac_rsvd::math
