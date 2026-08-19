#pragma once

#include <cstdint>

namespace ac_rsvd::math {

// A coordinate is fixed by its seed, stream, direction, and row.
double gaussian_value(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t direction,
    std::uint64_t row);

// Fill a column-major panel without depending on earlier draws.
void fill_gaussian(
    double* values,
    int rows,
    int cols,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_direction);

}  // namespace ac_rsvd::math
