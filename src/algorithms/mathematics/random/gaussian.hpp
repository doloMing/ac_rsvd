#pragma once

#include <cstdint>

namespace ac_rsvd::math {

double gaussian_value(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t direction,
    std::uint64_t row);

// The output matrix is column-major.
void fill_gaussian(
    double* values,
    int rows,
    int cols,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_direction);

}  // namespace ac_rsvd::math
