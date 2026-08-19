#pragma once

#include <vector>

namespace ac_rsvd {
namespace math {

// tail[k] is the squared error after keeping k singular values.
std::vector<double> squared_tail_sums(const std::vector<double>& singular_values);

int smallest_rank_for_tail(
    const std::vector<double>& singular_values,
    double squared_error_budget);

}  // namespace math
}  // namespace ac_rsvd
