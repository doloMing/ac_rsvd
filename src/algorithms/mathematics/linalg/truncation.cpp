#include "algorithms/mathematics/linalg/truncation.hpp"

namespace ac_rsvd {
namespace math {

std::vector<double> squared_tail_sums(const std::vector<double>& singular_values) {
    int count = static_cast<int>(singular_values.size());
    std::vector<double> tail(count + 1, 0.0);
    // Summing backward gives every Frobenius tail in one pass.
    for (int index = count - 1; index >= 0; --index) {
        double value = singular_values[index];
        tail[index] = tail[index + 1] + value * value;
    }
    return tail;
}

int smallest_rank_for_tail(
    const std::vector<double>& singular_values,
    double squared_error_budget) {
    std::vector<double> tail = squared_tail_sums(singular_values);
    for (int rank = 0; rank <= static_cast<int>(singular_values.size()); ++rank) {
        if (tail[rank] <= squared_error_budget) {
            return rank;
        }
    }
    return static_cast<int>(singular_values.size());
}

}  // namespace math
}  // namespace ac_rsvd
