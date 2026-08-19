#pragma once

#include <cstdint>
#include <vector>

#include "ac_rsvd/matrix_operator.hpp"

namespace ac_rsvd {
namespace math {

// A = U diag(sigma) V^T with U = P_u H D_u and V = P_v H D_v.
// Philox streams 0, 1, 2, and 3 draw the two signs and permutations.
class StructuredHadamardOperator : public MatrixOperator {
public:
    StructuredHadamardOperator(
        const std::vector<double>& singular_values,
        std::uint64_t seed);

    int rows() const override;
    int cols() const override;

    void apply(const double* x, int block_cols, double* y) const override;
    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override;

    const std::vector<double>& singular_values() const;
    double frobenius_norm() const;

private:
    int size_;
    std::vector<double> singular_values_;
    std::vector<double> left_signs_;
    std::vector<double> right_signs_;
    std::vector<int> left_permutation_;
    std::vector<int> right_permutation_;
    double frobenius_norm_;
};

}  // namespace math
}  // namespace ac_rsvd
