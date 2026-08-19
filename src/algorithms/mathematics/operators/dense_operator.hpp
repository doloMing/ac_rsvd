#pragma once

#include "ac_rsvd/matrix_operator.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"

namespace ac_rsvd {
namespace math {

// A direct matrix operator for dense inputs.
class DenseOperator : public MatrixOperator {
public:
    explicit DenseOperator(const Matrix& matrix);

    int rows() const override;
    int cols() const override;

    void apply(const double* x, int block_cols, double* y) const override;
    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override;

private:
    Matrix matrix_;
};

}  // namespace math
}  // namespace ac_rsvd
