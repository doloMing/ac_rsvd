#include "algorithms/mathematics/operators/dense_operator.hpp"

#include <algorithm>

namespace ac_rsvd {
namespace math {

DenseOperator::DenseOperator(const Matrix& matrix) : matrix_(matrix) {}

int DenseOperator::rows() const {
    return matrix_.rows();
}

int DenseOperator::cols() const {
    return matrix_.cols();
}

void DenseOperator::apply(const double* x, int block_cols, double* y) const {
    // Compute every column of Y = A X in column-major order.
    std::fill(y, y + rows() * block_cols, 0.0);
    for (int col = 0; col < block_cols; ++col) {
        for (int inner = 0; inner < cols(); ++inner) {
            double value = x[inner + col * cols()];
            for (int row = 0; row < rows(); ++row) {
                y[row + col * rows()] += matrix_(row, inner) * value;
            }
        }
    }
}

void DenseOperator::apply_transpose(
    const double* y,
    int block_cols,
    double* x) const {
    // The transpose path computes X = A^T Y with the same layout.
    std::fill(x, x + cols() * block_cols, 0.0);
    for (int col = 0; col < block_cols; ++col) {
        for (int inner = 0; inner < rows(); ++inner) {
            double value = y[inner + col * rows()];
            for (int row = 0; row < cols(); ++row) {
                x[row + col * cols()] += matrix_(inner, row) * value;
            }
        }
    }
}

}  // namespace math
}  // namespace ac_rsvd
