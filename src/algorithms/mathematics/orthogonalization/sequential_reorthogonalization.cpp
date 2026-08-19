#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"

#include <algorithm>
#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/projection.hpp"

namespace ac_rsvd {
namespace math {

BlockOrthogonalizationResult orthogonalize_sequential_block(
    const Matrix& block,
    const Matrix& basis,
    int projection_passes) {
    BlockOrthogonalizationResult result;
    result.projected = Matrix(block.rows(), block.cols());

    // Remove the old basis one column at a time.
    for (int col = 0; col < block.cols(); ++col) {
        Matrix column = copy_block(block, 0, col, block.rows(), 1);
        project_out(basis, column, projection_passes);
        set_block(result.projected, 0, col, column);
    }

    int q_columns = std::min(block.rows(), block.cols());
    result.q = Matrix(block.rows(), q_columns);
    result.r = Matrix(q_columns, block.cols());
    result.diagonal.assign(q_columns, 0.0);

    for (int col = 0; col < q_columns; ++col) {
        Matrix column = copy_block(
            result.projected, 0, col, block.rows(), 1);

        // Two passes keep the new columns mutually orthogonal.
        for (int pass = 0; pass < projection_passes; ++pass) {
            for (int previous = 0; previous < col; ++previous) {
                double coefficient = 0.0;
                for (int row = 0; row < block.rows(); ++row) {
                    coefficient += result.q(row, previous) * column(row, 0);
                }
                result.r(previous, col) += coefficient;
                for (int row = 0; row < block.rows(); ++row) {
                    column(row, 0) -= coefficient * result.q(row, previous);
                }
            }
        }

        double norm = vector_norm(column.data(), column.rows());
        result.r(col, col) = norm;
        result.diagonal[col] = norm;
        if (norm > 0.0) {
            for (int row = 0; row < block.rows(); ++row) {
                result.q(row, col) = column(row, 0) / norm;
            }
        }
    }
    return result;
}

ColumnOrthogonalizationResult orthogonalize_column(
    const Matrix& column,
    const Matrix& basis,
    double absolute_tolerance,
    int projection_passes) {
    if (column.cols() != 1) {
        throw std::invalid_argument("Expected one column");
    }

    ColumnOrthogonalizationResult result;
    result.q = column;
    // Normalize only the part left outside the current basis.
    project_out(basis, result.q, projection_passes);
    result.norm = vector_norm(result.q.data(), result.q.rows());
    result.accepted = result.norm > absolute_tolerance;

    if (result.accepted) {
        for (int row = 0; row < result.q.rows(); ++row) {
            result.q(row, 0) /= result.norm;
        }
    } else {
        result.q.fill(0.0);
    }
    return result;
}

}  // namespace math
}  // namespace ac_rsvd
