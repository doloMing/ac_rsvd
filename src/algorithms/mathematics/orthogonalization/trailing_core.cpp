#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"

#include <algorithm>
#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"

namespace ac_rsvd {
namespace math {

Matrix trailing_basis(
    const QrResult& block_qr,
    int first_column,
    double relative_tolerance) {
    int thin_cols = block_qr.q.cols();
    if (first_column < 0 || first_column > block_qr.r.cols()) {
        throw std::invalid_argument("Trailing column is out of range");
    }
    if (first_column >= thin_cols || first_column >= block_qr.r.cols()) {
        return Matrix(block_qr.q.rows(), 0);
    }

    int tail_rows = thin_cols - first_column;
    int tail_cols = block_qr.r.cols() - first_column;
    Matrix core = copy_block(
        block_qr.r,
        first_column,
        first_column,
        tail_rows,
        tail_cols);
    // range(Q_tail R_tail) = Q_tail range(R_tail).
    Matrix small_basis = column_space(core, relative_tolerance);
    Matrix q_tail = copy_block(
        block_qr.q,
        0,
        first_column,
        block_qr.q.rows(),
        tail_rows);
    return multiply(q_tail, small_basis);
}

}  // namespace math
}  // namespace ac_rsvd
