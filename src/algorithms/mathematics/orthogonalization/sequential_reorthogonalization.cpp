#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"

#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/projection.hpp"

namespace ac_rsvd {
namespace math {

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
