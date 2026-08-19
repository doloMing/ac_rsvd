#pragma once

#include "ac_rsvd/factorization_result.hpp"
#include "ac_rsvd/matrix_operator.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"

namespace ac_rsvd {
namespace math {

// These panel calls also account for operator work in the run statistics.
Matrix apply_a(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics);

Matrix apply_at(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics);

}  // namespace math
}  // namespace ac_rsvd
