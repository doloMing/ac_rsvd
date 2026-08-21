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

void apply_a_into(
    const MatrixOperator& matrix,
    const Matrix& input,
    Matrix& output,
    RunStatistics& statistics);

void apply_a_columns_into(
    const MatrixOperator& matrix,
    const Matrix& input,
    int first_column,
    int column_count,
    Matrix& output,
    RunStatistics& statistics);

Matrix apply_at(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics);

}  // namespace math
}  // namespace ac_rsvd
