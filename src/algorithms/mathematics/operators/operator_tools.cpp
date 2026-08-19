#include "algorithms/mathematics/operators/operator_tools.hpp"

namespace ac_rsvd {
namespace math {

Matrix apply_a(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics) {
    Matrix output(matrix.rows(), input.cols());
    matrix.apply(input.data(), input.cols(), output.data());
    statistics.a_columns += input.cols();
    statistics.a_block_calls += 1;
    return output;
}

Matrix apply_at(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics) {
    Matrix output(matrix.cols(), input.cols());
    matrix.apply_transpose(input.data(), input.cols(), output.data());
    statistics.at_columns += input.cols();
    statistics.at_block_calls += 1;
    return output;
}

}  // namespace math
}  // namespace ac_rsvd
