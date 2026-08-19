#include "algorithms/mathematics/operators/operator_tools.hpp"

#include <chrono>

namespace ac_rsvd {
namespace math {

Matrix apply_a(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics) {
    Matrix output(matrix.rows(), input.cols());
    // One panel is one operator call, regardless of its column count.
    auto start = std::chrono::steady_clock::now();
    matrix.apply(input.data(), input.cols(), output.data());
    auto end = std::chrono::steady_clock::now();
    statistics.a_seconds += std::chrono::duration<double>(end - start).count();
    statistics.a_columns += input.cols();
    statistics.a_block_calls += 1;
    return output;
}

Matrix apply_at(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics) {
    Matrix output(matrix.cols(), input.cols());
    // Count A^T separately because the algorithms use it only at chosen stages.
    auto start = std::chrono::steady_clock::now();
    matrix.apply_transpose(input.data(), input.cols(), output.data());
    auto end = std::chrono::steady_clock::now();
    statistics.at_seconds += std::chrono::duration<double>(end - start).count();
    statistics.at_columns += input.cols();
    statistics.at_block_calls += 1;
    return output;
}

}  // namespace math
}  // namespace ac_rsvd
