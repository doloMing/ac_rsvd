#include "algorithms/mathematics/operators/operator_tools.hpp"

#include <chrono>
#include <stdexcept>

namespace ac_rsvd {
namespace math {

Matrix apply_a(
    const MatrixOperator& matrix,
    const Matrix& input,
    RunStatistics& statistics) {
    Matrix output(matrix.rows(), input.cols());
    apply_a_into(matrix, input, output, statistics);
    return output;
}

void apply_a_into(
    const MatrixOperator& matrix,
    const Matrix& input,
    Matrix& output,
    RunStatistics& statistics) {
    if (output.rows() != matrix.rows() || output.cols() != input.cols()) {
        throw std::invalid_argument("Operator output dimensions do not match");
    }
    // One panel is one operator call, regardless of its column count.
    auto start = std::chrono::steady_clock::now();
    matrix.apply(input.data(), input.cols(), output.data());
    auto end = std::chrono::steady_clock::now();
    statistics.a_seconds += std::chrono::duration<double>(end - start).count();
    statistics.a_columns += input.cols();
    statistics.a_block_calls += 1;
}

void apply_a_columns_into(
    const MatrixOperator& matrix,
    const Matrix& input,
    int first_column,
    int column_count,
    Matrix& output,
    RunStatistics& statistics) {
    if (first_column < 0 || column_count < 0 ||
        first_column + column_count > input.cols() ||
        output.rows() != matrix.rows() ||
        output.cols() != input.cols()) {
        throw std::invalid_argument("Operator panel is out of range");
    }

    auto start = std::chrono::steady_clock::now();
    matrix.apply(
        input.column_data(first_column),
        column_count,
        output.column_data(first_column));
    auto end = std::chrono::steady_clock::now();
    statistics.a_seconds += std::chrono::duration<double>(end - start).count();
    statistics.a_columns += column_count;
    statistics.a_block_calls += 1;
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
