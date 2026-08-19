#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ac_rsvd/ac_rsvd.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/operators/dense_operator.hpp"

using ac_rsvd::FactorizationResult;
using ac_rsvd::StopReason;
using ac_rsvd::math::DenseOperator;
using ac_rsvd::math::Matrix;

static Matrix diagonal_matrix(int rows, int cols, double tail) {
    Matrix matrix(rows, cols);
    int count = std::min(rows, cols);
    for (int index = 0; index < count; ++index) {
        matrix(index, index) = tail;
    }
    return matrix;
}

static double residual_norm(
    const Matrix& matrix,
    const FactorizationResult& result) {
    double squared_norm = 0.0;
    for (int col = 0; col < matrix.cols(); ++col) {
        for (int row = 0; row < matrix.rows(); ++row) {
            double approximation = 0.0;
            for (int index = 0; index < result.rank; ++index) {
                approximation +=
                    result.u[row + index * result.rows]
                    * result.singular_values[index]
                    * result.v[col + index * result.cols];
            }
            double difference = matrix(row, col) - approximation;
            squared_norm += difference * difference;
        }
    }
    return std::sqrt(squared_norm);
}

static ac_rsvd::AcRsvdOptions options(int block_size) {
    ac_rsvd::AcRsvdOptions value;
    value.tolerance = 1e-10;
    value.failure_probability = 0.1;
    value.block_size = block_size;
    value.seed = 29;
    value.stream = 3;
    return value;
}

static void check_zero_matrix() {
    Matrix matrix(6, 4);
    DenseOperator op(matrix);
    FactorizationResult result = ac_rsvd::compute_ac_rsvd(op, options(3));

    if (result.stop_reason != StopReason::zero_residual || result.rank != 0) {
        throw std::runtime_error("Zero matrix took the wrong terminal route");
    }
    if (result.statistics.directions_processed != 1 ||
        result.statistics.a_columns != 3 ||
        result.statistics.a_block_calls != 1 ||
        result.statistics.at_block_calls != 0) {
        throw std::runtime_error("Zero matrix product counts are wrong");
    }
}

static void check_output_space_stop() {
    Matrix matrix(2, 5);
    matrix(0, 0) = 2.0;
    matrix(0, 2) = -1.0;
    matrix(0, 4) = 0.5;
    matrix(1, 1) = 1.5;
    matrix(1, 3) = 0.25;
    DenseOperator op(matrix);

    FactorizationResult result = ac_rsvd::compute_ac_rsvd(op, options(4));
    if (result.stop_reason != StopReason::full_output_space) {
        throw std::runtime_error("Full output space was not detected");
    }
    if (result.statistics.directions_processed != 2 ||
        result.statistics.a_columns != 4 ||
        result.statistics.at_block_calls != 1) {
        throw std::runtime_error("Output-space product counts are wrong");
    }
    if (residual_norm(matrix, result) > 1e-11) {
        throw std::runtime_error("Output-space result is inaccurate");
    }
}

static void check_final_input_stop() {
    Matrix matrix = diagonal_matrix(6, 3, 0.0);
    matrix(0, 0) = 2.0;
    matrix(1, 1) = 0.5;
    matrix(2, 2) = 0.125;
    DenseOperator op(matrix);

    FactorizationResult result = ac_rsvd::compute_ac_rsvd(op, options(2));
    if (result.stop_reason != StopReason::final_input_direction ||
        result.statistics.directions_processed != 3 ||
        result.statistics.a_columns != 3 ||
        result.statistics.a_block_calls != 2 ||
        result.statistics.at_block_calls != 1) {
        throw std::runtime_error("Final input direction route is wrong");
    }
    if (residual_norm(matrix, result) > 1e-11) {
        throw std::runtime_error("Final-direction result is inaccurate");
    }
}

static void check_certificate_and_block_order() {
    Matrix matrix = diagonal_matrix(64, 64, 1e-8);
    matrix(0, 0) = 1.0;
    matrix(1, 1) = 0.4;
    matrix(2, 2) = 0.15;
    DenseOperator op(matrix);

    ac_rsvd::AcRsvdOptions scalar_options = options(1);
    scalar_options.tolerance = 1e-2;
    scalar_options.failure_probability = 0.5;
    FactorizationResult scalar =
        ac_rsvd::compute_ac_rsvd(op, scalar_options);

    ac_rsvd::AcRsvdOptions block_options = scalar_options;
    block_options.block_size = 8;
    FactorizationResult blocked =
        ac_rsvd::compute_ac_rsvd(op, block_options);

    if (scalar.stop_reason != StopReason::certificate ||
        blocked.stop_reason != StopReason::certificate) {
        throw std::runtime_error("Certificate route did not stop the run");
    }
    if (scalar.statistics.directions_processed !=
        blocked.statistics.directions_processed) {
        throw std::runtime_error("Block QR changed the stopping round");
    }
    if (blocked.statistics.a_columns <
            blocked.statistics.directions_processed ||
        blocked.statistics.a_columns - blocked.statistics.directions_processed >=
            block_options.block_size) {
        throw std::runtime_error("Final block accounting is wrong");
    }
    if (blocked.statistics.at_columns != blocked.statistics.a_columns) {
        throw std::runtime_error("Final block products were not all used");
    }
    if (scalar.statistics.at_block_calls != 1 ||
        blocked.statistics.at_block_calls != 1) {
        throw std::runtime_error("AC-RSVD used more than one adjoint call");
    }
    if (scalar.rank != 3 || blocked.rank != 3 ||
        blocked.residual_bound_squared >
            block_options.tolerance * block_options.tolerance ||
        blocked.truncation_budget_squared < 0.0) {
        throw std::runtime_error("Certificate truncation is wrong");
    }
    if (residual_norm(matrix, scalar) > scalar_options.tolerance ||
        residual_norm(matrix, blocked) > block_options.tolerance) {
        throw std::runtime_error("Certified result missed the tolerance");
    }
}

int main() {
    try {
        check_zero_matrix();
        check_output_space_stop();
        check_final_input_stop();
        check_certificate_and_block_order();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "AC-RSVD tests passed\n";
    return 0;
}
