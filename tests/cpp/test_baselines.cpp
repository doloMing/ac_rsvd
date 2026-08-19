#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ac_rsvd/randqb_ei.hpp"
#include "ac_rsvd/randqb_mf_fro.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/operators/dense_operator.hpp"

using ac_rsvd::FactorizationResult;
using ac_rsvd::math::Matrix;

static Matrix diagonal_matrix() {
    Matrix matrix(8, 6);
    double values[] = {10.0, 5.0, 2.0, 0.5, 0.1, 0.01};
    for (int index = 0; index < 6; ++index) {
        matrix(index, index) = values[index];
    }
    return matrix;
}

static double factorization_error(
    const Matrix& matrix,
    const FactorizationResult& result) {
    double error_squared = 0.0;
    for (int col = 0; col < matrix.cols(); ++col) {
        for (int row = 0; row < matrix.rows(); ++row) {
            double approximation = 0.0;
            for (int index = 0; index < result.rank; ++index) {
                approximation +=
                    result.u[row + index * matrix.rows()] *
                    result.singular_values[index] *
                    result.v[col + index * matrix.cols()];
            }
            double difference = matrix(row, col) - approximation;
            error_squared += difference * difference;
        }
    }
    return std::sqrt(error_squared);
}

static double exact_frobenius_norm(const Matrix& matrix) {
    return ac_rsvd::math::frobenius_norm(matrix);
}

static void check_result_shape(const FactorizationResult& result) {
    if (static_cast<int>(result.u.size()) != result.rows * result.rank ||
        static_cast<int>(result.singular_values.size()) != result.rank ||
        static_cast<int>(result.v.size()) != result.cols * result.rank) {
        throw std::runtime_error("Factorization shape is wrong");
    }
}

static void test_randqb_ei(const Matrix& matrix) {
    ac_rsvd::math::DenseOperator op(matrix);
    ac_rsvd::RandQbEiOptions options;
    options.tolerance = 0.2;
    options.frobenius_norm = exact_frobenius_norm(matrix);
    options.block_size = 2;
    options.seed = 7;

    FactorizationResult result = ac_rsvd::compute_randqb_ei(op, options);
    check_result_shape(result);
    double error = factorization_error(matrix, result);
    if (error > options.tolerance) {
        throw std::runtime_error("randQB_EI missed the test tolerance");
    }
    if (std::abs(error * error - result.residual_estimate_squared) > 1e-10) {
        throw std::runtime_error("randQB_EI error indicator is wrong");
    }
    if (result.rank != 5 || result.statistics.directions_processed != 6 ||
        result.statistics.a_columns != 6 || result.statistics.at_columns != 6 ||
        result.statistics.a_block_calls != 3 ||
        result.statistics.at_block_calls != 3) {
        throw std::runtime_error("randQB_EI work counters are wrong");
    }
}

static void test_randqb_mf_fro(const Matrix& matrix) {
    ac_rsvd::math::DenseOperator op(matrix);
    ac_rsvd::RandQbMfFroOptions options;
    options.tolerance = 1.0;
    options.block_size = 2;
    options.seed = 7;

    FactorizationResult result = ac_rsvd::compute_randqb_mf_fro(op, options);
    check_result_shape(result);
    double error = factorization_error(matrix, result);
    if (error > options.tolerance) {
        throw std::runtime_error("randQB_MF_Fro missed the test tolerance");
    }
    if (result.residual_estimate_squared > options.tolerance * options.tolerance) {
        throw std::runtime_error("randQB_MF_Fro stopped above its indicator tolerance");
    }
    if (result.rank != 3 || result.statistics.directions_processed != 6 ||
        result.statistics.a_columns != 6 || result.statistics.at_columns != 4 ||
        result.statistics.a_block_calls != 3 ||
        result.statistics.at_block_calls != 2) {
        throw std::runtime_error("randQB_MF_Fro work counters are wrong");
    }
}

static void test_empty_randqb_ei(const Matrix& matrix) {
    ac_rsvd::math::DenseOperator op(matrix);
    ac_rsvd::RandQbEiOptions options;
    options.tolerance = exact_frobenius_norm(matrix);
    options.frobenius_norm = options.tolerance;

    FactorizationResult result = ac_rsvd::compute_randqb_ei(op, options);
    if (result.rank != 0 || result.statistics.a_columns != 0 ||
        result.statistics.at_columns != 0) {
        throw std::runtime_error("randQB_EI did work for an empty answer");
    }
}

static void test_final_lookahead_mf_fro() {
    Matrix matrix(3, 3);
    matrix(0, 0) = 1000.0;
    matrix(1, 1) = 10.0;
    matrix(2, 2) = 0.1;
    ac_rsvd::math::DenseOperator op(matrix);

    ac_rsvd::RandQbMfFroOptions options;
    options.tolerance = 0.2;
    options.block_size = 2;
    options.seed = 6;

    FactorizationResult result = ac_rsvd::compute_randqb_mf_fro(op, options);
    if (result.rank != 2 || factorization_error(matrix, result) > 0.2) {
        throw std::runtime_error("randQB_MF_Fro missed final block pruning");
    }
    if (result.statistics.directions_processed != 5 ||
        result.statistics.a_columns != 5 ||
        result.statistics.at_columns != 3 ||
        result.statistics.a_block_calls != 3 ||
        result.statistics.at_block_calls != 2) {
        throw std::runtime_error("Final look-ahead counters are wrong");
    }
}

int main() {
    try {
        Matrix matrix = diagonal_matrix();
        test_randqb_ei(matrix);
        test_randqb_mf_fro(matrix);
        test_empty_randqb_ei(matrix);
        test_final_lookahead_mf_fro();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "baseline tests passed\n";
    return 0;
}
