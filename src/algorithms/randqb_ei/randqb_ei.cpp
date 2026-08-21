#include "ac_rsvd/randqb_ei.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/qr.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/operators/operator_tools.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"

namespace ac_rsvd {

using math::Matrix;

static void subtract_matrix(Matrix& left, const Matrix& right) {
    for (int index = 0; index < left.size(); ++index) {
        left.data()[index] -= right.data()[index];
    }
}

static Matrix make_basis_block(const Matrix& sample, const Matrix& basis) {
    Matrix block = math::thin_qr(sample).q;
    if (basis.cols() > 0) {
        Matrix coefficients = math::transpose_multiply(basis, block);
        Matrix projection = math::multiply(basis, coefficients);
        subtract_matrix(block, projection);
    }
    // Algorithm 2 takes this second QR even for the first block.
    return math::thin_qr(block).q;
}

static FactorizationResult make_result(
    Matrix& q,
    Matrix& bt,
    RunStatistics statistics,
    StopReason stop_reason,
    double residual_estimate_squared,
    std::chrono::steady_clock::time_point algorithm_start) {
    FactorizationResult result;
    result.rows = q.rows();
    result.cols = bt.rows();
    result.rank = q.cols();
    result.range_rank = q.cols();
    result.boundary_only_rank = q.cols();
    result.statistics = statistics;
    result.stop_reason = stop_reason;
    result.residual_estimate_squared = residual_estimate_squared;

    if (result.rank == 0) {
        auto end = std::chrono::steady_clock::now();
        result.statistics.total_seconds =
            std::chrono::duration<double>(end - algorithm_start).count();
        return result;
    }

    // The columns of B^T become the right singular vectors in place.
    auto svd_start = std::chrono::steady_clock::now();
    math::TallSvdResult svd = math::tall_svd_in_place(bt);
    Matrix core_left_vectors(q.cols(), q.cols());
    for (int col = 0; col < q.cols(); ++col) {
        for (int row = 0; row < q.cols(); ++row) {
            core_left_vectors(row, col) = svd.vt(col, row);
        }
    }
    Matrix u = math::multiply(q, core_left_vectors);
    auto svd_end = std::chrono::steady_clock::now();
    result.statistics.svd_seconds +=
        std::chrono::duration<double>(svd_end - svd_start).count();
    result.singular_values = svd.singular_values;
    u.give_values_to(result.u);
    bt.give_values_to(result.v);
    auto end = std::chrono::steady_clock::now();
    result.statistics.total_seconds =
        std::chrono::duration<double>(end - algorithm_start).count();
    return result;
}

FactorizationResult compute_randqb_ei(
    const MatrixOperator& matrix,
    const RandQbEiOptions& options) {
    if (!std::isfinite(options.tolerance) ||
        options.tolerance <= 0.0 || options.block_size <= 0) {
        throw std::invalid_argument("Tolerance and block size must be positive");
    }
    if (!std::isfinite(options.frobenius_norm) ||
        options.frobenius_norm < 0.0) {
        throw std::invalid_argument("Frobenius norm cannot be negative");
    }

    int rows = matrix.rows();
    int cols = matrix.cols();
    int max_rank = std::min(rows, cols);
    double tolerance_squared = options.tolerance * options.tolerance;
    // EI starts from E = ||A||_F^2 and subtracts captured row energy.
    double error_squared = options.frobenius_norm * options.frobenius_norm;
    RunStatistics statistics;

    Matrix q(rows, 0);
    // B^T keeps every accepted block contiguous.
    Matrix bt(cols, 0);
    auto algorithm_start = std::chrono::steady_clock::now();

    while (q.cols() < max_rank) {
        int block_size = std::min(options.block_size, max_rank - q.cols());
        Matrix omega(cols, block_size);
        math::fill_gaussian(
            omega.data(),
            cols,
            block_size,
            options.seed,
            options.stream,
            statistics.directions_processed);
        statistics.directions_processed += block_size;

        Matrix sample = math::apply_a(matrix, omega, statistics);
        if (q.cols() > 0) {
            // This is (A - QB) Omega without forming the residual.
            Matrix b_omega = math::transpose_multiply(bt, omega);
            math::gemm(q, false, b_omega, false, -1.0, 1.0, sample);
        }

        auto orth_start = std::chrono::steady_clock::now();
        Matrix q_block = make_basis_block(sample, q);
        auto orth_end = std::chrono::steady_clock::now();
        statistics.orthogonalization_seconds +=
            std::chrono::duration<double>(orth_end - orth_start).count();
        Matrix at_q = math::apply_at(matrix, q_block, statistics);
        if (q.cols() > 0) {
            // This is zero exactly and repairs lost orthogonality in FP64.
            Matrix overlap = math::transpose_multiply(q_block, q);
            Matrix correction = math::multiply_transpose(bt, overlap);
            subtract_matrix(at_q, correction);
        }

        int keep = block_size;
        bool tolerance_met = false;
        // Each row is one exact Pythagorean decrease in E.
        for (int row = 0; row < block_size; ++row) {
            double row_energy = 0.0;
            for (int col = 0; col < cols; ++col) {
                double value = at_q(col, row);
                row_energy += value * value;
            }
            error_squared -= row_energy;
            if (error_squared < tolerance_squared) {
                keep = row + 1;
                tolerance_met = true;
                break;
            }
        }

        q.append_columns(q_block, keep);
        bt.append_columns(at_q, keep);

        if (tolerance_met) {
            return make_result(
                q,
                bt,
                statistics,
                StopReason::tolerance_met,
                error_squared,
                algorithm_start);
        }
    }

    return make_result(
        q,
        bt,
        statistics,
        StopReason::full_rank,
        error_squared,
        algorithm_start);
}

}  // namespace ac_rsvd
