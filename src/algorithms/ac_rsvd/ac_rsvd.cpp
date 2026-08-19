#include "ac_rsvd/ac_rsvd.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "algorithms/mathematics/certificate/certificate.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/projection.hpp"
#include "algorithms/mathematics/linalg/qr.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/linalg/truncation.hpp"
#include "algorithms/mathematics/operators/operator_tools.hpp"
#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"

namespace ac_rsvd {

using math::Matrix;

static Matrix prefix_columns(const Matrix& matrix, int count) {
    return math::copy_block(matrix, 0, 0, matrix.rows(), count);
}

static Matrix basis_with_prefix(
    const Matrix& basis,
    const Matrix& block,
    int prefix_size) {
    return math::join_columns(basis, prefix_columns(block, prefix_size));
}

static math::BlockOrthogonalizationResult orthogonalize_panel(
    const Matrix& block,
    const Matrix& basis,
    bool sequential) {
    if (sequential) {
        return math::orthogonalize_sequential_block(block, basis, 2);
    }
    return math::orthogonalize_block(block, basis, 2);
}

static FactorizationResult finish_factorization(
    const MatrixOperator& matrix,
    const Matrix& final_basis,
    int first_extra_column,
    double tolerance_squared,
    double residual_bound_squared,
    StopReason stop_reason,
    RunStatistics statistics,
    const math::Certificate& certificate,
    std::chrono::steady_clock::time_point algorithm_start) {
    FactorizationResult result;
    result.rows = matrix.rows();
    result.cols = matrix.cols();
    result.range_rank = final_basis.cols();
    result.statistics = statistics;
    result.stop_reason = stop_reason;
    result.residual_bound_squared = residual_bound_squared;

    const std::vector<math::CertificateTraceEntry>& trace = certificate.trace();
    for (const math::CertificateTraceEntry& entry : trace) {
        result.certificate_observations.push_back(entry.x);
        result.certificate_dimensions.push_back(entry.unused_dimension);
    }

    if (final_basis.cols() == 0) {
        result.truncation_budget_squared = tolerance_squared;
        auto end = std::chrono::steady_clock::now();
        result.statistics.total_seconds =
            std::chrono::duration<double>(end - algorithm_start).count();
        return result;
    }

    // The range is fixed. One adjoint panel now gives the whole core.
    Matrix at_basis = math::apply_at(matrix, final_basis, result.statistics);
    Matrix core = math::transpose(at_basis);

    double residual_decrease_squared = 0.0;
    if (first_extra_column >= 0) {
        // These rows are H_t^T A, so their squared norm is Delta_t.
        for (int col = 0; col < core.cols(); ++col) {
            for (int row = first_extra_column; row < core.rows(); ++row) {
                double value = core(row, col);
                residual_decrease_squared += value * value;
            }
        }
    }
    result.residual_decrease_squared = residual_decrease_squared;

    double budget = tolerance_squared;
    if (first_extra_column >= 0) {
        budget = tolerance_squared
               - residual_bound_squared
               + residual_decrease_squared;
    }
    result.truncation_budget_squared = budget;

    auto svd_start = std::chrono::steady_clock::now();
    math::SvdResult svd = math::thin_svd(core);
    int rank = math::smallest_rank_for_tail(svd.singular_values, budget);
    int boundary_only_rank = rank;
    if (first_extra_column >= 0) {
        boundary_only_rank = math::smallest_rank_for_tail(
            svd.singular_values,
            residual_decrease_squared);
    }
    result.boundary_only_rank = boundary_only_rank;
    svd = math::truncate_svd(svd, rank);
    svd.u = math::multiply(final_basis, svd.u);
    Matrix v = math::transpose(svd.vt);
    auto svd_end = std::chrono::steady_clock::now();
    result.statistics.svd_seconds +=
        std::chrono::duration<double>(svd_end - svd_start).count();

    result.rank = rank;
    result.singular_values = svd.singular_values;
    if (rank > 0) {
        result.u.assign(svd.u.data(), svd.u.data() + svd.u.size());
        result.v.assign(v.data(), v.data() + v.size());
    }
    auto end = std::chrono::steady_clock::now();
    result.statistics.total_seconds =
        std::chrono::duration<double>(end - algorithm_start).count();
    return result;
}

FactorizationResult compute_ac_rsvd(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options) {
    if (matrix.rows() < 1 || matrix.cols() < 1) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (options.tolerance <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    if (options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("Failure probability must be between zero and one");
    }
    if (options.block_size <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }

    int rows = matrix.rows();
    int cols = matrix.cols();
    double tolerance_squared = options.tolerance * options.tolerance;

    Matrix input_basis(cols, 0);
    Matrix output_basis(rows, 0);
    // In the exact model, output_basis spans A times input_basis.
    math::Certificate certificate(
        std::max(2, cols),
        tolerance_squared,
        options.failure_probability);
    RunStatistics statistics;
    auto algorithm_start = std::chrono::steady_clock::now();

    while (true) {
        if (output_basis.cols() == rows) {
            return finish_factorization(
                matrix,
                output_basis,
                -1,
                tolerance_squared,
                0.0,
                StopReason::full_output_space,
                statistics,
                certificate,
                algorithm_start);
        }

        int unused_at_block_start = cols - input_basis.cols();
        int block_size = std::min(options.block_size, unused_at_block_start);

        Matrix gaussian(cols, block_size);
        math::BlockOrthogonalizationResult input_block;
        std::uint64_t redraw = 0;
        while (true) {
            math::fill_gaussian(
                gaussian.data(),
                cols,
                block_size,
                options.seed,
                options.stream + redraw,
                static_cast<std::uint64_t>(input_basis.cols()));
            auto orth_start = std::chrono::steady_clock::now();
            input_block = orthogonalize_panel(
                gaussian,
                input_basis,
                options.use_sequential_orthogonalization);
            auto orth_end = std::chrono::steady_clock::now();
            statistics.orthogonalization_seconds +=
                std::chrono::duration<double>(orth_end - orth_start).count();

            bool singular = false;
            for (double diagonal : input_block.diagonal) {
                if (diagonal == 0.0) {
                    singular = true;
                }
            }
            if (!singular) {
                break;
            }
            // A singular projected Gaussian panel is a zero-probability draw.
            ++redraw;
        }

        Matrix products = math::apply_a(matrix, input_block.q, statistics);
        auto orth_start = std::chrono::steady_clock::now();
        math::BlockOrthogonalizationResult output_block =
            orthogonalize_panel(
                products,
                output_basis,
                options.use_sequential_orthogonalization);
        auto orth_end = std::chrono::steady_clock::now();
        statistics.orthogonalization_seconds +=
            std::chrono::duration<double>(orth_end - orth_start).count();

        // The fixed QR order lets us reveal the block one column at a time.
        for (int column = 0; column < block_size; ++column) {
            if (output_basis.cols() + column == rows) {
                Matrix current_basis = basis_with_prefix(
                    output_basis, output_block.q, column);
                return finish_factorization(
                    matrix,
                    current_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::full_output_space,
                    statistics,
                    certificate,
                    algorithm_start);
            }

            int unused_dimension = unused_at_block_start - column;
            double diagonal = output_block.diagonal[column];
            ++statistics.directions_processed;

            if (unused_dimension == 1) {
                Matrix current_basis = basis_with_prefix(
                    output_basis, output_block.q, column);
                if (diagonal > 0.0) {
                    Matrix current_direction = math::copy_block(
                        output_block.q, 0, column, rows, 1);
                    current_basis.append_columns(current_direction);
                }
                return finish_factorization(
                    matrix,
                    current_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::final_input_direction,
                    statistics,
                    certificate,
                    algorithm_start);
            }

            if (diagonal == 0.0) {
                Matrix current_basis = basis_with_prefix(
                    output_basis, output_block.q, column);
                return finish_factorization(
                    matrix,
                    current_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::zero_residual,
                    statistics,
                    certificate,
                    algorithm_start);
            }

            double observation =
                unused_dimension * diagonal * diagonal;
            auto certificate_start = std::chrono::steady_clock::now();
            bool crossed = certificate.update(observation, unused_dimension);
            auto certificate_end = std::chrono::steady_clock::now();
            statistics.certificate_seconds +=
                std::chrono::duration<double>(
                    certificate_end - certificate_start).count();
            if (crossed) {
                // Fix U_t before using the unprocessed suffix of this block.
                certificate_start = std::chrono::steady_clock::now();
                double residual_bound_squared =
                    certificate.continuous_inverse();
                certificate_end = std::chrono::steady_clock::now();
                statistics.certificate_seconds +=
                    std::chrono::duration<double>(
                        certificate_end - certificate_start).count();
                Matrix current_basis = basis_with_prefix(
                    output_basis, output_block.q, column);
                math::QrResult qr;
                qr.q = output_block.q;
                qr.r = output_block.r;
                qr.diagonal = output_block.diagonal;

                // The lower-right QR core contains the useful block suffix.
                orth_start = std::chrono::steady_clock::now();
                Matrix trailing = math::trailing_basis(qr, column);
                math::project_out(current_basis, trailing, 2);
                trailing = math::column_space(trailing);
                orth_end = std::chrono::steady_clock::now();
                statistics.orthogonalization_seconds +=
                    std::chrono::duration<double>(
                        orth_end - orth_start).count();

                int first_extra_column = current_basis.cols();
                current_basis.append_columns(trailing);

                return finish_factorization(
                    matrix,
                    current_basis,
                    first_extra_column,
                    tolerance_squared,
                    residual_bound_squared,
                    StopReason::certificate,
                    statistics,
                    certificate,
                    algorithm_start);
            }
        }

        // A completed panel now becomes part of the persistent state.
        input_basis.append_columns(input_block.q);
        output_basis.append_columns(output_block.q);
    }
}

}  // namespace ac_rsvd
