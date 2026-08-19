#include "ac_rsvd/randqb_ei.hpp"

#include <algorithm>
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
    if (basis.cols() == 0) {
        return block;
    }

    Matrix coefficients = math::transpose_multiply(basis, block);
    Matrix projection = math::multiply(basis, coefficients);
    subtract_matrix(block, projection);
    return math::thin_qr(block).q;
}

static Matrix join_rows(const Matrix& top, const Matrix& bottom) {
    Matrix joined(top.rows() + bottom.rows(), top.cols());
    math::set_block(joined, 0, 0, top);
    math::set_block(joined, top.rows(), 0, bottom);
    return joined;
}

static FactorizationResult make_result(
    const Matrix& q,
    const Matrix& b,
    const RunStatistics& statistics,
    StopReason stop_reason,
    double residual_estimate_squared) {
    FactorizationResult result;
    result.rows = q.rows();
    result.cols = b.cols();
    result.rank = q.cols();
    result.statistics = statistics;
    result.stop_reason = stop_reason;
    result.residual_estimate_squared = residual_estimate_squared;

    if (result.rank == 0) {
        return result;
    }

    math::SvdResult svd = math::compact_qb_svd(q, b);
    Matrix v = math::transpose(svd.vt);
    result.u.assign(svd.u.data(), svd.u.data() + svd.u.size());
    result.singular_values = svd.singular_values;
    result.v.assign(v.data(), v.data() + v.size());
    return result;
}

FactorizationResult compute_randqb_ei(
    const MatrixOperator& matrix,
    const RandQbEiOptions& options) {
    if (options.tolerance <= 0.0 || options.block_size <= 0 ||
        options.frobenius_norm < 0.0) {
        throw std::invalid_argument("Tolerance and block size must be positive");
    }

    int rows = matrix.rows();
    int cols = matrix.cols();
    int max_rank = std::min(rows, cols);
    double tolerance_squared = options.tolerance * options.tolerance;
    double error_squared = options.frobenius_norm * options.frobenius_norm;
    RunStatistics statistics;

    if (error_squared <= tolerance_squared) {
        return make_result(
            Matrix(rows, 0),
            Matrix(0, cols),
            statistics,
            StopReason::tolerance_met,
            error_squared);
    }

    Matrix q(rows, 0);
    Matrix b(0, cols);

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

        Matrix z = math::apply_a(matrix, omega, statistics);
        Matrix sample = z;
        if (q.cols() > 0) {
            Matrix b_omega = math::multiply(b, omega);
            Matrix represented = math::multiply(q, b_omega);
            subtract_matrix(sample, represented);
        }

        Matrix q_block = make_basis_block(sample, q);
        Matrix at_q = math::apply_at(matrix, q_block, statistics);
        Matrix b_block = math::transpose(at_q);

        int keep = block_size;
        bool tolerance_met = false;
        // Stop on the first row that gets us below the target.
        for (int row = 0; row < block_size; ++row) {
            double row_energy = 0.0;
            for (int col = 0; col < cols; ++col) {
                double value = b_block(row, col);
                row_energy += value * value;
            }
            error_squared -= row_energy;
            if (error_squared < tolerance_squared) {
                keep = row + 1;
                tolerance_met = true;
                break;
            }
        }

        Matrix kept_q = math::copy_block(q_block, 0, 0, rows, keep);
        Matrix kept_b = math::copy_block(b_block, 0, 0, keep, cols);
        q = math::join_columns(q, kept_q);
        b = join_rows(b, kept_b);

        if (tolerance_met) {
            return make_result(
                q,
                b,
                statistics,
                StopReason::tolerance_met,
                std::max(0.0, error_squared));
        }
    }

    return make_result(
        q,
        b,
        statistics,
        StopReason::full_rank,
        std::max(0.0, error_squared));
}

}  // namespace ac_rsvd
