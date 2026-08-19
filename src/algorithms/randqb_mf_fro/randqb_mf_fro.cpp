#include "ac_rsvd/randqb_mf_fro.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

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

static void prune_last_block(
    const Matrix& old_q,
    const Matrix& old_b,
    int last_block_size,
    double tolerance_squared,
    Matrix& q,
    Matrix& b,
    double& residual_estimate_squared) {
    int first = old_q.cols() - last_block_size;
    std::vector<std::pair<double, int>> row_energy;
    row_energy.reserve(last_block_size);

    for (int row = 0; row < last_block_size; ++row) {
        double energy = 0.0;
        for (int col = 0; col < old_b.cols(); ++col) {
            double value = old_b(first + row, col);
            energy += value * value;
        }
        row_energy.push_back(std::make_pair(energy, row));
    }
    std::sort(row_energy.begin(), row_energy.end());

    std::vector<int> discard(last_block_size, 0);
    // Try the smallest rows first.
    for (int index = 0; index < last_block_size; ++index) {
        double next = residual_estimate_squared + row_energy[index].first;
        if (next >= tolerance_squared) {
            break;
        }
        residual_estimate_squared = next;
        discard[row_energy[index].second] = 1;
    }

    int kept_from_last = 0;
    for (int index = 0; index < last_block_size; ++index) {
        if (!discard[index]) {
            ++kept_from_last;
        }
    }

    q = Matrix(old_q.rows(), first + kept_from_last);
    b = Matrix(first + kept_from_last, old_b.cols());
    if (first > 0) {
        math::set_block(q, 0, 0, math::copy_block(old_q, 0, 0, q.rows(), first));
        math::set_block(b, 0, 0, math::copy_block(old_b, 0, 0, first, b.cols()));
    }

    int output_col = first;
    for (int index = 0; index < last_block_size; ++index) {
        if (discard[index]) {
            continue;
        }
        for (int row = 0; row < q.rows(); ++row) {
            q(row, output_col) = old_q(row, first + index);
        }
        for (int col = 0; col < b.cols(); ++col) {
            b(output_col, col) = old_b(first + index, col);
        }
        ++output_col;
    }
}

FactorizationResult compute_randqb_mf_fro(
    const MatrixOperator& matrix,
    const RandQbMfFroOptions& options) {
    if (options.tolerance <= 0.0 || options.block_size <= 0) {
        throw std::invalid_argument("Tolerance and block size must be positive");
    }

    int rows = matrix.rows();
    int cols = matrix.cols();
    int max_rank = std::min(rows, cols);
    double tolerance_squared = options.tolerance * options.tolerance;

    Matrix q(rows, 0);
    Matrix b(0, cols);
    RunStatistics statistics;
    int last_block_size = 0;
    double residual_estimate_squared = 0.0;

    while (true) {
        int remaining_rank = max_rank - q.cols();
        int block_size = options.block_size;
        if (remaining_rank > 0) {
            block_size = std::min(block_size, remaining_rank);
        }
        Matrix omega(cols, block_size);
        math::fill_gaussian(
            omega.data(),
            cols,
            block_size,
            options.seed,
            options.stream,
            statistics.directions_processed);
        statistics.directions_processed += block_size;

        double scale = 1.0 / std::sqrt(static_cast<double>(block_size));
        for (int index = 0; index < omega.size(); ++index) {
            omega.data()[index] *= scale;
        }

        Matrix z = math::apply_a(matrix, omega, statistics);
        Matrix sample = z;
        if (q.cols() > 0) {
            Matrix b_omega = math::multiply(b, omega);
            Matrix represented = math::multiply(q, b_omega);
            subtract_matrix(sample, represented);
        }

        // This sketch tests the basis we already have.
        residual_estimate_squared = math::squared_frobenius_norm(sample);
        if (residual_estimate_squared <= tolerance_squared) {
            if (q.cols() == 0) {
                return make_result(
                    q,
                    b,
                    statistics,
                    StopReason::tolerance_met,
                    residual_estimate_squared);
            }

            Matrix pruned_q;
            Matrix pruned_b;
            prune_last_block(
                q,
                b,
                last_block_size,
                tolerance_squared,
                pruned_q,
                pruned_b,
                residual_estimate_squared);
            return make_result(
                pruned_q,
                pruned_b,
                statistics,
                StopReason::tolerance_met,
                residual_estimate_squared);
        }

        if (remaining_rank == 0) {
            return make_result(
                q,
                b,
                statistics,
                StopReason::full_rank,
                residual_estimate_squared);
        }

        Matrix q_block = make_basis_block(sample, q);
        Matrix at_q = math::apply_at(matrix, q_block, statistics);
        Matrix b_block = math::transpose(at_q);
        last_block_size = q_block.cols();
        q = math::join_columns(q, q_block);
        b = join_rows(b, b_block);
    }
}

}  // namespace ac_rsvd
