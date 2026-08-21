#include "ac_rsvd/ac_rsvd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/certificate/certificate.hpp"
#include "algorithms/mathematics/certificate/raw_gaussian_certificate.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/projection.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/linalg/truncation.hpp"
#include "algorithms/mathematics/operators/operator_tools.hpp"
#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"

namespace ac_rsvd {

using math::Matrix;

static void orthogonalize_panel_in_place(
    Matrix& block,
    const Matrix& basis,
    bool sequential,
    Matrix& r,
    std::vector<double>& diagonal) {
    if (sequential) {
        math::BlockOrthogonalizationResult result =
            math::orthogonalize_sequential_block(block, basis, 2);
        block = result.q;
        r = result.r;
        diagonal = result.diagonal;
        return;
    }
    math::orthogonalize_block_in_place(block, basis, r, diagonal, 2);
}

static double column_energy(const Matrix& matrix, int column) {
    double value = 0.0;
    for (int row = 0; row < matrix.rows(); ++row) {
        double entry = matrix(row, column);
        value += entry * entry;
    }
    return value;
}

static void column_energies(
    const Matrix& matrix,
    std::vector<double>& energies) {
    energies.resize(matrix.cols());
    long long element_count =
        static_cast<long long>(matrix.rows()) * matrix.cols();
    // Columns are independent. Each column keeps the same summation order.
    #pragma omp parallel for schedule(static) if(element_count >= 262144)
    for (int column = 0; column < matrix.cols(); ++column) {
        energies[column] = column_energy(matrix, column);
    }
}

static double energy_difference(double total, double projected) {
    double difference = total - projected;
    double scale = std::max(total, projected);
    double roundoff = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (difference < 0.0 && difference >= -roundoff) {
        return 0.0;
    }
    if (difference < 0.0) {
        throw std::runtime_error("Projected energy exceeds total energy");
    }
    return difference;
}

static Matrix draw_raw_gaussian(
    const AcRsvdOptions& options,
    int rows,
    int columns,
    std::uint64_t first_direction) {
    Matrix gaussian(rows, columns);
    math::fill_gaussian(
        gaussian.data(),
        rows,
        columns,
        options.seed,
        options.stream,
        first_direction);
    return gaussian;
}

static void raw_input_energies(
    const Matrix& gaussian,
    std::vector<double>& energies,
    RunStatistics& statistics) {
    auto start = std::chrono::steady_clock::now();
    column_energies(gaussian, energies);
    auto end = std::chrono::steady_clock::now();
    statistics.orthogonalization_seconds +=
        std::chrono::duration<double>(end - start).count();
}

struct DiagnosticWorkspace {
    Matrix gaussian;
    Matrix responses;
    Matrix coefficients;
    Matrix projected_gram;
    Matrix residual_gram;
    std::vector<double> total_energies;
    std::vector<double> projected_energies;
    std::vector<double> residual_energies;
};

static bool ensure_size(Matrix& matrix, int rows, int cols) {
    if (matrix.rows() != rows || matrix.cols() != cols) {
        matrix = Matrix(rows, cols);
        return true;
    }
    return false;
}

static void analyze_responses(
    const Matrix& responses,
    const Matrix& output_basis,
    bool form_gram,
    DiagnosticWorkspace& workspace,
    RunStatistics& statistics) {
    auto start = std::chrono::steady_clock::now();
    column_energies(responses, workspace.total_energies);
    workspace.projected_energies.resize(responses.cols());
    if (output_basis.cols() > 0) {
        ensure_size(
            workspace.coefficients,
            output_basis.cols(),
            responses.cols());
        math::gemm(
            output_basis,
            true,
            responses,
            false,
            1.0,
            0.0,
            workspace.coefficients);
        column_energies(
            workspace.coefficients,
            workspace.projected_energies);
    } else {
        std::fill(
            workspace.projected_energies.begin(),
            workspace.projected_energies.end(),
            0.0);
    }

    workspace.residual_energies.resize(responses.cols());
    for (int column = 0; column < responses.cols(); ++column) {
        workspace.residual_energies[column] = energy_difference(
            workspace.total_energies[column],
            workspace.projected_energies[column]);
    }

    if (form_gram) {
        ensure_size(
            workspace.residual_gram,
            responses.cols(),
            responses.cols());
        math::gemm(
            responses,
            true,
            responses,
            false,
            1.0,
            0.0,
            workspace.residual_gram);
        if (output_basis.cols() > 0) {
            ensure_size(
                workspace.projected_gram,
                responses.cols(),
                responses.cols());
            math::gemm(
                workspace.coefficients,
                true,
                workspace.coefficients,
                false,
                1.0,
                0.0,
                workspace.projected_gram);
            for (int column = 0; column < responses.cols(); ++column) {
                for (int row = 0; row < responses.cols(); ++row) {
                    workspace.residual_gram(row, column) -=
                        workspace.projected_gram(row, column);
                }
            }
        }
        for (int column = 0; column < responses.cols(); ++column) {
            workspace.residual_gram(column, column) =
                workspace.residual_energies[column];
            for (int row = 0; row < column; ++row) {
                double value = 0.5 * (
                    workspace.residual_gram(row, column)
                    + workspace.residual_gram(column, row));
                workspace.residual_gram(row, column) = value;
                workspace.residual_gram(column, row) = value;
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    statistics.orthogonalization_seconds +=
        std::chrono::duration<double>(end - start).count();
}

static void draw_diagnostic_block(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options,
    int column_count,
    std::uint64_t first_direction,
    const Matrix& output_basis,
    bool form_gram,
    DiagnosticWorkspace& workspace,
    RunStatistics& statistics) {
    ensure_size(workspace.gaussian, matrix.cols(), column_count);
    math::fill_gaussian(
        workspace.gaussian.data(),
        matrix.cols(),
        column_count,
        options.seed,
        options.stream,
        first_direction);
    bool response_resized = ensure_size(
        workspace.responses, matrix.rows(), column_count);
    if (!response_resized) {
        workspace.responses.fill(0.0);
    }
    math::apply_a_into(
        matrix, workspace.gaussian, workspace.responses, statistics);
    ++statistics.diagnostic_block_calls;
    statistics.diagnostic_columns += column_count;
    analyze_responses(
        workspace.responses,
        output_basis,
        form_gram,
        workspace,
        statistics);
}

static double recent_mean_ratio(
    const math::Certificate& certificate,
    double tolerance_squared) {
    const std::vector<math::CertificateTraceEntry>& trace = certificate.trace();
    const int history = 32;
    if (static_cast<int>(trace.size()) < history) {
        return std::numeric_limits<double>::infinity();
    }

    double ratio = 0.0;
    for (int index = static_cast<int>(trace.size()) - history;
         index < static_cast<int>(trace.size());
         ++index) {
        ratio += trace[index].x / tolerance_squared;
    }
    return ratio / history;
}

static double diagnostic_trigger_threshold(int range_rank) {
    // Fixed empirical policy. The proof only requires this value and the
    // refinement ratio to be fixed from the range rank before the pilot draw.
    if (range_rank <= 128) {
        return 0.925;
    }
    return range_rank <= 256 ? 1.0 : 0.90;
}

static double diagnostic_refinement_ratio(int range_rank) {
    return range_rank <= 128 ? 0.95 : 0.98;
}

static FactorizationResult finish_factorization(
    const MatrixOperator& matrix,
    const Matrix& final_basis,
    int first_delta_column,
    double base_budget_squared,
    double residual_bound_squared,
    StopReason stop_reason,
    RunStatistics statistics,
    const math::Certificate& certificate,
    std::chrono::steady_clock::time_point algorithm_start,
    ResidualBoundSource bound_source = ResidualBoundSource::none,
    const DiagnosticTrace* diagnostic = nullptr,
    std::optional<double> exact_frobenius_norm_squared = std::nullopt) {
    FactorizationResult result;
    result.rows = matrix.rows();
    result.cols = matrix.cols();
    result.range_rank = final_basis.cols();
    result.certificate_max_observations = certificate.max_observations();
    result.statistics = statistics;
    result.statistics.validation_directions =
        result.statistics.a_columns
        - result.statistics.assimilated_directions;
    result.stop_reason = stop_reason;
    result.residual_bound_source = bound_source;
    result.exact_frobenius_side_information_used =
        exact_frobenius_norm_squared.has_value();
    result.residual_bound_squared = residual_bound_squared;
    result.base_truncation_budget_squared = base_budget_squared;
    if (diagnostic != nullptr) {
        result.diagnostic = *diagnostic;
    }

    const std::vector<math::CertificateTraceEntry>& trace = certificate.trace();
    for (const math::CertificateTraceEntry& entry : trace) {
        result.certificate_observations.push_back(entry.x);
        result.certificate_dimensions.push_back(entry.unused_dimension);
        result.certificate_leakages.push_back(entry.leakage);
        for (double scale : entry.predictable_scales) {
            result.certificate_scales.push_back(scale);
        }
    }

    if (final_basis.cols() == 0) {
        if (exact_frobenius_norm_squared.has_value()) {
            double tolerance_squared =
                base_budget_squared + residual_bound_squared;
            result.residual_estimate_squared =
                *exact_frobenius_norm_squared;
            result.truncation_budget_squared = std::max(
                0.0,
                tolerance_squared - *exact_frobenius_norm_squared);
            if (*exact_frobenius_norm_squared > tolerance_squared) {
                result.status = FactorizationStatus::certificate_miss;
            }
        } else {
            result.truncation_budget_squared = base_budget_squared;
        }
        auto end = std::chrono::steady_clock::now();
        result.statistics.total_seconds =
            std::chrono::duration<double>(end - algorithm_start).count();
        return result;
    }

    // The range is fixed. One adjoint panel now gives the whole core.
    Matrix at_basis = math::apply_at(matrix, final_basis, result.statistics);

    double residual_decrease_squared = 0.0;
    if (first_delta_column >= 0) {
        // These columns are A^T H_t, so their squared norm is Delta_t.
        for (int row = 0; row < at_basis.rows(); ++row) {
            for (int col = first_delta_column; col < at_basis.cols(); ++col) {
                double value = at_basis(row, col);
                residual_decrease_squared += value * value;
            }
        }
    }
    result.residual_decrease_squared = residual_decrease_squared;

    auto svd_start = std::chrono::steady_clock::now();
    // If A^T Q = V S U_b^T, then Q U_b S V^T is the final SVD.
    double budget = base_budget_squared + residual_decrease_squared;
    math::TallSvdResult svd;
    if (exact_frobenius_norm_squared.has_value()) {
        double tolerance_squared =
            base_budget_squared + residual_bound_squared;
        svd = math::tall_svd_exact_frobenius_in_place(
            at_basis,
            *exact_frobenius_norm_squared,
            tolerance_squared);
        result.residual_estimate_squared = svd.residual_norm_squared;
        budget = svd.tail_budget_squared;
    } else {
        svd = math::tall_svd_in_place(at_basis, budget);
    }
    result.truncation_budget_squared = budget;
    if (exact_frobenius_norm_squared.has_value() &&
        !svd.exact_frobenius_budget_feasible) {
        result.status = FactorizationStatus::certificate_miss;
        auto svd_end = std::chrono::steady_clock::now();
        result.statistics.svd_seconds +=
            std::chrono::duration<double>(svd_end - svd_start).count();
        auto end = std::chrono::steady_clock::now();
        result.statistics.total_seconds =
            std::chrono::duration<double>(end - algorithm_start).count();
        return result;
    }
    int rank = svd.selected_rank;
    int boundary_only_rank = rank;
    if (first_delta_column >= 0) {
        boundary_only_rank = math::smallest_rank_for_tail(
            svd.singular_values,
            residual_decrease_squared);
    } else if (
        bound_source == ResidualBoundSource::global_certificate ||
        bound_source == ResidualBoundSource::diagnostic_certificate ||
        bound_source == ResidualBoundSource::diagnostic_spectral_cap) {
        boundary_only_rank = final_basis.cols();
    }
    result.boundary_only_rank = boundary_only_rank;

    Matrix core_left_vectors(final_basis.cols(), rank);
    for (int col = 0; col < rank; ++col) {
        for (int row = 0; row < final_basis.cols(); ++row) {
            core_left_vectors(row, col) = svd.vt(col, row);
        }
    }
    Matrix u = math::multiply(final_basis, core_left_vectors);
    auto svd_end = std::chrono::steady_clock::now();
    result.statistics.svd_seconds +=
        std::chrono::duration<double>(svd_end - svd_start).count();

    result.rank = rank;
    result.singular_values.assign(
        svd.singular_values.begin(), svd.singular_values.begin() + rank);
    if (rank > 0) {
        at_basis.truncate_columns(rank);
        u.give_values_to(result.u);
        at_basis.give_values_to(result.v);
    }
    auto end = std::chrono::steady_clock::now();
    result.statistics.total_seconds =
        std::chrono::duration<double>(end - algorithm_start).count();
    return result;
}

static FactorizationResult compute_immediate_ac_rsvd(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options) {
    if (matrix.rows() < 1 || matrix.cols() < 1) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (!std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("Failure probability must be between zero and one");
    }
    if (options.block_size <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }

    int rows = matrix.rows();
    int cols = matrix.cols();
    double tolerance_squared = options.tolerance * options.tolerance;
    auto algorithm_start = std::chrono::steady_clock::now();

    Matrix input_basis(cols, 0);
    Matrix output_basis(rows, 0);
    Matrix input_panel;
    Matrix output_panel;
    Matrix input_r;
    Matrix output_r;
    std::vector<double> input_diagonal;
    std::vector<double> output_diagonal;
    // In the exact model, output_basis spans A times input_basis.
    math::Certificate certificate(
        std::max(2, cols),
        tolerance_squared,
        options.failure_probability);
    RunStatistics statistics;

    while (true) {
        if (output_basis.cols() == rows) {
            input_basis = Matrix();
            input_panel = Matrix();
            output_panel = Matrix();
            return finish_factorization(
                matrix,
                output_basis,
                -1,
                tolerance_squared,
                0.0,
                StopReason::full_output_space,
                statistics,
                certificate,
                algorithm_start,
                ResidualBoundSource::exact_residual,
                nullptr,
                options.exact_frobenius_norm_squared);
        }

        int unused_at_block_start = cols - input_basis.cols();
        int block_size = std::min(options.block_size, unused_at_block_start);

        if (input_panel.rows() != cols || input_panel.cols() != block_size) {
            input_panel = Matrix(cols, block_size);
        }
        if (output_panel.rows() != rows || output_panel.cols() != block_size) {
            output_panel = Matrix(rows, block_size);
        }

        std::uint64_t redraw = 0;
        while (true) {
            math::fill_gaussian(
                input_panel.data(),
                cols,
                block_size,
                options.seed,
                options.stream + redraw,
                static_cast<std::uint64_t>(input_basis.cols()));
            auto orth_start = std::chrono::steady_clock::now();
            orthogonalize_panel_in_place(
                input_panel,
                input_basis,
                options.use_sequential_orthogonalization,
                input_r,
                input_diagonal);
            auto orth_end = std::chrono::steady_clock::now();
            statistics.orthogonalization_seconds +=
                std::chrono::duration<double>(orth_end - orth_start).count();

            bool singular = false;
            for (double diagonal : input_diagonal) {
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

        math::apply_a_into(matrix, input_panel, output_panel, statistics);
        auto orth_start = std::chrono::steady_clock::now();
        orthogonalize_panel_in_place(
            output_panel,
            output_basis,
            options.use_sequential_orthogonalization,
            output_r,
            output_diagonal);
        auto orth_end = std::chrono::steady_clock::now();
        statistics.orthogonalization_seconds +=
            std::chrono::duration<double>(orth_end - orth_start).count();

        // The fixed QR order lets us reveal the block one column at a time.
        for (int column = 0; column < block_size; ++column) {
            if (output_basis.cols() + column == rows) {
                output_basis.append_columns(output_panel, column);
                input_basis = Matrix();
                input_panel = Matrix();
                output_panel = Matrix();
                return finish_factorization(
                    matrix,
                    output_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::full_output_space,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::exact_residual,
                    nullptr,
                    options.exact_frobenius_norm_squared);
            }

            int unused_dimension = unused_at_block_start - column;
            double diagonal = output_diagonal[column];
            ++statistics.directions_processed;
            statistics.assimilated_directions =
                statistics.directions_processed;

            if (unused_dimension == 1) {
                int prefix_size = diagonal > 0.0 ? column + 1 : column;
                output_basis.append_columns(output_panel, prefix_size);
                input_basis = Matrix();
                input_panel = Matrix();
                output_panel = Matrix();
                return finish_factorization(
                    matrix,
                    output_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::final_input_direction,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::exact_residual,
                    nullptr,
                    options.exact_frobenius_norm_squared);
            }

            if (diagonal == 0.0) {
                output_basis.append_columns(output_panel, column);
                input_basis = Matrix();
                input_panel = Matrix();
                output_panel = Matrix();
                return finish_factorization(
                    matrix,
                    output_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::zero_residual,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::exact_residual,
                    nullptr,
                    options.exact_frobenius_norm_squared);
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
                statistics.certificate_bound_round =
                    static_cast<long long>(certificate.trace().size());
                certificate_end = std::chrono::steady_clock::now();
                statistics.certificate_seconds +=
                    std::chrono::duration<double>(
                        certificate_end - certificate_start).count();
                output_basis.append_columns(output_panel, column);

                // The lower-right QR core contains the useful block suffix.
                orth_start = std::chrono::steady_clock::now();
                Matrix trailing = math::trailing_basis(
                    output_panel, output_r, column);
                math::project_out(output_basis, trailing, 2);
                trailing = math::column_space(trailing);
                orth_end = std::chrono::steady_clock::now();
                statistics.orthogonalization_seconds +=
                    std::chrono::duration<double>(
                        orth_end - orth_start).count();

                int first_extra_column = output_basis.cols();
                output_basis.append_columns(trailing);
                trailing = Matrix();
                input_basis = Matrix();
                input_panel = Matrix();
                output_panel = Matrix();

                return finish_factorization(
                    matrix,
                    output_basis,
                    first_extra_column,
                    tolerance_squared - residual_bound_squared,
                    residual_bound_squared,
                    StopReason::certificate,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::global_certificate,
                    nullptr,
                    options.exact_frobenius_norm_squared);
            }
        }

        // A completed panel now becomes part of the persistent state.
        input_basis.append_columns(input_panel);
        output_basis.append_columns(output_panel);
    }
}

static bool run_predictable_diagnostic(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options,
    const Matrix& output_basis,
    double tolerance_squared,
    double refinement_ratio,
    int heldout_budget,
    DiagnosticTrace& trace,
    RunStatistics& statistics) {
    // Fixed empirical diagnostic sizes. Blocking changes work granularity, not
    // the validity of the ordered held-out process.
    const int pilot_size = 32;
    const int heldout_block_size = 32;
    // Fixed empirical allocation. These parts and the global part below sum to
    // the requested failure probability; the proof does not require this split.
    const double diagnostic_failure_probability =
        0.70 * options.failure_probability;
    const double cap_failure_probability =
        0.25 * options.failure_probability;
    const double refinement_target = refinement_ratio * tolerance_squared;

    trace.attempted = true;
    trace.trigger_range_rank = output_basis.cols();
    trace.pilot_directions = pilot_size;

    DiagnosticWorkspace workspace;
    draw_diagnostic_block(
        matrix,
        options,
        pilot_size,
        static_cast<std::uint64_t>(matrix.cols()),
        output_basis,
        true,
        workspace,
        statistics);
    statistics.diagnostic_pilot_columns += pilot_size;
    statistics.directions_processed += pilot_size;

    double pilot_sum = 0.0;
    bool exact_zero_pilot = true;
    for (int column = 0; column < pilot_size; ++column) {
        pilot_sum += workspace.residual_gram(column, column);
        if (workspace.residual_energies[column] != 0.0) {
            exact_zero_pilot = false;
        }
    }
    trace.pilot_mean = pilot_sum / pilot_size;

    auto certificate_start = std::chrono::steady_clock::now();
    double largest_eigenvalue =
        math::largest_symmetric_eigenvalue_in_place(workspace.residual_gram);
    double quantile = math::chi_square_32_lower_quantile(
        cap_failure_probability);
    auto certificate_end = std::chrono::steady_clock::now();
    statistics.certificate_seconds +=
        std::chrono::duration<double>(
            certificate_end - certificate_start).count();

    if (!std::isfinite(pilot_sum) || !std::isfinite(largest_eigenvalue) ||
        !std::isfinite(quantile) || quantile <= 0.0) {
        return false;
    }
    if (largest_eigenvalue <= 0.0) {
        if (largest_eigenvalue == 0.0 && pilot_sum == 0.0 &&
            exact_zero_pilot) {
            trace.crossed = true;
            trace.reached_refinement_target = true;
            trace.residual_bound_squared = 0.0;
            return true;
        }
        return false;
    }

    double spectral_cap = largest_eigenvalue / quantile;
    trace.spectral_cap = spectral_cap;
    if (!std::isfinite(spectral_cap) || spectral_cap <= 0.0) {
        return false;
    }

    double deterministic_bound = matrix.cols() * spectral_cap;
    if (deterministic_bound <= tolerance_squared) {
        trace.crossed = true;
        trace.reached_refinement_target =
            deterministic_bound <= refinement_target;
        trace.residual_bound_squared = deterministic_bound;
        return true;
    }
    if (trace.pilot_mean <= 0.0 ||
        trace.pilot_mean >= tolerance_squared ||
        tolerance_squared > deterministic_bound) {
        return false;
    }

    certificate_start = std::chrono::steady_clock::now();
    double gamma = math::choose_raw_gaussian_gamma(
        matrix.cols(),
        trace.pilot_mean,
        spectral_cap,
        tolerance_squared);
    certificate_end = std::chrono::steady_clock::now();
    statistics.certificate_seconds +=
        std::chrono::duration<double>(
            certificate_end - certificate_start).count();
    if (!std::isfinite(gamma)) {
        return false;
    }
    trace.gamma = gamma;
    trace.heldout_activated = true;

    int heldout_count = 0;
    double heldout_sum = 0.0;
    double first_crossing_sum = 0.0;
    while (heldout_count < heldout_budget) {
        bool finish_after_block = false;
        double best_bound = std::numeric_limits<double>::infinity();
        int best_bound_count = 0;
        double best_bound_sum = 0.0;
        int count = std::min(
            heldout_block_size, heldout_budget - heldout_count);
        draw_diagnostic_block(
            matrix,
            options,
            count,
            static_cast<std::uint64_t>(matrix.cols())
                + pilot_size + heldout_count,
            output_basis,
            false,
            workspace,
            statistics);
        for (int column = 0; column < count; ++column) {
            heldout_sum += workspace.residual_energies[column];
            ++heldout_count;
            ++statistics.diagnostic_heldout_columns;
            ++statistics.directions_processed;
            ++trace.heldout_endpoints;
            trace.heldout_directions = heldout_count;

            certificate_start = std::chrono::steady_clock::now();
            if (trace.first_crossing_direction == 0) {
                bool crossed = math::raw_gaussian_crosses(
                    heldout_count,
                    heldout_sum,
                    matrix.cols(),
                    spectral_cap,
                    gamma,
                    tolerance_squared,
                    diagnostic_failure_probability);
                if (crossed) {
                    trace.crossed = true;
                    trace.first_crossing_direction = heldout_count;
                    trace.first_crossing_sum = heldout_sum;
                    first_crossing_sum = heldout_sum;
                }
            }
            if (!finish_after_block) {
                bool reached_target = math::raw_gaussian_crosses(
                    heldout_count,
                    heldout_sum,
                    matrix.cols(),
                    spectral_cap,
                    gamma,
                    refinement_target,
                    diagnostic_failure_probability);
                if (reached_target) {
                    trace.reached_refinement_target = true;
                    finish_after_block = true;
                    best_bound_count = heldout_count;
                    best_bound_sum = heldout_sum;
                    best_bound =
                        math::raw_gaussian_continuous_inverse(
                            heldout_count,
                            heldout_sum,
                            matrix.cols(),
                            spectral_cap,
                            gamma,
                            diagnostic_failure_probability,
                            refinement_target);
                }
            } else if (math::raw_gaussian_crosses(
                           heldout_count,
                           heldout_sum,
                           matrix.cols(),
                           spectral_cap,
                           gamma,
                           best_bound,
                           diagnostic_failure_probability)) {
                best_bound = math::raw_gaussian_continuous_inverse(
                    heldout_count,
                    heldout_sum,
                    matrix.cols(),
                    spectral_cap,
                    gamma,
                    diagnostic_failure_probability,
                    best_bound);
                best_bound_count = heldout_count;
                best_bound_sum = heldout_sum;
            }
            certificate_end = std::chrono::steady_clock::now();
            statistics.certificate_seconds +=
                std::chrono::duration<double>(
                    certificate_end - certificate_start).count();
        }
        if (finish_after_block) {
            trace.bound_direction = best_bound_count;
            trace.bound_sum = best_bound_sum;
            trace.residual_bound_squared = best_bound;
            return true;
        }
    }
    if (trace.first_crossing_direction > 0) {
        auto certificate_start = std::chrono::steady_clock::now();
        trace.bound_direction = trace.first_crossing_direction;
        trace.bound_sum = first_crossing_sum;
        trace.residual_bound_squared =
            math::raw_gaussian_continuous_inverse(
                trace.first_crossing_direction,
                first_crossing_sum,
                matrix.cols(),
                trace.spectral_cap,
                trace.gamma,
                diagnostic_failure_probability,
                tolerance_squared);
        auto certificate_end = std::chrono::steady_clock::now();
        statistics.certificate_seconds +=
            std::chrono::duration<double>(
                certificate_end - certificate_start).count();
        return true;
    }
    return false;
}

static FactorizationResult compute_enhanced_iid_ac_rsvd(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options) {
    // Fixed empirical trigger history. It is read before any diagnostic draw.
    const int history_size = 32;
    int rows = matrix.rows();
    int cols = matrix.cols();
    double tolerance_squared = options.tolerance * options.tolerance;
    // This is the remaining part of the fixed enhanced failure allocation.
    double global_failure_probability =
        0.05 * options.failure_probability;
    int heldout_budget = options.diagnostic_test_size;

    auto algorithm_start = std::chrono::steady_clock::now();
    Matrix output_basis(rows, 0);
    math::Certificate certificate(
        std::max(2, cols),
        tolerance_squared,
        global_failure_probability,
        std::max(1, cols));
    RunStatistics statistics;
    DiagnosticTrace diagnostic;
    bool diagnostic_attempted = false;

    if (cols == 1) {
        Matrix gaussian = draw_raw_gaussian(options, 1, 1, 0);
        Matrix response(rows, 1);
        math::apply_a_into(matrix, gaussian, response, statistics);
        ++statistics.ordinary_block_calls;
        ++statistics.ordinary_columns;

        Matrix output_r;
        std::vector<double> output_diagonal;
        auto orthogonalization_start = std::chrono::steady_clock::now();
        orthogonalize_panel_in_place(
            response,
            output_basis,
            options.use_sequential_orthogonalization,
            output_r,
            output_diagonal);
        auto orthogonalization_end = std::chrono::steady_clock::now();
        statistics.orthogonalization_seconds +=
            std::chrono::duration<double>(
                orthogonalization_end - orthogonalization_start).count();
        ++statistics.ordinary_observations;
        ++statistics.directions_processed;

        StopReason stop_reason = StopReason::zero_residual;
        if (output_diagonal[0] > 0.0) {
            output_basis.append_columns(response);
            ++statistics.ordinary_assimilated;
            ++statistics.assimilated_directions;
            stop_reason = StopReason::final_input_direction;
        } else {
            ++statistics.ordinary_discarded;
        }
        return finish_factorization(
            matrix,
            output_basis,
            -1,
            tolerance_squared,
            0.0,
            stop_reason,
            statistics,
            certificate,
            algorithm_start,
            ResidualBoundSource::exact_residual,
            &diagnostic,
            options.exact_frobenius_norm_squared);
    }

    std::vector<double> input_energies;
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
                algorithm_start,
                ResidualBoundSource::exact_residual,
                &diagnostic,
                options.exact_frobenius_norm_squared);
        }

        if (!diagnostic_attempted && heldout_budget > 0 &&
            static_cast<int>(certificate.trace().size()) >= history_size) {
            double trigger_ratio = recent_mean_ratio(
                certificate, tolerance_squared);
            int range_rank = output_basis.cols();
            double threshold = diagnostic_trigger_threshold(range_rank);
            double refinement_ratio = diagnostic_refinement_ratio(range_rank);
            if (trigger_ratio < threshold) {
                diagnostic_attempted = true;
                diagnostic.trigger_mean_ratio = trigger_ratio;
                bool diagnostic_success = run_predictable_diagnostic(
                    matrix,
                    options,
                    output_basis,
                    tolerance_squared,
                    refinement_ratio,
                    heldout_budget,
                    diagnostic,
                    statistics);
                if (diagnostic_success) {
                    statistics.certificate_bound_round = 0;
                    ResidualBoundSource source =
                        ResidualBoundSource::diagnostic_certificate;
                    if (diagnostic.bound_direction == 0) {
                        source = diagnostic.residual_bound_squared == 0.0
                            ? ResidualBoundSource::exact_residual
                            : ResidualBoundSource::diagnostic_spectral_cap;
                    }
                    return finish_factorization(
                        matrix,
                        output_basis,
                        -1,
                        std::max(
                            0.0,
                            tolerance_squared
                                - diagnostic.residual_bound_squared),
                        diagnostic.residual_bound_squared,
                        StopReason::certificate,
                        statistics,
                        certificate,
                        algorithm_start,
                        source,
                        &diagnostic,
                        options.exact_frobenius_norm_squared);
                }
            }
        }

        int remaining_directions = cols -
            static_cast<int>(statistics.ordinary_observations);
        int remaining_output = rows - output_basis.cols();
        int block_size = std::min(
            options.block_size,
            std::min(remaining_directions, remaining_output));

        Matrix gaussian = draw_raw_gaussian(
            options,
            cols,
            block_size,
            static_cast<std::uint64_t>(statistics.ordinary_observations));
        raw_input_energies(gaussian, input_energies, statistics);
        for (double energy : input_energies) {
            if (!std::isfinite(energy) || energy <= 0.0) {
                throw std::runtime_error(
                    "Gaussian input column has invalid energy");
            }
        }
        Matrix responses(rows, block_size);
        math::apply_a_into(matrix, gaussian, responses, statistics);
        ++statistics.ordinary_block_calls;
        statistics.ordinary_columns += block_size;

        Matrix output_r;
        std::vector<double> output_diagonal;
        auto orthogonalization_start = std::chrono::steady_clock::now();
        orthogonalize_panel_in_place(
            responses,
            output_basis,
            options.use_sequential_orthogonalization,
            output_r,
            output_diagonal);
        auto orthogonalization_end = std::chrono::steady_clock::now();
        statistics.orthogonalization_seconds +=
            std::chrono::duration<double>(
                orthogonalization_end - orthogonalization_start).count();

        for (int column = 0; column < block_size; ++column) {
            double diagonal = output_diagonal[column];
            ++statistics.ordinary_observations;
            ++statistics.directions_processed;

            if (diagonal == 0.0) {
                output_basis.append_columns(responses, column);
                statistics.ordinary_assimilated += column;
                statistics.assimilated_directions += column;
                statistics.ordinary_discarded += block_size - column;
                return finish_factorization(
                    matrix,
                    output_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::zero_residual,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::exact_residual,
                    &diagnostic,
                    options.exact_frobenius_norm_squared);
            }

            double observation =
                cols * diagonal * diagonal / input_energies[column];
            auto certificate_start = std::chrono::steady_clock::now();
            bool crossed = certificate.update(observation, cols);
            auto certificate_end = std::chrono::steady_clock::now();
            statistics.certificate_seconds +=
                std::chrono::duration<double>(
                    certificate_end - certificate_start).count();

            if (crossed) {
                certificate_start = std::chrono::steady_clock::now();
                double residual_bound_squared =
                    certificate.continuous_inverse();
                certificate_end = std::chrono::steady_clock::now();
                statistics.certificate_seconds +=
                    std::chrono::duration<double>(
                        certificate_end - certificate_start).count();
                statistics.certificate_bound_round =
                    static_cast<long long>(certificate.trace().size());

                output_basis.append_columns(responses, column);
                statistics.ordinary_assimilated += column;
                statistics.assimilated_directions += column;
                statistics.ordinary_discarded += block_size - column;
                return finish_factorization(
                    matrix,
                    output_basis,
                    -1,
                    tolerance_squared - residual_bound_squared,
                    residual_bound_squared,
                    StopReason::certificate,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::global_certificate,
                    &diagnostic,
                    options.exact_frobenius_norm_squared);
            }

            if (statistics.ordinary_observations == cols) {
                output_basis.append_columns(responses, column + 1);
                statistics.ordinary_assimilated += column + 1;
                statistics.assimilated_directions += column + 1;
                statistics.ordinary_discarded += block_size - column - 1;
                return finish_factorization(
                    matrix,
                    output_basis,
                    -1,
                    tolerance_squared,
                    0.0,
                    StopReason::final_input_direction,
                    statistics,
                    certificate,
                    algorithm_start,
                    ResidualBoundSource::exact_residual,
                    &diagnostic,
                    options.exact_frobenius_norm_squared);
            }
        }

        output_basis.append_columns(responses);
        statistics.ordinary_assimilated += block_size;
        statistics.assimilated_directions += block_size;
    }
}

FactorizationResult compute_ac_rsvd(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options) {
    if (matrix.rows() < 1 || matrix.cols() < 1) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (!std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument("Failure probability must be between zero and one");
    }
    if (options.block_size <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }
    if (options.diagnostic_test_size < 0 ||
        options.diagnostic_test_size > 512) {
        throw std::invalid_argument(
            "Diagnostic test size must be between zero and 512");
    }
    if (options.exact_frobenius_norm_squared.has_value() &&
        (!std::isfinite(*options.exact_frobenius_norm_squared) ||
         *options.exact_frobenius_norm_squared < 0.0)) {
        throw std::invalid_argument(
            "Exact Frobenius norm squared must be finite and nonnegative");
    }

    if (options.use_enhanced_mode) {
        return compute_enhanced_iid_ac_rsvd(matrix, options);
    }
    return compute_immediate_ac_rsvd(matrix, options);
}

}  // namespace ac_rsvd
