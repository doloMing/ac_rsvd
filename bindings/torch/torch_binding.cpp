#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "ac_rsvd/ac_rsvd.hpp"
#include "ac_rsvd/randqb_ei.hpp"
#include "ac_rsvd/randqb_mf_fro.hpp"
#include "algorithms/mathematics/analysis/theory_bounds.hpp"
#include "algorithms/mathematics/certificate/certificate.hpp"
#include "algorithms/mathematics/certificate/raw_gaussian_certificate.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/operators/dense_operator.hpp"
#include "algorithms/mathematics/operators/structured_hadamard_operator.hpp"

namespace ac_rsvd {
namespace {

using TorchResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::int64_t>;

using TorchExactFroResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::int64_t,
    std::int64_t>;

using TorchTheoryResult = std::tuple<at::Tensor, at::Tensor>;

math::Matrix copy_input(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_cpu(), "matrix must be on the CPU");
    TORCH_CHECK(input.scalar_type() == at::kDouble, "matrix must use float64");
    TORCH_CHECK(input.dim() == 2, "matrix must have two dimensions");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0,
                "matrix dimensions must be positive");
    TORCH_CHECK(
        input.size(0) <= std::numeric_limits<int>::max() &&
        input.size(1) <= std::numeric_limits<int>::max(),
        "matrix dimensions exceed the LP64 backend limit");

    int rows = static_cast<int>(input.size(0));
    int cols = static_cast<int>(input.size(1));
    math::Matrix matrix(rows, cols);
    auto values = input.accessor<double, 2>();
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            matrix(row, col) = values[row][col];
        }
    }
    return matrix;
}

int copy_block_size(std::int64_t block_size) {
    TORCH_CHECK(
        block_size > 0 && block_size <= std::numeric_limits<int>::max(),
        "block_size must fit a positive LP64 integer");
    return static_cast<int>(block_size);
}

std::vector<double> copy_spectrum(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_cpu(), "spectrum must be on the CPU");
    TORCH_CHECK(input.scalar_type() == at::kDouble, "spectrum must use float64");
    TORCH_CHECK(input.dim() == 1, "spectrum must have one dimension");
    TORCH_CHECK(input.size(0) > 0, "spectrum cannot be empty");
    TORCH_CHECK(
        input.size(0) <= std::numeric_limits<int>::max(),
        "spectrum exceeds the LP64 backend limit");

    auto values = input.accessor<double, 1>();
    std::vector<double> spectrum(input.size(0));
    for (std::int64_t index = 0; index < input.size(0); ++index) {
        spectrum[index] = values[index];
    }
    return spectrum;
}

at::Tensor copy_matrix(
    const std::vector<double>& values,
    int rows,
    int cols) {
    at::Tensor tensor = at::empty({rows, cols}, at::kDouble);
    auto output = tensor.accessor<double, 2>();
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            output[row][col] = values[row + col * rows];
        }
    }
    return tensor;
}

TorchResult copy_result(const FactorizationResult& result) {
    at::Tensor u = copy_matrix(result.u, result.rows, result.rank);
    at::Tensor singular_values = at::empty({result.rank}, at::kDouble);
    double* singular_data = singular_values.data_ptr<double>();
    for (int index = 0; index < result.rank; ++index) {
        singular_data[index] = result.singular_values[index];
    }
    at::Tensor v = copy_matrix(result.v, result.cols, result.rank);

    at::Tensor counters = at::empty({26}, at::kLong);
    std::int64_t* counter_data = counters.data_ptr<std::int64_t>();
    counter_data[0] = result.statistics.directions_processed;
    counter_data[1] = result.statistics.a_columns;
    counter_data[2] = result.statistics.at_columns;
    counter_data[3] = result.statistics.a_block_calls;
    counter_data[4] = result.statistics.at_block_calls;
    counter_data[5] = result.range_rank;
    counter_data[6] = result.boundary_only_rank;
    counter_data[7] = result.statistics.assimilated_directions;
    counter_data[8] = result.statistics.validation_directions;
    counter_data[9] = result.statistics.certificate_bound_round;
    counter_data[10] = result.certificate_max_observations;
    counter_data[11] = result.statistics.ordinary_columns;
    counter_data[12] = result.statistics.ordinary_block_calls;
    counter_data[13] = result.statistics.ordinary_observations;
    counter_data[14] = result.statistics.ordinary_assimilated;
    counter_data[15] = result.statistics.ordinary_discarded;
    counter_data[16] = result.statistics.diagnostic_columns;
    counter_data[17] = result.statistics.diagnostic_block_calls;
    counter_data[18] = result.statistics.diagnostic_pilot_columns;
    counter_data[19] = result.statistics.diagnostic_heldout_columns;
    counter_data[20] = result.diagnostic.trigger_range_rank;
    counter_data[21] = result.diagnostic.heldout_directions;
    counter_data[22] = result.diagnostic.heldout_endpoints;
    counter_data[23] = result.diagnostic.first_crossing_direction;
    counter_data[24] = result.diagnostic.bound_direction;
    counter_data[25] = static_cast<std::int64_t>(
        result.residual_bound_source);

    at::Tensor diagnostics = at::empty({16}, at::kDouble);
    double* diagnostic_data = diagnostics.data_ptr<double>();
    diagnostic_data[0] = result.residual_estimate_squared;
    diagnostic_data[1] = result.residual_bound_squared;
    diagnostic_data[2] = result.truncation_budget_squared;
    diagnostic_data[3] = result.residual_decrease_squared;
    diagnostic_data[4] = result.direct_error_squared;
    diagnostic_data[5] = result.base_truncation_budget_squared;
    diagnostic_data[6] = result.diagnostic.attempted ? 1.0 : 0.0;
    diagnostic_data[7] = result.diagnostic.heldout_activated ? 1.0 : 0.0;
    diagnostic_data[8] = result.diagnostic.crossed ? 1.0 : 0.0;
    diagnostic_data[9] =
        result.diagnostic.reached_refinement_target ? 1.0 : 0.0;
    diagnostic_data[10] = result.diagnostic.trigger_mean_ratio;
    diagnostic_data[11] = result.diagnostic.pilot_mean;
    diagnostic_data[12] = result.diagnostic.spectral_cap;
    diagnostic_data[13] = result.diagnostic.gamma;
    diagnostic_data[14] = result.diagnostic.bound_sum;
    diagnostic_data[15] = result.diagnostic.first_crossing_sum;

    at::Tensor timings = at::empty({6}, at::kDouble);
    double* timing_data = timings.data_ptr<double>();
    timing_data[0] = result.statistics.total_seconds;
    timing_data[1] = result.statistics.a_seconds;
    timing_data[2] = result.statistics.at_seconds;
    timing_data[3] = result.statistics.orthogonalization_seconds;
    timing_data[4] = result.statistics.certificate_seconds;
    timing_data[5] = result.statistics.svd_seconds;

    std::int64_t trace_size =
        static_cast<std::int64_t>(result.certificate_observations.size());
    at::Tensor trace = at::empty({trace_size, 7}, at::kDouble);
    auto trace_values = trace.accessor<double, 2>();
    for (std::int64_t index = 0; index < trace_size; ++index) {
        trace_values[index][0] = result.certificate_observations[index];
        trace_values[index][1] = result.certificate_dimensions[index];
        trace_values[index][2] = result.certificate_leakages[index];
        for (int scale = 0; scale < 4; ++scale) {
            trace_values[index][3 + scale] =
                result.certificate_scales[4 * index + scale];
        }
    }

    return std::make_tuple(
        u,
        singular_values,
        v,
        counters,
        diagnostics,
        timings,
        trace,
        static_cast<std::int64_t>(result.stop_reason));
}

TorchExactFroResult copy_exact_fro_result(
    const FactorizationResult& result) {
    TorchResult copied = copy_result(result);
    return std::make_tuple(
        std::get<0>(copied),
        std::get<1>(copied),
        std::get<2>(copied),
        std::get<3>(copied),
        std::get<4>(copied),
        std::get<5>(copied),
        std::get<6>(copied),
        std::get<7>(copied),
        static_cast<std::int64_t>(result.status));
}

double structured_error_squared(
    const math::StructuredHadamardOperator& matrix,
    const FactorizationResult& result) {
    double matrix_norm = matrix.frobenius_norm();
    if (result.rank == 0) {
        return matrix_norm * matrix_norm;
    }

    math::Matrix u(result.rows, result.rank, result.u.data());
    math::Matrix v(result.cols, result.rank, result.v.data());
    math::Matrix av(result.rows, result.rank);
    matrix.apply(v.data(), v.cols(), av.data());

    math::Matrix gram_u = math::transpose_multiply(u, u);
    math::Matrix gram_v = math::transpose_multiply(v, v);
    long double approximation_norm_squared = 0.0L;
    for (int col = 0; col < result.rank; ++col) {
        for (int row = 0; row < result.rank; ++row) {
            approximation_norm_squared +=
                result.singular_values[row]
                * result.singular_values[col]
                * gram_u(row, col)
                * gram_v(row, col);
        }
    }

    long double inner_product = 0.0L;
    for (int col = 0; col < result.rank; ++col) {
        long double column_product = 0.0L;
        for (int row = 0; row < result.rows; ++row) {
            column_product += u(row, col) * av(row, col);
        }
        inner_product += result.singular_values[col] * column_product;
    }

    long double error_squared =
        static_cast<long double>(matrix_norm) * matrix_norm
        + approximation_norm_squared
        - 2.0L * inner_product;
    return std::max(0.0, static_cast<double>(error_squared));
}

math::Certificate certificate_from_trace(
    const at::Tensor& trace,
    std::int64_t input_dimension,
    double tolerance,
    double failure_probability,
    std::int64_t max_observations) {
    TORCH_CHECK(trace.device().is_cpu(), "trace must be on the CPU");
    TORCH_CHECK(trace.scalar_type() == at::kDouble, "trace must use float64");
    TORCH_CHECK(
        trace.dim() == 2 && (trace.size(1) == 2 || trace.size(1) == 7),
        "trace must have shape (rounds, 2) or (rounds, 7)");
    TORCH_CHECK(
        input_dimension >= 2 &&
        input_dimension <= std::numeric_limits<int>::max(),
        "input_dimension must fit an LP64 integer above one");
    TORCH_CHECK(
        max_observations >= 1 &&
        max_observations <= std::numeric_limits<int>::max(),
        "max_observations must fit a positive LP64 integer");

    int horizon = static_cast<int>(max_observations);

    math::Certificate certificate(
        static_cast<int>(input_dimension),
        tolerance * tolerance,
        failure_probability,
        horizon);
    auto values = trace.accessor<double, 2>();
    for (std::int64_t row = 0; row < trace.size(0); ++row) {
        double dimension_value = values[row][1];
        TORCH_CHECK(
            std::isfinite(dimension_value) &&
            std::floor(dimension_value) == dimension_value &&
            dimension_value >= 2.0 &&
            dimension_value <= input_dimension,
            "trace dimensions must be integers between two and input_dimension");
        int unused_dimension = static_cast<int>(dimension_value);
        if (trace.size(1) == 2) {
            certificate.update(
                values[row][0],
                unused_dimension);
        } else {
            math::CertificateTraceEntry entry;
            entry.x = values[row][0];
            entry.unused_dimension = unused_dimension;
            entry.leakage = values[row][2];
            for (int scale = 0; scale < 4; ++scale) {
                entry.predictable_scales[scale] = values[row][3 + scale];
            }
            certificate.update_from_trace(entry);
        }
    }
    return certificate;
}

double replay_certificate(
    const at::Tensor& trace,
    std::int64_t input_dimension,
    double tolerance,
    double failure_probability,
    double candidate,
    std::int64_t max_observations) {
    math::Certificate certificate = certificate_from_trace(
        trace,
        input_dimension,
        tolerance,
        failure_probability,
        max_observations);
    return certificate.replay(candidate);
}

double invert_certificate(
    const at::Tensor& trace,
    std::int64_t input_dimension,
    double tolerance,
    double failure_probability,
    std::int64_t max_observations) {
    math::Certificate certificate = certificate_from_trace(
        trace,
        input_dimension,
        tolerance,
        failure_probability,
        max_observations);
    return certificate.continuous_inverse();
}

double replay_raw_gaussian_certificate(
    std::int64_t count,
    double sum,
    std::int64_t dimension,
    double spectral_cap,
    double gamma,
    double candidate) {
    TORCH_CHECK(
        count >= 0 && count <= std::numeric_limits<int>::max(),
        "count must fit a nonnegative LP64 integer");
    TORCH_CHECK(
        dimension >= 1 && dimension <= std::numeric_limits<int>::max(),
        "dimension must fit a positive LP64 integer");
    return std::exp(math::raw_gaussian_log_value(
        static_cast<int>(count),
        sum,
        static_cast<int>(dimension),
        spectral_cap,
        gamma,
        candidate));
}

double invert_raw_gaussian_certificate(
    std::int64_t count,
    double sum,
    std::int64_t dimension,
    double spectral_cap,
    double gamma,
    double failure_probability,
    double upper) {
    TORCH_CHECK(
        count >= 0 && count <= std::numeric_limits<int>::max(),
        "count must fit a nonnegative LP64 integer");
    TORCH_CHECK(
        dimension >= 1 && dimension <= std::numeric_limits<int>::max(),
        "dimension must fit a positive LP64 integer");
    return math::raw_gaussian_continuous_inverse(
        static_cast<int>(count),
        sum,
        static_cast<int>(dimension),
        spectral_cap,
        gamma,
        failure_probability,
        upper);
}

TorchResult run_ac_rsvd(
    const at::Tensor& input,
    double tolerance,
    double failure_probability,
    std::int64_t block_size,
    std::int64_t seed,
    std::int64_t stream,
    bool sequential_orthogonalization,
    bool enhanced_mode,
    std::int64_t diagnostic_test_size) {
    math::Matrix matrix = copy_input(input);
    math::DenseOperator op(matrix);

    AcRsvdOptions options;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(seed);
    options.stream = static_cast<std::uint64_t>(stream);
    options.use_sequential_orthogonalization = sequential_orthogonalization;
    options.use_enhanced_mode = enhanced_mode;
    TORCH_CHECK(
        diagnostic_test_size >= 0 && diagnostic_test_size <= 512,
        "diagnostic_test_size must be between zero and 512");
    options.diagnostic_test_size =
        static_cast<int>(diagnostic_test_size);
    return copy_result(compute_ac_rsvd(op, options));
}

TorchExactFroResult run_ac_rsvd_fro(
    const at::Tensor& input,
    double tolerance,
    double failure_probability,
    double frobenius_norm_squared,
    std::int64_t block_size,
    std::int64_t seed,
    std::int64_t stream,
    bool sequential_orthogonalization,
    bool enhanced_mode,
    std::int64_t diagnostic_test_size) {
    math::Matrix matrix = copy_input(input);
    math::DenseOperator op(matrix);

    AcRsvdOptions options;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.exact_frobenius_norm_squared = frobenius_norm_squared;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(seed);
    options.stream = static_cast<std::uint64_t>(stream);
    options.use_sequential_orthogonalization = sequential_orthogonalization;
    options.use_enhanced_mode = enhanced_mode;
    TORCH_CHECK(
        diagnostic_test_size >= 0 && diagnostic_test_size <= 512,
        "diagnostic_test_size must be between zero and 512");
    options.diagnostic_test_size = static_cast<int>(diagnostic_test_size);
    return copy_exact_fro_result(compute_ac_rsvd(op, options));
}

TorchResult run_randqb_mf_fro(
    const at::Tensor& input,
    double tolerance,
    std::int64_t block_size,
    std::int64_t seed,
    std::int64_t stream) {
    math::Matrix matrix = copy_input(input);
    math::DenseOperator op(matrix);

    RandQbMfFroOptions options;
    options.tolerance = tolerance;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(seed);
    options.stream = static_cast<std::uint64_t>(stream);
    return copy_result(compute_randqb_mf_fro(op, options));
}

TorchResult run_randqb_ei(
    const at::Tensor& input,
    double tolerance,
    double frobenius_norm,
    std::int64_t block_size,
    std::int64_t seed,
    std::int64_t stream) {
    math::Matrix matrix = copy_input(input);
    math::DenseOperator op(matrix);

    RandQbEiOptions options;
    options.tolerance = tolerance;
    options.frobenius_norm = frobenius_norm;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(seed);
    options.stream = static_cast<std::uint64_t>(stream);
    return copy_result(compute_randqb_ei(op, options));
}

TorchResult run_ac_rsvd_hadamard(
    const at::Tensor& spectrum,
    double tolerance,
    double failure_probability,
    std::int64_t block_size,
    std::int64_t matrix_seed,
    std::int64_t algorithm_seed,
    std::int64_t stream,
    bool sequential_orthogonalization,
    bool enhanced_mode,
    std::int64_t diagnostic_test_size) {
    auto setup_start = std::chrono::steady_clock::now();
    math::StructuredHadamardOperator op(
        copy_spectrum(spectrum),
        static_cast<std::uint64_t>(matrix_seed));

    AcRsvdOptions options;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(algorithm_seed);
    options.stream = static_cast<std::uint64_t>(stream);
    options.use_sequential_orthogonalization = sequential_orthogonalization;
    options.use_enhanced_mode = enhanced_mode;
    TORCH_CHECK(
        diagnostic_test_size >= 0 && diagnostic_test_size <= 512,
        "diagnostic_test_size must be between zero and 512");
    options.diagnostic_test_size =
        static_cast<int>(diagnostic_test_size);

    auto setup_end = std::chrono::steady_clock::now();
    FactorizationResult result = compute_ac_rsvd(op, options);
    result.statistics.total_seconds +=
        std::chrono::duration<double>(setup_end - setup_start).count();
    result.direct_error_squared = structured_error_squared(op, result);
    return copy_result(result);
}

TorchExactFroResult run_ac_rsvd_fro_hadamard(
    const at::Tensor& spectrum,
    double tolerance,
    double failure_probability,
    std::int64_t block_size,
    std::int64_t matrix_seed,
    std::int64_t algorithm_seed,
    std::int64_t stream,
    bool sequential_orthogonalization,
    std::int64_t diagnostic_test_size) {
    auto setup_start = std::chrono::steady_clock::now();
    math::StructuredHadamardOperator op(
        copy_spectrum(spectrum),
        static_cast<std::uint64_t>(matrix_seed));

    AcRsvdOptions options;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(algorithm_seed);
    options.stream = static_cast<std::uint64_t>(stream);
    options.use_sequential_orthogonalization = sequential_orthogonalization;
    options.use_enhanced_mode = true;
    TORCH_CHECK(
        diagnostic_test_size >= 0 && diagnostic_test_size <= 512,
        "diagnostic_test_size must be between zero and 512");
    options.diagnostic_test_size = static_cast<int>(diagnostic_test_size);
    double frobenius_norm = op.frobenius_norm();
    options.exact_frobenius_norm_squared =
        frobenius_norm * frobenius_norm;

    auto setup_end = std::chrono::steady_clock::now();
    FactorizationResult result = compute_ac_rsvd(op, options);
    result.statistics.total_seconds +=
        std::chrono::duration<double>(setup_end - setup_start).count();
    if (result.status == FactorizationStatus::success) {
        result.direct_error_squared = structured_error_squared(op, result);
    }
    return copy_exact_fro_result(result);
}

TorchResult run_randqb_mf_fro_hadamard(
    const at::Tensor& spectrum,
    double tolerance,
    std::int64_t block_size,
    std::int64_t matrix_seed,
    std::int64_t algorithm_seed,
    std::int64_t stream) {
    auto setup_start = std::chrono::steady_clock::now();
    math::StructuredHadamardOperator op(
        copy_spectrum(spectrum),
        static_cast<std::uint64_t>(matrix_seed));

    RandQbMfFroOptions options;
    options.tolerance = tolerance;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(algorithm_seed);
    options.stream = static_cast<std::uint64_t>(stream);

    auto setup_end = std::chrono::steady_clock::now();
    FactorizationResult result = compute_randqb_mf_fro(op, options);
    result.statistics.total_seconds +=
        std::chrono::duration<double>(setup_end - setup_start).count();
    result.direct_error_squared = structured_error_squared(op, result);
    return copy_result(result);
}

TorchResult run_randqb_ei_hadamard(
    const at::Tensor& spectrum,
    double tolerance,
    std::int64_t block_size,
    std::int64_t matrix_seed,
    std::int64_t algorithm_seed,
    std::int64_t stream) {
    auto setup_start = std::chrono::steady_clock::now();
    math::StructuredHadamardOperator op(
        copy_spectrum(spectrum),
        static_cast<std::uint64_t>(matrix_seed));

    RandQbEiOptions options;
    options.tolerance = tolerance;
    options.frobenius_norm = op.frobenius_norm();
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(algorithm_seed);
    options.stream = static_cast<std::uint64_t>(stream);

    auto setup_end = std::chrono::steady_clock::now();
    FactorizationResult result = compute_randqb_ei(op, options);
    result.statistics.total_seconds +=
        std::chrono::duration<double>(setup_end - setup_start).count();
    result.direct_error_squared = structured_error_squared(op, result);
    return copy_result(result);
}

TorchTheoryResult theory_bounds_e5_analytic(
    const at::Tensor& spectrum,
    std::int64_t rows,
    std::int64_t cols,
    double tolerance,
    double failure_probability,
    double range_failure_probability,
    double certificate_failure_probability,
    double rho,
    double alpha,
    std::int64_t block_size) {
    TORCH_CHECK(
        rows > 0 && rows <= std::numeric_limits<int>::max() &&
        cols > 0 && cols <= std::numeric_limits<int>::max(),
        "matrix dimensions must fit positive LP64 integers");

    int block = copy_block_size(block_size);
    math::AnalyticTheoryBounds result = math::evaluate_theory_bounds_e5(
        copy_spectrum(spectrum),
        static_cast<int>(rows),
        static_cast<int>(cols),
        tolerance,
        failure_probability,
        range_failure_probability,
        certificate_failure_probability,
        rho,
        alpha,
        block);

    // Rows are the deterministic, Theorem 2 + E.5, and Corollary 2.1 + E.5 bounds.
    // Columns are directions, rounded columns, and block calls.
    at::Tensor bounds = at::empty({3, 3}, at::kLong);
    auto bound_values = bounds.accessor<std::int64_t, 2>();
    bound_values[0][0] = result.deterministic_cap;
    bound_values[0][1] = math::block_rounded_columns(
        static_cast<int>(cols), block, result.deterministic_cap);
    bound_values[0][2] = math::block_call_bound(
        block, result.deterministic_cap);

    bound_values[1][0] = result.theorem2_e5.directions;
    bound_values[1][1] = result.theorem2_e5.forward_columns;
    bound_values[1][2] = result.theorem2_e5.forward_block_calls;
    bound_values[2][0] = result.corollary21_e5.directions;
    bound_values[2][1] = result.corollary21_e5.forward_columns;
    bound_values[2][2] = result.corollary21_e5.forward_block_calls;

    // Each row stores cap flag, k, p, k+p, delay, and r_alpha.
    at::Tensor selected = at::empty({2, 6}, at::kLong);
    auto selected_values = selected.accessor<std::int64_t, 2>();
    const math::AnalyticTheoryBound* rows_to_copy[2] = {
        &result.theorem2_e5,
        &result.corollary21_e5,
    };
    for (int row = 0; row < 2; ++row) {
        const math::AnalyticTheoryBound& value = *rows_to_copy[row];
        selected_values[row][0] = value.uses_deterministic_cap ? 1 : 0;
        selected_values[row][1] = value.k;
        selected_values[row][2] = value.p;
        selected_values[row][3] = value.range_directions;
        selected_values[row][4] = value.appendix_e5_delay;
        selected_values[row][5] = value.tolerance_rank;
    }
    return std::make_tuple(bounds, selected);
}

}  // namespace
}  // namespace ac_rsvd

TORCH_LIBRARY(ac_rsvd, module) {
    module.def(
        "run_ac_rsvd(Tensor matrix, float tolerance, float failure_probability, "
        "int block_size, int seed=0, int stream=0, "
        "bool sequential_orthogonalization=False, "
        "bool enhanced_mode=True, int diagnostic_test_size=512) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_ac_rsvd_fro(Tensor matrix, float tolerance, "
        "float failure_probability, float frobenius_norm_squared, "
        "int block_size, int seed=0, int stream=0, "
        "bool sequential_orthogonalization=False, "
        "bool enhanced_mode=True, int diagnostic_test_size=512) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_randqb_mf_fro(Tensor matrix, float tolerance, int block_size, "
        "int seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_randqb_ei(Tensor matrix, float tolerance, float frobenius_norm, "
        "int block_size, int seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_ac_rsvd_hadamard(Tensor spectrum, float tolerance, "
        "float failure_probability, int block_size, int matrix_seed, "
        "int algorithm_seed=0, int stream=0, "
        "bool sequential_orthogonalization=False, "
        "bool enhanced_mode=True, int diagnostic_test_size=512) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_ac_rsvd_fro_hadamard(Tensor spectrum, float tolerance, "
        "float failure_probability, int block_size, int matrix_seed, "
        "int algorithm_seed=0, int stream=0, "
        "bool sequential_orthogonalization=False, "
        "int diagnostic_test_size=512) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_randqb_mf_fro_hadamard(Tensor spectrum, float tolerance, "
        "int block_size, int matrix_seed, int algorithm_seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_randqb_ei_hadamard(Tensor spectrum, float tolerance, "
        "int block_size, int matrix_seed, int algorithm_seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "replay_certificate(Tensor trace, int input_dimension, float tolerance, "
        "float failure_probability, float candidate, "
        "int max_observations) -> float");
    module.def(
        "invert_certificate(Tensor trace, int input_dimension, float tolerance, "
        "float failure_probability, int max_observations) -> float");
    module.def(
        "replay_raw_gaussian_certificate(int count, float sum, int dimension, "
        "float spectral_cap, float gamma, float candidate) -> float");
    module.def(
        "invert_raw_gaussian_certificate(int count, float sum, int dimension, "
        "float spectral_cap, float gamma, float failure_probability, "
        "float upper) -> float");
    module.def(
        "theory_bounds_e5_analytic(Tensor spectrum, int rows, int cols, "
        "float tolerance, float failure_probability, "
        "float range_failure_probability, "
        "float certificate_failure_probability, float rho, float alpha, "
        "int block_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ac_rsvd, CPU, module) {
    module.impl("run_ac_rsvd", &ac_rsvd::run_ac_rsvd);
    module.impl("run_ac_rsvd_fro", &ac_rsvd::run_ac_rsvd_fro);
    module.impl("run_randqb_mf_fro", &ac_rsvd::run_randqb_mf_fro);
    module.impl("run_randqb_ei", &ac_rsvd::run_randqb_ei);
    module.impl("run_ac_rsvd_hadamard", &ac_rsvd::run_ac_rsvd_hadamard);
    module.impl(
        "run_ac_rsvd_fro_hadamard",
        &ac_rsvd::run_ac_rsvd_fro_hadamard);
    module.impl(
        "run_randqb_mf_fro_hadamard",
        &ac_rsvd::run_randqb_mf_fro_hadamard);
    module.impl("run_randqb_ei_hadamard", &ac_rsvd::run_randqb_ei_hadamard);
    module.impl("replay_certificate", &ac_rsvd::replay_certificate);
    module.impl("invert_certificate", &ac_rsvd::invert_certificate);
    module.impl(
        "theory_bounds_e5_analytic",
        &ac_rsvd::theory_bounds_e5_analytic);
}

TORCH_LIBRARY_IMPL(ac_rsvd, CatchAll, module) {
    module.impl(
        "replay_raw_gaussian_certificate",
        &ac_rsvd::replay_raw_gaussian_certificate);
    module.impl(
        "invert_raw_gaussian_certificate",
        &ac_rsvd::invert_raw_gaussian_certificate);
}
