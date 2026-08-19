#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
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

    at::Tensor counters = at::empty({7}, at::kLong);
    std::int64_t* counter_data = counters.data_ptr<std::int64_t>();
    counter_data[0] = result.statistics.directions_processed;
    counter_data[1] = result.statistics.a_columns;
    counter_data[2] = result.statistics.at_columns;
    counter_data[3] = result.statistics.a_block_calls;
    counter_data[4] = result.statistics.at_block_calls;
    counter_data[5] = result.range_rank;
    counter_data[6] = result.boundary_only_rank;

    at::Tensor diagnostics = at::empty({5}, at::kDouble);
    double* diagnostic_data = diagnostics.data_ptr<double>();
    diagnostic_data[0] = result.residual_estimate_squared;
    diagnostic_data[1] = result.residual_bound_squared;
    diagnostic_data[2] = result.truncation_budget_squared;
    diagnostic_data[3] = result.residual_decrease_squared;
    diagnostic_data[4] = result.direct_error_squared;

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
    at::Tensor trace = at::empty({trace_size, 2}, at::kDouble);
    auto trace_values = trace.accessor<double, 2>();
    for (std::int64_t index = 0; index < trace_size; ++index) {
        trace_values[index][0] = result.certificate_observations[index];
        trace_values[index][1] = result.certificate_dimensions[index];
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
    double failure_probability) {
    TORCH_CHECK(trace.device().is_cpu(), "trace must be on the CPU");
    TORCH_CHECK(trace.scalar_type() == at::kDouble, "trace must use float64");
    TORCH_CHECK(
        trace.dim() == 2 && trace.size(1) == 2,
        "trace must have shape (rounds, 2)");
    TORCH_CHECK(
        input_dimension >= 2 &&
        input_dimension <= std::numeric_limits<int>::max(),
        "input_dimension must fit an LP64 integer above one");

    math::Certificate certificate(
        static_cast<int>(input_dimension),
        tolerance * tolerance,
        failure_probability);
    auto values = trace.accessor<double, 2>();
    for (std::int64_t row = 0; row < trace.size(0); ++row) {
        certificate.update(
            values[row][0],
            static_cast<int>(values[row][1]));
    }
    return certificate;
}

double replay_certificate(
    const at::Tensor& trace,
    std::int64_t input_dimension,
    double tolerance,
    double failure_probability,
    double candidate) {
    math::Certificate certificate = certificate_from_trace(
        trace,
        input_dimension,
        tolerance,
        failure_probability);
    return certificate.replay(candidate);
}

double invert_certificate(
    const at::Tensor& trace,
    std::int64_t input_dimension,
    double tolerance,
    double failure_probability) {
    math::Certificate certificate = certificate_from_trace(
        trace,
        input_dimension,
        tolerance,
        failure_probability);
    return certificate.continuous_inverse();
}

TorchResult run_ac_rsvd(
    const at::Tensor& input,
    double tolerance,
    double failure_probability,
    std::int64_t block_size,
    std::int64_t seed,
    std::int64_t stream,
    bool sequential_orthogonalization) {
    math::Matrix matrix = copy_input(input);
    math::DenseOperator op(matrix);

    AcRsvdOptions options;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(seed);
    options.stream = static_cast<std::uint64_t>(stream);
    options.use_sequential_orthogonalization = sequential_orthogonalization;
    return copy_result(compute_ac_rsvd(op, options));
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
    bool sequential_orthogonalization) {
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

    FactorizationResult result = compute_ac_rsvd(op, options);
    result.direct_error_squared = structured_error_squared(op, result);
    return copy_result(result);
}

TorchResult run_randqb_mf_fro_hadamard(
    const at::Tensor& spectrum,
    double tolerance,
    std::int64_t block_size,
    std::int64_t matrix_seed,
    std::int64_t algorithm_seed,
    std::int64_t stream) {
    math::StructuredHadamardOperator op(
        copy_spectrum(spectrum),
        static_cast<std::uint64_t>(matrix_seed));

    RandQbMfFroOptions options;
    options.tolerance = tolerance;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(algorithm_seed);
    options.stream = static_cast<std::uint64_t>(stream);

    FactorizationResult result = compute_randqb_mf_fro(op, options);
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
    math::StructuredHadamardOperator op(
        copy_spectrum(spectrum),
        static_cast<std::uint64_t>(matrix_seed));

    RandQbEiOptions options;
    options.tolerance = tolerance;
    options.frobenius_norm = op.frobenius_norm();
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(algorithm_seed);
    options.stream = static_cast<std::uint64_t>(stream);

    FactorizationResult result = compute_randqb_ei(op, options);
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
        "bool sequential_orthogonalization=False) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
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
        "bool sequential_orthogonalization=False) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
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
        "float failure_probability, float candidate) -> float");
    module.def(
        "invert_certificate(Tensor trace, int input_dimension, float tolerance, "
        "float failure_probability) -> float");
    module.def(
        "theory_bounds_e5_analytic(Tensor spectrum, int rows, int cols, "
        "float tolerance, float failure_probability, "
        "float range_failure_probability, "
        "float certificate_failure_probability, float rho, float alpha, "
        "int block_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ac_rsvd, CPU, module) {
    module.impl("run_ac_rsvd", &ac_rsvd::run_ac_rsvd);
    module.impl("run_randqb_mf_fro", &ac_rsvd::run_randqb_mf_fro);
    module.impl("run_randqb_ei", &ac_rsvd::run_randqb_ei);
    module.impl("run_ac_rsvd_hadamard", &ac_rsvd::run_ac_rsvd_hadamard);
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
