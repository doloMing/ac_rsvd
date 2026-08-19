#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "ac_rsvd/ac_rsvd.hpp"
#include "ac_rsvd/randqb_ei.hpp"
#include "ac_rsvd/randqb_mf_fro.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/operators/dense_operator.hpp"

namespace ac_rsvd {
namespace {

using TorchResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::int64_t>;

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

    at::Tensor counters = at::empty({5}, at::kLong);
    std::int64_t* counter_data = counters.data_ptr<std::int64_t>();
    counter_data[0] = result.statistics.directions_processed;
    counter_data[1] = result.statistics.a_columns;
    counter_data[2] = result.statistics.at_columns;
    counter_data[3] = result.statistics.a_block_calls;
    counter_data[4] = result.statistics.at_block_calls;

    at::Tensor diagnostics = at::empty({3}, at::kDouble);
    double* diagnostic_data = diagnostics.data_ptr<double>();
    diagnostic_data[0] = result.residual_estimate_squared;
    diagnostic_data[1] = result.residual_bound_squared;
    diagnostic_data[2] = result.truncation_budget_squared;

    return std::make_tuple(
        u,
        singular_values,
        v,
        counters,
        diagnostics,
        static_cast<std::int64_t>(result.stop_reason));
}

TorchResult run_ac_rsvd(
    const at::Tensor& input,
    double tolerance,
    double failure_probability,
    std::int64_t block_size,
    std::int64_t seed,
    std::int64_t stream) {
    math::Matrix matrix = copy_input(input);
    math::DenseOperator op(matrix);

    AcRsvdOptions options;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.block_size = copy_block_size(block_size);
    options.seed = static_cast<std::uint64_t>(seed);
    options.stream = static_cast<std::uint64_t>(stream);
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

}  // namespace
}  // namespace ac_rsvd

TORCH_LIBRARY(ac_rsvd, module) {
    module.def(
        "run_ac_rsvd(Tensor matrix, float tolerance, float failure_probability, "
        "int block_size, int seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_randqb_mf_fro(Tensor matrix, float tolerance, int block_size, "
        "int seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "run_randqb_ei(Tensor matrix, float tolerance, float frobenius_norm, "
        "int block_size, int seed=0, int stream=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, int)");
}

TORCH_LIBRARY_IMPL(ac_rsvd, CPU, module) {
    module.impl("run_ac_rsvd", &ac_rsvd::run_ac_rsvd);
    module.impl("run_randqb_mf_fro", &ac_rsvd::run_randqb_mf_fro);
    module.impl("run_randqb_ei", &ac_rsvd::run_randqb_ei);
}
