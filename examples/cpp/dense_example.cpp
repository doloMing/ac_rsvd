#include <cmath>
#include <iostream>

#include "ac_rsvd/ac_rsvd.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/operators/dense_operator.hpp"

int main() {
    ac_rsvd::math::Matrix matrix(6, 5);
    for (int index = 0; index < 5; ++index) {
        matrix(index, index) = std::pow(0.2, index);
    }

    ac_rsvd::math::DenseOperator op(matrix);
    ac_rsvd::AcRsvdOptions options;
    options.tolerance = 1e-2;
    options.failure_probability = 1e-6;
    options.block_size = 3;
    options.seed = 7;

    ac_rsvd::FactorizationResult result =
        ac_rsvd::compute_ac_rsvd(op, options);

    std::cout << "rank: " << result.rank << '\n';
    std::cout << "A columns: " << result.statistics.a_columns << '\n';
    std::cout << "A^T block calls: "
              << result.statistics.at_block_calls << '\n';
    return 0;
}
