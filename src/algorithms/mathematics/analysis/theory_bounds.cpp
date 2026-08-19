#include "algorithms/mathematics/analysis/theory_bounds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac_rsvd {
namespace math {
namespace {

struct BoundData {
    const std::vector<double>& singular_values;
    const std::vector<double>& tail_squared;
    int input_dimension;
    double tolerance;
    double range_failure_probability;
    double rho;
    double harmonic_number;
    int scale_count;
};

void check_probability(double value, const char* name) {
    if (value <= 0.0 || value >= 1.0) {
        throw std::invalid_argument(std::string(name) + " must be between zero and one");
    }
}

double harmonic_number(int last_index) {
    double result = 0.0;
    for (int index = 1; index <= last_index; ++index) {
        result += 1.0 / index;
    }
    return result;
}

int scale_count(int input_dimension) {
    return 1 + static_cast<int>(
        std::ceil(2.0 * std::log2(input_dimension)));
}

int e5_delay(
    int input_dimension,
    int range_directions,
    double failure_probability,
    double rho,
    double certificate_failure_probability,
    double harmonic,
    int scales) {
    int remaining = input_dimension - range_directions;
    if (remaining <= 1) {
        return remaining;
    }

    double threshold = std::log(
        (range_directions + 1.0) * harmonic * scales
        / failure_probability);
    double gap = 1.0 - rho;
    double raw_bound =
        (24.0 * threshold
         + 32.0 * std::log(1.0 / certificate_failure_probability))
        / (gap * gap);
    if (!std::isfinite(raw_bound) || raw_bound >= remaining) {
        return remaining;
    }
    return static_cast<int>(std::ceil(raw_bound));
}

double full_range_envelope_squared(
    const BoundData& data,
    int k,
    int p) {
    double t = std::exp(
        std::log(10.0 / data.range_failure_probability) / p);
    double tail = std::sqrt(data.tail_squared[k]);
    double next_singular_value = 0.0;
    if (k < static_cast<int>(data.singular_values.size())) {
        next_singular_value = data.singular_values[k];
    }

    double f = t * std::sqrt(12.0 * k / p);
    double g = t * std::exp(1.0) * std::sqrt(k + p) / (p + 1.0);
    double deviation =
        f * tail
        + g * next_singular_value
          * std::sqrt(2.0 * std::log(2.0 / data.range_failure_probability));
    return data.tail_squared[k] + deviation * deviation;
}

bool full_pair_is_feasible(const BoundData& data, int k, int p) {
    return full_range_envelope_squared(data, k, p)
           <= data.rho * data.tolerance * data.tolerance;
}

double corollary_factor_squared(
    int k,
    int p,
    double range_failure_probability) {
    double t_squared = std::exp(
        2.0 * std::log(10.0 / range_failure_probability) / p);
    double first = std::sqrt(12.0 * k / p);
    double second =
        std::exp(1.0) * std::sqrt(k + p) / (p + 1.0)
        * std::sqrt(2.0 * std::log(2.0 / range_failure_probability));
    return 1.0 + t_squared * (first + second) * (first + second);
}

int first_full_p(const BoundData& data, int k) {
    int maximum_p = data.input_dimension - k;
    if (maximum_p < 4 || !full_pair_is_feasible(data, k, maximum_p)) {
        return -1;
    }

    // For fixed k, both coefficients in H decrease as p grows.
    int low = 4;
    int high = maximum_p;
    while (low < high) {
        int middle = low + (high - low) / 2;
        if (full_pair_is_feasible(data, k, middle)) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

int first_corollary_p(
    int input_dimension,
    int k,
    double alpha,
    double rho,
    double range_failure_probability) {
    int maximum_p = input_dimension - k;
    if (maximum_p < 4 ||
        alpha * alpha * corollary_factor_squared(
            k, maximum_p, range_failure_probability) > rho) {
        return -1;
    }

    // The scalar Gaussian factor also decreases as p grows.
    int low = 4;
    int high = maximum_p;
    while (low < high) {
        int middle = low + (high - low) / 2;
        double factor_squared = corollary_factor_squared(
            k, middle, range_failure_probability);
        if (alpha * alpha * factor_squared <= rho) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

AnalyticTheoryBound cap_bound(
    int cap,
    int input_dimension,
    int block_size) {
    AnalyticTheoryBound result;
    result.directions = cap;
    result.forward_columns =
        block_rounded_columns(input_dimension, block_size, cap);
    result.forward_block_calls = block_call_bound(block_size, cap);
    return result;
}

int tolerance_rank(
    const std::vector<double>& tail_squared,
    int matrix_rank,
    double tolerance) {
    double tolerance_squared = tolerance * tolerance;
    for (int rank = 0; rank <= matrix_rank; ++rank) {
        if (tail_squared[rank] <= tolerance_squared) {
            return rank;
        }
    }
    return matrix_rank;
}

void fill_candidate(
    AnalyticTheoryBound& result,
    int directions,
    int k,
    int p,
    int delay,
    int input_dimension,
    int block_size) {
    result.directions = directions;
    result.forward_columns =
        block_rounded_columns(input_dimension, block_size, directions);
    result.forward_block_calls = block_call_bound(block_size, directions);
    result.k = k;
    result.p = p;
    result.range_directions = k + p;
    result.appendix_e5_delay = delay;
    result.uses_deterministic_cap = false;
}

}  // namespace

int deterministic_direction_cap(
    int output_dimension,
    int input_dimension,
    int matrix_rank) {
    if (output_dimension <= 0 || input_dimension <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    int maximum_rank = std::min(output_dimension, input_dimension);
    if (matrix_rank < 0 || matrix_rank > maximum_rank) {
        throw std::invalid_argument("Matrix rank does not fit its dimensions");
    }
    return std::min({input_dimension, output_dimension, matrix_rank + 1});
}

int appendix_e5_certificate_delay(
    int input_dimension,
    int range_directions,
    double failure_probability,
    double rho,
    double certificate_failure_probability) {
    if (input_dimension <= 0 ||
        range_directions < 0 ||
        range_directions > input_dimension) {
        throw std::invalid_argument("Range directions must fit the input dimension");
    }
    check_probability(failure_probability, "Failure probability");
    check_probability(rho, "rho");
    check_probability(
        certificate_failure_probability,
        "Certificate failure probability");

    int remaining = input_dimension - range_directions;
    if (remaining <= 1) {
        return remaining;
    }
    return e5_delay(
        input_dimension,
        range_directions,
        failure_probability,
        rho,
        certificate_failure_probability,
        harmonic_number(input_dimension - 1),
        scale_count(input_dimension));
}

int block_rounded_columns(
    int input_dimension,
    int block_size,
    int directions) {
    if (input_dimension <= 0 || block_size <= 0 ||
        directions < 0 || directions > input_dimension) {
        throw std::invalid_argument("Block rounding received invalid counts");
    }
    long long calls =
        (static_cast<long long>(directions) + block_size - 1) / block_size;
    long long columns = calls * block_size;
    return static_cast<int>(std::min<long long>(input_dimension, columns));
}

int block_call_bound(int block_size, int directions) {
    if (block_size <= 0 || directions < 0) {
        throw std::invalid_argument("Block call count received invalid counts");
    }
    return static_cast<int>(
        (static_cast<long long>(directions) + block_size - 1) / block_size);
}

AnalyticTheoryBounds evaluate_theory_bounds_e5(
    const std::vector<double>& singular_values,
    int output_dimension,
    int input_dimension,
    double tolerance,
    double failure_probability,
    double range_failure_probability,
    double certificate_failure_probability,
    double rho,
    double alpha,
    int block_size) {
    int singular_count = std::min(output_dimension, input_dimension);
    if (output_dimension <= 0 || input_dimension <= 0 ||
        static_cast<int>(singular_values.size()) != singular_count) {
        throw std::invalid_argument("Spectrum must match the matrix dimensions");
    }
    if (tolerance <= 0.0 || block_size <= 0) {
        throw std::invalid_argument("Tolerance and block size must be positive");
    }
    check_probability(failure_probability, "Failure probability");
    check_probability(range_failure_probability, "Range failure probability");
    check_probability(
        certificate_failure_probability,
        "Certificate failure probability");
    check_probability(rho, "rho");
    check_probability(alpha, "alpha");
    if (alpha * alpha >= rho) {
        throw std::invalid_argument("alpha squared must be below rho");
    }

    int matrix_rank = 0;
    double previous = std::numeric_limits<double>::infinity();
    for (double value : singular_values) {
        if (!std::isfinite(value) || value < 0.0 || value > previous) {
            throw std::invalid_argument(
                "Singular values must be finite, nonnegative, and nonincreasing");
        }
        if (value > 0.0) {
            ++matrix_rank;
        }
        previous = value;
    }

    std::vector<double> tail_squared(singular_count + 1, 0.0);
    for (int index = singular_count - 1; index >= 0; --index) {
        tail_squared[index] =
            tail_squared[index + 1]
            + singular_values[index] * singular_values[index];
    }

    int cap = deterministic_direction_cap(
        output_dimension, input_dimension, matrix_rank);
    AnalyticTheoryBounds bounds;
    bounds.deterministic_cap = cap;
    bounds.theorem2_e5 = cap_bound(cap, input_dimension, block_size);
    bounds.corollary21_e5 = cap_bound(cap, input_dimension, block_size);

    BoundData data{
        singular_values,
        tail_squared,
        input_dimension,
        tolerance,
        range_failure_probability,
        rho,
        input_dimension > 1 ? harmonic_number(input_dimension - 1) : 0.0,
        input_dimension > 1 ? scale_count(input_dimension) : 1,
    };

    int maximum_k = std::min(singular_count, input_dimension - 4);
    for (int k = 2; k <= maximum_k; ++k) {
        if (k + 4 >= bounds.theorem2_e5.directions) {
            break;
        }

        int p = first_full_p(data, k);
        if (p < 0) {
            continue;
        }

        // q0 + q_simple is min(n, q0 plus an increasing log term).
        // The first feasible p therefore gives this k's best score.
        int range_directions = k + p;
        int delay = e5_delay(
            input_dimension,
            range_directions,
            failure_probability,
            rho,
            certificate_failure_probability,
            data.harmonic_number,
            data.scale_count);
        int candidate = range_directions + delay;
        if (candidate < bounds.theorem2_e5.directions) {
            fill_candidate(
                bounds.theorem2_e5,
                candidate,
                k,
                p,
                delay,
                input_dimension,
                block_size);
        }
    }

    int rank_alpha = tolerance_rank(
        tail_squared, matrix_rank, alpha * tolerance);
    int k_alpha = std::max(2, rank_alpha);
    bounds.corollary21_e5.k = k_alpha;
    bounds.corollary21_e5.tolerance_rank = rank_alpha;

    if (k_alpha <= singular_count && k_alpha + 4 <= input_dimension) {
        int p = first_corollary_p(
            input_dimension,
            k_alpha,
            alpha,
            rho,
            range_failure_probability);
        if (p >= 0) {
            int range_directions = k_alpha + p;
            int delay = e5_delay(
                input_dimension,
                range_directions,
                failure_probability,
                rho,
                certificate_failure_probability,
                data.harmonic_number,
                data.scale_count);
            int candidate = range_directions + delay;
            if (candidate < bounds.corollary21_e5.directions) {
                fill_candidate(
                    bounds.corollary21_e5,
                    candidate,
                    k_alpha,
                    p,
                    delay,
                    input_dimension,
                    block_size);
                bounds.corollary21_e5.tolerance_rank = rank_alpha;
            }
        }
    }

    return bounds;
}

}  // namespace math
}  // namespace ac_rsvd
