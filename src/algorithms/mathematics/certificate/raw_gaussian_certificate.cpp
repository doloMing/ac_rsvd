#include "algorithms/mathematics/certificate/raw_gaussian_certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ac_rsvd::math {
namespace {

long double chi_square_32_cdf_long_double(long double value) {
    if (value <= 0.0L) {
        return 0.0L;
    }
    if (!std::isfinite(value)) {
        return 1.0L;
    }

    long double z = 0.5L * value;
    if (z <= 16.0L) {
        // This lower-tail series stays accurate when the probability is tiny.
        long double term = 1.0L;
        long double sum = 1.0L;
        for (int index = 1; index < 10000; ++index) {
            term *= z / (16.0L + index);
            sum += term;
            if (term <=
                std::numeric_limits<long double>::epsilon() * sum) {
                break;
            }
        }

        long double log_front =
            16.0L * std::log(z) - z - std::lgamma(17.0L);
        return std::exp(log_front) * sum;
    }

    if (z >= 100.0L) {
        return 1.0L;
    }

    long double term = 1.0L;
    long double upper_sum = 1.0L;
    for (int index = 1; index < 16; ++index) {
        term *= z / index;
        upper_sum += term;
    }
    return 1.0L - std::exp(-z) * upper_sum;
}

void check_raw_inputs(
    int dimension,
    double spectral_cap,
    double candidate,
    double gamma) {
    if (dimension < 1) {
        throw std::invalid_argument("Raw Gaussian dimension must be positive");
    }
    if (!std::isfinite(spectral_cap) || spectral_cap <= 0.0) {
        throw std::invalid_argument("Raw Gaussian spectral cap must be positive");
    }
    if (!std::isfinite(candidate) || candidate <= 0.0) {
        throw std::invalid_argument("Raw Gaussian candidate must be positive");
    }
    long double exact_largest_candidate =
        static_cast<long double>(dimension) * spectral_cap;
    double largest_candidate =
        static_cast<double>(exact_largest_candidate);
    if (static_cast<long double>(largest_candidate) <
        exact_largest_candidate) {
        largest_candidate = std::nextafter(
            largest_candidate,
            std::numeric_limits<double>::infinity());
    }
    if (candidate > largest_candidate) {
        throw std::invalid_argument("Raw Gaussian candidate exceeds the cap domain");
    }
    if (!std::isfinite(gamma) || gamma < 0.0) {
        throw std::invalid_argument("Raw Gaussian gamma cannot be negative");
    }
}

void capped_parts(
    int dimension,
    long double spectral_cap,
    long double candidate,
    int& full_count,
    long double& remainder) {
    long double ratio = candidate / spectral_cap;
    if (ratio <= 0.0L) {
        full_count = 0;
    } else if (ratio >= dimension) {
        full_count = dimension;
    } else {
        full_count = static_cast<int>(std::floor(ratio));
    }

    remainder = candidate - full_count * spectral_cap;
    if (remainder < 0.0L && full_count > 0) {
        --full_count;
        remainder += spectral_cap;
    }
    if (remainder >= spectral_cap && full_count < dimension) {
        ++full_count;
        remainder -= spectral_cap;
    }
    if (full_count == dimension) {
        remainder = 0.0L;
    }
}

long double raw_gaussian_log_psi_long_double(
    int dimension,
    long double spectral_cap,
    long double candidate,
    long double gamma) {
    if (gamma == 0.0L) {
        return 0.0L;
    }

    int full_count = 0;
    long double remainder = 0.0L;
    capped_parts(
        dimension,
        spectral_cap,
        candidate,
        full_count,
        remainder);

    long double result = -0.5L * full_count * std::log1p(
        2.0L * gamma * spectral_cap / candidate);
    if (remainder > 0.0L) {
        result -= 0.5L * std::log1p(
            2.0L * gamma * remainder / candidate);
    }
    return result;
}

long double gamma_derivative(
    int dimension,
    long double pilot_mean,
    long double spectral_cap,
    long double candidate,
    long double gamma) {
    int full_count = 0;
    long double remainder = 0.0L;
    capped_parts(
        dimension,
        spectral_cap,
        candidate,
        full_count,
        remainder);

    long double result = -pilot_mean / candidate;
    long double cap_ratio = spectral_cap / candidate;
    result += full_count * cap_ratio /
        (1.0L + 2.0L * gamma * cap_ratio);
    if (remainder > 0.0L) {
        long double remainder_ratio = remainder / candidate;
        result += remainder_ratio /
            (1.0L + 2.0L * gamma * remainder_ratio);
    }
    return result;
}

void check_count_and_sum(int count, double sum) {
    if (count < 0) {
        throw std::invalid_argument("Raw Gaussian count cannot be negative");
    }
    if (!std::isfinite(sum) || sum < 0.0) {
        throw std::invalid_argument("Raw Gaussian sum cannot be negative");
    }
}

void check_failure_probability(double failure_probability) {
    if (!std::isfinite(failure_probability) ||
        failure_probability <= 0.0 || failure_probability >= 1.0) {
        throw std::invalid_argument(
            "Failure probability must be between zero and one");
    }
}

}  // namespace

double chi_square_32_cdf(double value) {
    if (std::isnan(value)) {
        throw std::invalid_argument("Chi-square value cannot be NaN");
    }
    long double result = chi_square_32_cdf_long_double(value);
    if (result <= 0.0L) {
        return 0.0;
    }
    if (result >= 1.0L) {
        return 1.0;
    }
    return static_cast<double>(result);
}

double chi_square_32_lower_quantile(double probability) {
    if (!std::isfinite(probability) ||
        probability <= 0.0 || probability > 0.25) {
        throw std::invalid_argument(
            "Chi-square lower probability must be between zero and one quarter");
    }

    long double target = probability;
    long double low = 0.0L;
    long double high = 32.0L;
    for (int iteration = 0; iteration < 128; ++iteration) {
        long double middle = low + 0.5L * (high - low);
        if (chi_square_32_cdf_long_double(middle) <= target) {
            low = middle;
        } else {
            high = middle;
        }
    }

    double result = static_cast<double>(low);
    if (static_cast<long double>(result) > low) {
        result = std::nextafter(result, 0.0);
    }
    while (result > 0.0 &&
           chi_square_32_cdf_long_double(result) > target) {
        result = std::nextafter(result, 0.0);
    }
    return result;
}

double raw_gaussian_log_psi(
    int dimension,
    double spectral_cap,
    double candidate,
    double gamma) {
    check_raw_inputs(dimension, spectral_cap, candidate, gamma);
    return static_cast<double>(raw_gaussian_log_psi_long_double(
        dimension,
        spectral_cap,
        candidate,
        gamma));
}

double choose_raw_gaussian_gamma(
    int dimension,
    double pilot_mean,
    double spectral_cap,
    double candidate) {
    check_raw_inputs(dimension, spectral_cap, candidate, 0.0);
    if (!std::isfinite(pilot_mean) ||
        pilot_mean <= 0.0 || pilot_mean >= candidate) {
        throw std::invalid_argument(
            "Pilot mean must lie between zero and the candidate");
    }

    long double mean = pilot_mean;
    long double cap = spectral_cap;
    long double value = candidate;
    long double low = 0.0L;
    long double high = 1.0L;
    long double largest = std::numeric_limits<double>::max();
    while (gamma_derivative(
               dimension, mean, cap, value, high) > 0.0L) {
        if (high >= largest) {
            return std::numeric_limits<double>::infinity();
        }
        high = std::min(2.0L * high, largest);
    }

    for (int iteration = 0; iteration < 128; ++iteration) {
        long double middle = low + 0.5L * (high - low);
        if (middle == low || middle == high) {
            break;
        }
        if (gamma_derivative(
                dimension, mean, cap, value, middle) > 0.0L) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return static_cast<double>(low + 0.5L * (high - low));
}

double raw_gaussian_log_value(
    int count,
    double sum,
    int dimension,
    double spectral_cap,
    double gamma,
    double candidate) {
    check_count_and_sum(count, sum);
    if (candidate == 0.0) {
        check_raw_inputs(dimension, spectral_cap, spectral_cap, gamma);
        if (count == 0 || gamma == 0.0) {
            return 0.0;
        }
        if (sum > 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        long double gamma_value = gamma;
        long double doubled_gamma = 2.0L * gamma_value;
        long double log_limit = std::isfinite(doubled_gamma)
            ? std::log1p(doubled_gamma)
            : std::log(gamma_value) + std::log(2.0L);
        long double result = 0.5L * count * log_limit;
        return static_cast<double>(result);
    }

    check_raw_inputs(dimension, spectral_cap, candidate, gamma);
    if (count == 0 || gamma == 0.0) {
        return 0.0;
    }
    long double log_psi = raw_gaussian_log_psi_long_double(
        dimension,
        spectral_cap,
        candidate,
        gamma);
    long double result =
        -static_cast<long double>(gamma) * sum / candidate
        - static_cast<long double>(count) * log_psi;
    return static_cast<double>(result);
}

bool raw_gaussian_crosses(
    int count,
    double sum,
    int dimension,
    double spectral_cap,
    double gamma,
    double candidate,
    double failure_probability) {
    check_failure_probability(failure_probability);
    return raw_gaussian_log_value(
               count,
               sum,
               dimension,
               spectral_cap,
               gamma,
               candidate)
        >= -std::log(failure_probability);
}

double raw_gaussian_continuous_inverse(
    int count,
    double sum,
    int dimension,
    double spectral_cap,
    double gamma,
    double failure_probability,
    double upper) {
    check_failure_probability(failure_probability);
    if (!raw_gaussian_crosses(
            count,
            sum,
            dimension,
            spectral_cap,
            gamma,
            upper,
            failure_probability)) {
        throw std::invalid_argument(
            "Raw Gaussian inverse needs a crossing upper endpoint");
    }
    if (raw_gaussian_crosses(
            count,
            sum,
            dimension,
            spectral_cap,
            gamma,
            0.0,
            failure_probability)) {
        return 0.0;
    }

    double low = 0.0;
    double high = upper;
    for (int iteration = 0; iteration < 100; ++iteration) {
        double middle = low + 0.5 * (high - low);
        if (middle == low || middle == high) {
            break;
        }
        if (raw_gaussian_crosses(
                count,
                sum,
                dimension,
                spectral_cap,
                gamma,
                middle,
                failure_probability)) {
            high = middle;
        } else {
            low = middle;
        }
    }
    return high;
}

}  // namespace ac_rsvd::math
