#include "algorithms/mathematics/certificate/spherical_moment.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ac_rsvd::math {

double log_phi(int dimension, double s) {
    if (dimension < 1) {
        throw std::invalid_argument("log_phi needs a positive dimension");
    }
    if (s < -1.0 / 3.0 || s > 0.0) {
        throw std::invalid_argument("log_phi only supports -1/3 <= s <= 0");
    }
    if (dimension == 1) {
        return s;
    }
    if (s == 0.0) {
        return 0.0;
    }

    // This is 1F1(1/2; d/2; s*d). The term ratio stays below 2*abs(s).
    long double term = 1.0L;
    long double sum = 1.0L;
    long double correction = 0.0L;
    long double z = static_cast<long double>(s) * dimension;
    long double half_dimension = 0.5L * dimension;

    for (int k = 0; k < 10000; ++k) {
        term *= z * (0.5L + k);
        term /= (half_dimension + k) * (1.0L + k);

        long double adjusted = term - correction;
        long double next_sum = sum + adjusted;
        correction = (next_sum - sum) - adjusted;
        sum = next_sum;

        long double scale = std::abs(sum);
        if (std::abs(term) <=
            4.0L * std::numeric_limits<long double>::epsilon() * scale) {
            break;
        }
    }

    return static_cast<double>(std::log(sum));
}

}  // namespace ac_rsvd::math
