#include "algorithms/mathematics/certificate/certificate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/certificate/spherical_moment.hpp"

namespace ac_rsvd::math {
namespace {

// Fixed empirical windows. The proof only needs each selected scale to depend
// on observations available before the current one.
constexpr std::array<int, 4> history_lengths = {8, 16, 32, 64};
// Fixed empirical weights: one default account and four predictable accounts
// have total weight one. The proof does not require this particular split.
constexpr double default_weight = 0.1;
constexpr double predictable_weight = 0.225;

double log_add(double first, double second) {
    if (first == -std::numeric_limits<double>::infinity()) {
        return second;
    }
    if (second == -std::numeric_limits<double>::infinity()) {
        return first;
    }

    double larger = std::max(first, second);
    double smaller = std::min(first, second);
    return larger + std::log1p(std::exp(smaller - larger));
}

double log_sum(double reserve, const std::vector<double>& components) {
    double result = -std::numeric_limits<double>::infinity();
    if (reserve > 0.0) {
        result = std::log(reserve);
    }
    for (double component : components) {
        result = log_add(result, component);
    }
    return result;
}

double hybrid_log_value(
    double default_log_value,
    const std::array<double, 4>& predictable_log_values) {
    double result = std::log(default_weight) + default_log_value;
    for (double value : predictable_log_values) {
        result = log_add(
            result,
            std::log(predictable_weight) + value);
    }
    return result;
}

double log_factor(
    const CertificateTraceEntry& entry,
    double gamma,
    double candidate) {
    if (gamma == 0.0) {
        return 0.0;
    }

    if (candidate == 0.0) {
        if (entry.x > 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        if (entry.leakage > 0.0) {
            return 0.0;
        }
        return -log_phi(entry.unused_dimension, -gamma);
    }

    double result = -gamma * entry.x / candidate;
    if (candidate <= entry.leakage) {
        return result;
    }

    double argument = -gamma * (1.0 - entry.leakage / candidate);
    return result - log_phi(entry.unused_dimension, argument);
}

}  // namespace

Certificate::Certificate(
    int input_dimension,
    double tolerance_squared,
    double failure_probability,
    int max_observations)
    : max_observations_(max_observations == 0
          ? input_dimension - 1
          : max_observations),
      tolerance_squared_(tolerance_squared) {
    if (input_dimension < 2) {
        throw std::invalid_argument("Certificate needs input dimension at least two");
    }
    if (!std::isfinite(tolerance_squared) || tolerance_squared <= 0.0) {
        throw std::invalid_argument("Certificate needs a positive squared tolerance");
    }
    if (max_observations_ < 1) {
        throw std::invalid_argument("Certificate needs at least one observation");
    }
    if (!std::isfinite(failure_probability) ||
        failure_probability <= 0.0 || failure_probability >= 1.0) {
        throw std::invalid_argument("Failure probability must be between zero and one");
    }

    for (long long value = 1; value <= max_observations_; ++value) {
        harmonic_number_ += 1.0 / value;
    }

    int scale_count =
        1 + static_cast<int>(std::ceil(2.0 * std::log2(input_dimension)));
    for (int index = 0; index < scale_count; ++index) {
        scales_.push_back(std::ldexp(1.0 / 3.0, -index));
    }

    log_components_.assign(
        scales_.size(),
        -std::numeric_limits<double>::infinity());
    log_threshold_ = -std::log(failure_probability);
}

bool Certificate::update(double x, int unused_dimension) {
    return update(x, unused_dimension, 0.0);
}

bool Certificate::update(
    double x,
    int unused_dimension,
    double leakage) {
    if (!std::isfinite(leakage) || leakage < 0.0) {
        throw std::invalid_argument("Certificate leakage cannot be negative");
    }

    // Freeze the bets before this observation enters the history.
    std::array<double, 4> predictable_scales =
        choose_predictable_scales(leakage);
    CertificateTraceEntry entry;
    entry.x = x;
    entry.leakage = leakage;
    entry.unused_dimension = unused_dimension;
    entry.predictable_scales = predictable_scales;
    return update_from_trace(entry);
}

bool Certificate::update_from_trace(const CertificateTraceEntry& entry) {
    if (!std::isfinite(entry.x) || entry.x < 0.0) {
        throw std::invalid_argument("Certificate observation cannot be negative");
    }
    if (!std::isfinite(entry.leakage) || entry.leakage < 0.0) {
        throw std::invalid_argument("Certificate leakage cannot be negative");
    }
    if (entry.unused_dimension <= 1) {
        throw std::invalid_argument("Certificate updates need unused dimension above one");
    }
    if (static_cast<int>(trace_.size()) >= max_observations_) {
        throw std::invalid_argument("Certificate received too many observations");
    }
    for (double gamma : entry.predictable_scales) {
        if (gamma < 0.0 || gamma > 1.0 / 3.0 || std::isnan(gamma)) {
            throw std::invalid_argument("Certificate scale must be between zero and one third");
        }
    }

    int round = static_cast<int>(trace_.size());
    double weight = start_weight(round);
    reserve_ = std::max(0.0, reserve_ - weight);
    double log_start = std::log(weight) - std::log(scales_.size());

    for (std::size_t index = 0; index < scales_.size(); ++index) {
        double factor = log_factor(
            entry,
            scales_[index],
            tolerance_squared_);
        log_components_[index] =
            factor + log_add(log_components_[index], log_start);
    }

    for (std::size_t index = 0;
         index < predictable_log_values_.size();
         ++index) {
        predictable_log_values_[index] += log_factor(
            entry,
            entry.predictable_scales[index],
            tolerance_squared_);
    }

    trace_.push_back(entry);
    current_log_value_ = hybrid_log_value(
        log_sum(reserve_, log_components_),
        predictable_log_values_);
    return crossed();
}

double Certificate::value() const {
    return std::exp(current_log_value_);
}

double Certificate::log_value() const {
    return current_log_value_;
}

bool Certificate::crossed() const {
    return current_log_value_ >= log_threshold_;
}

double Certificate::replay(double candidate) const {
    return std::exp(log_replay(candidate));
}

double Certificate::log_replay(double candidate) const {
    if (!std::isfinite(candidate) || candidate < 0.0) {
        throw std::invalid_argument("Certificate candidate cannot be negative");
    }
    return replay_log_value(candidate);
}

bool Certificate::crosses(double candidate) const {
    return log_replay(candidate) >= log_threshold_;
}

double Certificate::continuous_inverse() const {
    if (!crossed()) {
        return std::numeric_limits<double>::infinity();
    }
    if (crosses(0.0)) {
        return 0.0;
    }

    double low = 0.0;
    double high = tolerance_squared_;
    for (int iteration = 0; iteration < 100; ++iteration) {
        double middle = low + 0.5 * (high - low);
        if (middle == low || middle == high) {
            break;
        }
        if (crosses(middle)) {
            high = middle;
        } else {
            low = middle;
        }
    }
    return high;
}

const std::vector<CertificateTraceEntry>& Certificate::trace() const {
    return trace_;
}

int Certificate::max_observations() const {
    return max_observations_;
}

double Certificate::start_weight(int round) const {
    return 1.0 / ((round + 1.0) * harmonic_number_);
}

std::array<double, 4> Certificate::choose_predictable_scales(
    double leakage) const {
    std::array<double, 4> result = {};
    int round = static_cast<int>(trace_.size());
    double leakage_ratio = leakage / tolerance_squared_;
    double available = 1.0 - leakage_ratio;
    if (available <= 0.0) {
        return result;
    }

    for (std::size_t index = 0; index < history_lengths.size(); ++index) {
        int history = history_lengths[index];
        if (round < history) {
            continue;
        }

        double mean = 0.0;
        for (int offset = round - history; offset < round; ++offset) {
            mean +=
                (trace_[offset].x + trace_[offset].leakage)
                / tolerance_squared_;
        }
        // The two 0.05 floors and the 1/3 cap below are fixed empirical
        // defaults. The proof uses only predictability and the scale bound.
        mean = std::max(0.05, mean / history);

        double unused_estimate = std::max(
            mean - leakage_ratio,
            0.05 * available);
        double numerator = available - unused_estimate;
        if (numerator <= 0.0) {
            continue;
        }

        result[index] = std::min(
            1.0 / 3.0,
            numerator / (2.0 * available * unused_estimate));
    }
    return result;
}

double Certificate::replay_log_value(double candidate) const {
    double reserve = 1.0;
    std::vector<double> components(
        scales_.size(),
        -std::numeric_limits<double>::infinity());
    std::array<double, 4> predictable_values = {};

    for (std::size_t round = 0; round < trace_.size(); ++round) {
        double weight = start_weight(static_cast<int>(round));
        reserve = std::max(0.0, reserve - weight);
        double log_start = std::log(weight) - std::log(scales_.size());

        for (std::size_t index = 0; index < scales_.size(); ++index) {
            double factor = log_factor(
                trace_[round],
                scales_[index],
                candidate);
            components[index] =
                factor + log_add(components[index], log_start);
        }

        for (std::size_t index = 0;
             index < predictable_values.size();
             ++index) {
            predictable_values[index] += log_factor(
                trace_[round],
                trace_[round].predictable_scales[index],
                candidate);
        }
    }

    return hybrid_log_value(
        log_sum(reserve, components),
        predictable_values);
}

}  // namespace ac_rsvd::math
