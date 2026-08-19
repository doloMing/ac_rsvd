#include "algorithms/mathematics/certificate/certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/certificate/spherical_moment.hpp"

namespace ac_rsvd::math {
namespace {

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

}  // namespace

Certificate::Certificate(
    int input_dimension,
    double tolerance_squared,
    double failure_probability)
    : input_dimension_(input_dimension),
      tolerance_squared_(tolerance_squared) {
    if (input_dimension < 2) {
        throw std::invalid_argument("Certificate needs input dimension at least two");
    }
    if (tolerance_squared <= 0.0) {
        throw std::invalid_argument("Certificate needs a positive squared tolerance");
    }
    if (failure_probability <= 0.0 || failure_probability >= 1.0) {
        throw std::invalid_argument("Failure probability must be between zero and one");
    }

    for (int value = 1; value <= input_dimension - 1; ++value) {
        harmonic_number_ += 1.0 / value;
    }

    int scale_count =
        1 + static_cast<int>(std::ceil(2.0 * std::log2(input_dimension)));
    // The dyadic grid mixes the admissible betting scales from 1/3 downward.
    for (int index = 0; index < scale_count; ++index) {
        scales_.push_back(std::ldexp(1.0 / 3.0, -index));
    }

    log_components_.assign(
        scales_.size(),
        -std::numeric_limits<double>::infinity());
    log_threshold_ = -std::log(failure_probability);
}

bool Certificate::update(double x, int unused_dimension) {
    if (x < 0.0) {
        throw std::invalid_argument("Certificate observation cannot be negative");
    }
    if (unused_dimension <= 1) {
        throw std::invalid_argument("Certificate updates need unused dimension above one");
    }
    if (static_cast<int>(trace_.size()) >= input_dimension_ - 1) {
        throw std::invalid_argument("Certificate received too many observations");
    }

    int round = static_cast<int>(trace_.size());
    double weight = start_weight(round);
    reserve_ = std::max(0.0, reserve_ - weight);

    std::vector<double> normalizers(scales_.size());
    double log_start = std::log(weight) - std::log(scales_.size());

    // Each component gains exp(-gamma*x/tau^2) / Phi_d(-gamma).
    for (std::size_t index = 0; index < scales_.size(); ++index) {
        double gamma = scales_[index];
        normalizers[index] = log_phi(unused_dimension, -gamma);
        double log_factor =
            -gamma * x / tolerance_squared_ - normalizers[index];
        log_components_[index] =
            log_factor + log_add(log_components_[index], log_start);
    }

    trace_.push_back({x, unused_dimension});
    log_normalizers_.push_back(normalizers);
    current_log_value_ = log_sum(reserve_, log_components_);
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
    if (candidate < 0.0) {
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
    // The replay value increases monotonically with the candidate residual.
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

double Certificate::start_weight(int round) const {
    return 1.0 / ((round + 1.0) * harmonic_number_);
}

double Certificate::replay_log_value(double candidate) const {
    double reserve = 1.0;
    std::vector<double> components(
        scales_.size(),
        -std::numeric_limits<double>::infinity());

    for (std::size_t round = 0; round < trace_.size(); ++round) {
        double weight = start_weight(static_cast<int>(round));
        reserve = std::max(0.0, reserve - weight);
        double log_start = std::log(weight) - std::log(scales_.size());

        for (std::size_t index = 0; index < scales_.size(); ++index) {
            double log_factor = -std::numeric_limits<double>::infinity();
            if (candidate > 0.0) {
                log_factor =
                    -scales_[index] * trace_[round].x / candidate
                    - log_normalizers_[round][index];
            } else if (trace_[round].x == 0.0) {
                log_factor = -log_normalizers_[round][index];
            }

            components[index] =
                log_factor + log_add(components[index], log_start);
        }
    }

    return log_sum(reserve, components);
}

}  // namespace ac_rsvd::math
