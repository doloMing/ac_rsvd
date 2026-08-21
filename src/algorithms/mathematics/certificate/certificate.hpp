#pragma once

#include <array>
#include <vector>

namespace ac_rsvd::math {

struct CertificateTraceEntry {
    double x = 0.0;
    int unused_dimension = 0;
    double leakage = 0.0;
    std::array<double, 4> predictable_scales = {};
};

// The certificate combines the restart mixture with four predictable accounts.
class Certificate {
public:
    Certificate(
        int input_dimension,
        double tolerance_squared,
        double failure_probability,
        int max_observations = 0);

    bool update(double x, int unused_dimension);
    bool update(
        double x,
        int unused_dimension,
        double leakage);

    // Replay keeps the bets chosen during the original run.
    bool update_from_trace(const CertificateTraceEntry& entry);

    double value() const;
    double log_value() const;
    bool crossed() const;

    double replay(double candidate) const;
    double log_replay(double candidate) const;
    bool crosses(double candidate) const;

    // Find the smallest squared residual that still crosses the threshold.
    double continuous_inverse() const;

    const std::vector<CertificateTraceEntry>& trace() const;
    int max_observations() const;

private:
    double start_weight(int round) const;
    std::array<double, 4> choose_predictable_scales(double leakage) const;
    double replay_log_value(double candidate) const;

    int max_observations_ = 0;
    double tolerance_squared_ = 0.0;
    double log_threshold_ = 0.0;
    double harmonic_number_ = 0.0;
    double reserve_ = 1.0;
    double current_log_value_ = 0.0;

    std::vector<double> scales_;
    std::vector<double> log_components_;
    std::array<double, 4> predictable_log_values_ = {};
    std::vector<CertificateTraceEntry> trace_;
};

}  // namespace ac_rsvd::math
