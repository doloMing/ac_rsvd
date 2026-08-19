#pragma once

#include <vector>

namespace ac_rsvd::math {

struct CertificateTraceEntry {
    double x = 0.0;
    int unused_dimension = 0;
};

class Certificate {
public:
    Certificate(
        int input_dimension,
        double tolerance_squared,
        double failure_probability);

    bool update(double x, int unused_dimension);

    double value() const;
    double log_value() const;
    bool crossed() const;

    double replay(double candidate) const;
    double log_replay(double candidate) const;
    bool crosses(double candidate) const;

    double continuous_inverse() const;

    const std::vector<CertificateTraceEntry>& trace() const;

private:
    double start_weight(int round) const;
    double replay_log_value(double candidate) const;

    int input_dimension_ = 0;
    double tolerance_squared_ = 0.0;
    double log_threshold_ = 0.0;
    double harmonic_number_ = 0.0;
    double reserve_ = 1.0;
    double current_log_value_ = 0.0;

    std::vector<double> scales_;
    std::vector<double> log_components_;
    std::vector<CertificateTraceEntry> trace_;
    std::vector<std::vector<double>> log_normalizers_;
};

}  // namespace ac_rsvd::math
