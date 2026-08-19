#include "algorithms/mathematics/operators/structured_hadamard_operator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/random/philox.hpp"

namespace ac_rsvd {
namespace math {
namespace {

std::uint64_t random_word(
    std::uint64_t seed,
    std::uint32_t stream,
    std::uint64_t counter) {
    std::array<std::uint32_t, 4> value = philox4x32(
        {static_cast<std::uint32_t>(counter),
         static_cast<std::uint32_t>(counter >> 32),
         stream,
         0},
        {static_cast<std::uint32_t>(seed),
         static_cast<std::uint32_t>(seed >> 32)});
    return (static_cast<std::uint64_t>(value[0]) << 32) | value[1];
}

std::vector<double> make_signs(
    int size,
    std::uint64_t seed,
    std::uint32_t stream) {
    std::vector<double> signs(size);
    for (int index = 0; index < size; ++index) {
        std::uint64_t word = random_word(seed, stream, index);
        signs[index] = (word & 1U) == 0 ? -1.0 : 1.0;
    }
    return signs;
}

int random_index(
    int upper,
    std::uint64_t seed,
    std::uint32_t stream,
    std::uint64_t& counter) {
    std::uint64_t range = static_cast<std::uint64_t>(upper);
    std::uint64_t threshold = (0U - range) % range;
    std::uint64_t word = 0;
    do {
        word = random_word(seed, stream, counter);
        ++counter;
    } while (word < threshold);
    return static_cast<int>(word % range);
}

std::vector<int> make_permutation(
    int size,
    std::uint64_t seed,
    std::uint32_t stream) {
    std::vector<int> permutation(size);
    for (int index = 0; index < size; ++index) {
        permutation[index] = index;
    }

    std::uint64_t counter = 0;
    for (int index = size - 1; index > 0; --index) {
        int other = random_index(index + 1, seed, stream, counter);
        std::swap(permutation[index], permutation[other]);
    }
    return permutation;
}

void normalized_fwht(double* values, int size) {
    for (int width = 1; width < size; width *= 2) {
        for (int first = 0; first < size; first += 2 * width) {
            for (int offset = 0; offset < width; ++offset) {
                int left = first + offset;
                int right = left + width;
                double sum = values[left] + values[right];
                double difference = values[left] - values[right];
                values[left] = sum;
                values[right] = difference;
            }
        }
    }

    double scale = 1.0 / std::sqrt(static_cast<double>(size));
    for (int index = 0; index < size; ++index) {
        values[index] *= scale;
    }
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

}  // namespace

StructuredHadamardOperator::StructuredHadamardOperator(
    const std::vector<double>& singular_values,
    std::uint64_t seed)
    : size_(static_cast<int>(singular_values.size())),
      singular_values_(singular_values),
      frobenius_norm_(0.0) {
    if (!is_power_of_two(size_)) {
        throw std::invalid_argument("Hadamard size must be a power of two");
    }

    for (double value : singular_values_) {
        if (value < 0.0) {
            throw std::invalid_argument("Singular values must be nonnegative");
        }
        frobenius_norm_ += value * value;
    }
    frobenius_norm_ = std::sqrt(frobenius_norm_);

    // Four Philox streams keep both factors reproducible and independent.
    left_signs_ = make_signs(size_, seed, 0);
    left_permutation_ = make_permutation(size_, seed, 1);
    right_signs_ = make_signs(size_, seed, 2);
    right_permutation_ = make_permutation(size_, seed, 3);
}

int StructuredHadamardOperator::rows() const {
    return size_;
}

int StructuredHadamardOperator::cols() const {
    return size_;
}

void StructuredHadamardOperator::apply(
    const double* x,
    int block_cols,
    double* y) const {
    // U = P_u H D_u and V = P_v H D_v.
    #pragma omp parallel
    {
        std::vector<double> work(size_);
        #pragma omp for schedule(static)
        for (int col = 0; col < block_cols; ++col) {
            for (int index = 0; index < size_; ++index) {
                work[index] = x[right_permutation_[index] + col * size_];
            }
            normalized_fwht(work.data(), size_);
            for (int index = 0; index < size_; ++index) {
                work[index] *= right_signs_[index];
                work[index] *= singular_values_[index];
                work[index] *= left_signs_[index];
            }
            normalized_fwht(work.data(), size_);
            for (int index = 0; index < size_; ++index) {
                y[left_permutation_[index] + col * size_] = work[index];
            }
        }
    }
}

void StructuredHadamardOperator::apply_transpose(
    const double* y,
    int block_cols,
    double* x) const {
    #pragma omp parallel
    {
        std::vector<double> work(size_);
        #pragma omp for schedule(static)
        for (int col = 0; col < block_cols; ++col) {
            for (int index = 0; index < size_; ++index) {
                work[index] = y[left_permutation_[index] + col * size_];
            }
            normalized_fwht(work.data(), size_);
            for (int index = 0; index < size_; ++index) {
                work[index] *= left_signs_[index];
                work[index] *= singular_values_[index];
                work[index] *= right_signs_[index];
            }
            normalized_fwht(work.data(), size_);
            for (int index = 0; index < size_; ++index) {
                x[right_permutation_[index] + col * size_] = work[index];
            }
        }
    }
}

const std::vector<double>& StructuredHadamardOperator::singular_values() const {
    return singular_values_;
}

double StructuredHadamardOperator::frobenius_norm() const {
    return frobenius_norm_;
}

}  // namespace math
}  // namespace ac_rsvd
