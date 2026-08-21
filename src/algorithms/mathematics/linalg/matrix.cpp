#include "algorithms/mathematics/linalg/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ac_rsvd {
namespace math {

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(int rows, int cols) : Matrix(rows, cols, 0.0) {}

Matrix::Matrix(int rows, int cols, double value) : rows_(rows), cols_(cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be nonnegative");
    }
    long long element_count = static_cast<long long>(rows) * cols;
    if (element_count > std::numeric_limits<int>::max()) {
        throw std::length_error("Matrix is too large for the LP64 backend");
    }
    values_.assign(static_cast<std::size_t>(element_count), value);
}

Matrix::Matrix(int rows, int cols, const double* values) : Matrix(rows, cols) {
    if (!values_.empty()) {
        std::copy(values, values + values_.size(), values_.begin());
    }
}

int Matrix::rows() const {
    return rows_;
}

int Matrix::cols() const {
    return cols_;
}

int Matrix::leading_dimension() const {
    return std::max(1, rows_);
}

int Matrix::size() const {
    return static_cast<int>(values_.size());
}

bool Matrix::empty() const {
    return rows_ == 0 || cols_ == 0;
}

double* Matrix::data() {
    return values_.data();
}

const double* Matrix::data() const {
    return values_.data();
}

double* Matrix::column_data(int col) {
    return values_.data() + col * rows_;
}

const double* Matrix::column_data(int col) const {
    return values_.data() + col * rows_;
}

double& Matrix::operator()(int row, int col) {
    return values_[row + col * rows_];
}

double Matrix::operator()(int row, int col) const {
    return values_[row + col * rows_];
}

void Matrix::fill(double value) {
    std::fill(values_.begin(), values_.end(), value);
}

void Matrix::append_columns(const Matrix& block) {
    append_columns(block, block.cols());
}

void Matrix::append_columns(const Matrix& block, int count) {
    if (rows_ != block.rows()) {
        throw std::invalid_argument("Matrices must have the same row count");
    }
    if (count < 0 || count > block.cols()) {
        throw std::invalid_argument("Column count is out of range");
    }
    long long added_size = static_cast<long long>(rows_) * count;
    long long new_size = static_cast<long long>(size()) + added_size;
    long long new_column_count =
        static_cast<long long>(cols_) + count;
    if (new_size > std::numeric_limits<int>::max() ||
        new_column_count > std::numeric_limits<int>::max()) {
        throw std::length_error("Matrix is too large for the LP64 backend");
    }
    if (added_size > 0) {
        // Whole columns are contiguous, so appending is one vector insertion.
        values_.insert(
            values_.end(), block.data(), block.data() + added_size);
    }
    cols_ = static_cast<int>(new_column_count);
}

void Matrix::truncate_columns(int count) {
    if (count < 0 || count > cols_) {
        throw std::invalid_argument("Column count is out of range");
    }
    values_.resize(static_cast<std::size_t>(rows_) * count);
    cols_ = count;
}

void Matrix::give_values_to(std::vector<double>& output) {
    output.clear();
    output.swap(values_);
    rows_ = 0;
    cols_ = 0;
}

void Matrix::give_values_to(Matrix& output) {
    output.rows_ = rows_;
    output.cols_ = cols_;
    output.values_.clear();
    output.values_.swap(values_);
    rows_ = 0;
    cols_ = 0;
}

Matrix copy_block(
    const Matrix& source,
    int first_row,
    int first_col,
    int row_count,
    int col_count) {
    if (first_row < 0 || first_col < 0 || row_count < 0 || col_count < 0 ||
        first_row + row_count > source.rows() ||
        first_col + col_count > source.cols()) {
        throw std::invalid_argument("Matrix block is out of range");
    }

    Matrix block(row_count, col_count);
    for (int col = 0; col < col_count; ++col) {
        for (int row = 0; row < row_count; ++row) {
            block(row, col) = source(first_row + row, first_col + col);
        }
    }
    return block;
}

void set_block(Matrix& target, int first_row, int first_col, const Matrix& block) {
    if (first_row < 0 || first_col < 0 ||
        first_row + block.rows() > target.rows() ||
        first_col + block.cols() > target.cols()) {
        throw std::invalid_argument("Matrix block is out of range");
    }

    for (int col = 0; col < block.cols(); ++col) {
        for (int row = 0; row < block.rows(); ++row) {
            target(first_row + row, first_col + col) = block(row, col);
        }
    }
}

Matrix join_columns(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows()) {
        throw std::invalid_argument("Matrices must have the same row count");
    }
    long long column_count =
        static_cast<long long>(left.cols()) + right.cols();
    if (column_count > std::numeric_limits<int>::max()) {
        throw std::length_error("Matrix has too many columns");
    }

    Matrix joined(left.rows(), static_cast<int>(column_count));
    set_block(joined, 0, 0, left);
    set_block(joined, 0, left.cols(), right);
    return joined;
}

Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix.cols(), matrix.rows());
    for (int col = 0; col < matrix.cols(); ++col) {
        for (int row = 0; row < matrix.rows(); ++row) {
            result(col, row) = matrix(row, col);
        }
    }
    return result;
}

double squared_frobenius_norm(const Matrix& matrix) {
    double sum = 0.0;
    for (int index = 0; index < matrix.size(); ++index) {
        sum += matrix.data()[index] * matrix.data()[index];
    }
    return sum;
}

double frobenius_norm(const Matrix& matrix) {
    return std::sqrt(squared_frobenius_norm(matrix));
}

}  // namespace math
}  // namespace ac_rsvd
