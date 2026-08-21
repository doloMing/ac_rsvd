#pragma once

#include <vector>

namespace ac_rsvd {
namespace math {

// A small column-major matrix passed directly to BLAS and LAPACK.
class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);
    Matrix(int rows, int cols, const double* values);

    int rows() const;
    int cols() const;
    int leading_dimension() const;
    int size() const;
    bool empty() const;

    double* data();
    const double* data() const;
    double* column_data(int col);
    const double* column_data(int col) const;

    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    void fill(double value);
    void append_columns(const Matrix& block);
    void append_columns(const Matrix& block, int count);
    void truncate_columns(int count);
    void give_values_to(std::vector<double>& output);
    void give_values_to(Matrix& output);

private:
    int rows_;
    int cols_;
    std::vector<double> values_;
};

Matrix copy_block(
    const Matrix& source,
    int first_row,
    int first_col,
    int row_count,
    int col_count);

void set_block(Matrix& target, int first_row, int first_col, const Matrix& block);

Matrix join_columns(const Matrix& left, const Matrix& right);
Matrix transpose(const Matrix& matrix);

double squared_frobenius_norm(const Matrix& matrix);
double frobenius_norm(const Matrix& matrix);

}  // namespace math
}  // namespace ac_rsvd
