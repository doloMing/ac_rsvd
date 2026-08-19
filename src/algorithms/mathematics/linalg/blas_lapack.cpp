#include "algorithms/mathematics/linalg/blas_lapack.hpp"

#include <stdexcept>

extern "C" {
void dgemm_(
    const char* transpose_left,
    const char* transpose_right,
    const int* rows,
    const int* cols,
    const int* inner,
    const double* alpha,
    const double* left,
    const int* left_stride,
    const double* right,
    const int* right_stride,
    const double* beta,
    double* result,
    const int* result_stride);

double dnrm2_(const int* size, const double* values, const int* stride);
}

namespace ac_rsvd {
namespace math {

void gemm(
    const Matrix& left,
    bool transpose_left,
    const Matrix& right,
    bool transpose_right,
    double alpha,
    double beta,
    Matrix& result) {
    int left_rows = transpose_left ? left.cols() : left.rows();
    int left_cols = transpose_left ? left.rows() : left.cols();
    int right_rows = transpose_right ? right.cols() : right.rows();
    int right_cols = transpose_right ? right.rows() : right.cols();

    if (left_cols != right_rows ||
        result.rows() != left_rows || result.cols() != right_cols) {
        throw std::invalid_argument("Matrix multiply dimensions do not match");
    }

    if (result.empty()) {
        return;
    }

    if (left_cols == 0) {
        for (int index = 0; index < result.size(); ++index) {
            result.data()[index] *= beta;
        }
        return;
    }

    char left_flag = transpose_left ? 'T' : 'N';
    char right_flag = transpose_right ? 'T' : 'N';
    int rows = left_rows;
    int cols = right_cols;
    int inner = left_cols;
    int left_stride = left.leading_dimension();
    int right_stride = right.leading_dimension();
    int result_stride = result.leading_dimension();

    dgemm_(
        &left_flag,
        &right_flag,
        &rows,
        &cols,
        &inner,
        &alpha,
        left.data(),
        &left_stride,
        right.data(),
        &right_stride,
        &beta,
        result.data(),
        &result_stride);
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows(), right.cols());
    gemm(left, false, right, false, 1.0, 0.0, result);
    return result;
}

Matrix transpose_multiply(const Matrix& left, const Matrix& right) {
    Matrix result(left.cols(), right.cols());
    gemm(left, true, right, false, 1.0, 0.0, result);
    return result;
}

Matrix multiply_transpose(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows(), right.rows());
    gemm(left, false, right, true, 1.0, 0.0, result);
    return result;
}

double vector_norm(const double* values, int size) {
    int stride = 1;
    return dnrm2_(&size, values, &stride);
}

}  // namespace math
}  // namespace ac_rsvd
