#include "algorithms/mathematics/linalg/qr.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void dgeqrf_(
    const int* rows,
    const int* cols,
    double* matrix,
    const int* stride,
    double* tau,
    double* workspace,
    const int* workspace_size,
    int* info);

void dorgqr_(
    const int* rows,
    const int* cols,
    const int* reflectors,
    double* matrix,
    const int* stride,
    const double* tau,
    double* workspace,
    const int* workspace_size,
    int* info);
}

namespace ac_rsvd {
namespace math {

static void check_lapack_info(int info, const char* routine) {
    if (info != 0) {
        throw std::runtime_error(std::string(routine) + " failed");
    }
}

QrResult thin_qr(const Matrix& matrix) {
    QrResult result;
    result.q = matrix;
    thin_qr_in_place(result.q, result.r, result.diagonal);
    return result;
}

void thin_qr_in_place(
    Matrix& matrix,
    Matrix& r,
    std::vector<double>& diagonal) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int thin_cols = std::min(rows, cols);

    r = Matrix(thin_cols, cols);
    diagonal.assign(thin_cols, 0.0);

    if (thin_cols == 0) {
        matrix.truncate_columns(0);
        return;
    }

    std::vector<double> tau(thin_cols);
    int stride = matrix.leading_dimension();
    int info = 0;
    int workspace_size = -1;
    double workspace_query = 0.0;

    dgeqrf_(
        &rows,
        &cols,
        matrix.data(),
        &stride,
        tau.data(),
        &workspace_query,
        &workspace_size,
        &info);
    check_lapack_info(info, "dgeqrf");

    workspace_size = std::max(1, static_cast<int>(std::ceil(workspace_query)));
    std::vector<double> workspace(workspace_size);
    dgeqrf_(
        &rows,
        &cols,
        matrix.data(),
        &stride,
        tau.data(),
        workspace.data(),
        &workspace_size,
        &info);
    check_lapack_info(info, "dgeqrf");

    // LAPACK stores R above the diagonal and Householder data below it.
    for (int col = 0; col < cols; ++col) {
        int last_row = std::min(col, thin_cols - 1);
        for (int row = 0; row <= last_row; ++row) {
            r(row, col) = matrix(row, col);
        }
    }

    matrix.truncate_columns(thin_cols);
    workspace_size = -1;
    workspace_query = 0.0;
    dorgqr_(
        &rows,
        &thin_cols,
        &thin_cols,
        matrix.data(),
        &stride,
        tau.data(),
        &workspace_query,
        &workspace_size,
        &info);
    check_lapack_info(info, "dorgqr");

    workspace_size = std::max(1, static_cast<int>(std::ceil(workspace_query)));
    workspace.resize(workspace_size);
    dorgqr_(
        &rows,
        &thin_cols,
        &thin_cols,
        matrix.data(),
        &stride,
        tau.data(),
        workspace.data(),
        &workspace_size,
        &info);
    check_lapack_info(info, "dorgqr");

    // The nonnegative diagonal gives the sequential residual norms in order.
    for (int index = 0; index < thin_cols; ++index) {
        if (r(index, index) < 0.0) {
            for (int row = 0; row < rows; ++row) {
                matrix(row, index) = -matrix(row, index);
            }
            for (int col = index; col < cols; ++col) {
                r(index, col) = -r(index, col);
            }
        }
        diagonal[index] = r(index, index);
    }
}

}  // namespace math
}  // namespace ac_rsvd
