#include "algorithms/mathematics/linalg/svd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"

extern "C" {
void dgesdd_(
    const char* vectors,
    const int* rows,
    const int* cols,
    double* matrix,
    const int* stride,
    double* singular_values,
    double* left_vectors,
    const int* left_stride,
    double* right_vectors_transposed,
    const int* right_stride,
    double* workspace,
    const int* workspace_size,
    int* integer_workspace,
    int* info);
}

namespace ac_rsvd {
namespace math {

static void check_svd_info(int info) {
    if (info != 0) {
        throw std::runtime_error("dgesdd failed");
    }
}

SvdResult thin_svd(const Matrix& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int count = std::min(rows, cols);

    SvdResult result;
    result.u = Matrix(rows, count);
    result.singular_values.assign(count, 0.0);
    result.vt = Matrix(count, cols);

    if (count == 0) {
        return result;
    }

    Matrix work_matrix = matrix;
    char vectors = 'S';
    int stride = work_matrix.leading_dimension();
    int left_stride = result.u.leading_dimension();
    int right_stride = result.vt.leading_dimension();
    int workspace_size = -1;
    double workspace_query = 0.0;
    std::vector<int> integer_workspace(8 * count);
    int info = 0;

    dgesdd_(
        &vectors,
        &rows,
        &cols,
        work_matrix.data(),
        &stride,
        result.singular_values.data(),
        result.u.data(),
        &left_stride,
        result.vt.data(),
        &right_stride,
        &workspace_query,
        &workspace_size,
        integer_workspace.data(),
        &info);
    check_svd_info(info);

    workspace_size = std::max(1, static_cast<int>(std::ceil(workspace_query)));
    std::vector<double> workspace(workspace_size);
    work_matrix = matrix;
    dgesdd_(
        &vectors,
        &rows,
        &cols,
        work_matrix.data(),
        &stride,
        result.singular_values.data(),
        result.u.data(),
        &left_stride,
        result.vt.data(),
        &right_stride,
        workspace.data(),
        &workspace_size,
        integer_workspace.data(),
        &info);
    check_svd_info(info);

    return result;
}

SvdResult compact_qb_svd(const Matrix& q, const Matrix& b) {
    if (q.cols() != b.rows()) {
        throw std::invalid_argument("Q and B dimensions do not match");
    }

    // If B = U_b S V^T, then QB = (Q U_b) S V^T.
    SvdResult result = thin_svd(b);
    result.u = multiply(q, result.u);
    return result;
}

SvdResult truncate_svd(const SvdResult& svd, int rank) {
    int available = static_cast<int>(svd.singular_values.size());
    if (rank < 0 || rank > available) {
        throw std::invalid_argument("SVD rank is out of range");
    }

    SvdResult result;
    result.u = copy_block(svd.u, 0, 0, svd.u.rows(), rank);
    result.singular_values.assign(
        svd.singular_values.begin(), svd.singular_values.begin() + rank);
    result.vt = copy_block(svd.vt, 0, 0, rank, svd.vt.cols());
    return result;
}

Matrix column_space(const Matrix& matrix, double relative_tolerance) {
    SvdResult svd = thin_svd(matrix);
    if (svd.singular_values.empty()) {
        return Matrix(matrix.rows(), 0);
    }

    double tolerance = relative_tolerance;
    if (tolerance < 0.0) {
        // This is the usual numerical-rank cutoff scaled by the largest value.
        tolerance = std::numeric_limits<double>::epsilon() *
                    std::max(matrix.rows(), matrix.cols());
    }

    double cutoff = tolerance * svd.singular_values[0];
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           svd.singular_values[rank] > cutoff) {
        ++rank;
    }
    return copy_block(svd.u, 0, 0, matrix.rows(), rank);
}

}  // namespace math
}  // namespace ac_rsvd
