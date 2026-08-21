#include "algorithms/mathematics/linalg/svd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/truncation.hpp"

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

void dsyrk_(
    const char* uplo,
    const char* transpose,
    const int* size,
    const int* inner,
    const double* alpha,
    const double* matrix,
    const int* stride,
    const double* beta,
    double* result,
    const int* result_stride);

void dsyevd_(
    const char* vectors,
    const char* uplo,
    const int* size,
    double* matrix,
    const int* stride,
    double* eigenvalues,
    double* workspace,
    const int* workspace_size,
    int* integer_workspace,
    const int* integer_workspace_size,
    int* info);
}

namespace ac_rsvd {
namespace math {

static void check_svd_info(int info) {
    if (info != 0) {
        throw std::runtime_error("dgesdd failed");
    }
}

static void check_eigen_info(int info) {
    if (info != 0) {
        throw std::runtime_error("dsyevd failed");
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

double largest_symmetric_eigenvalue_in_place(Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Symmetric eigenvalue needs a square matrix");
    }

    int size = matrix.rows();
    if (size == 0) {
        return 0.0;
    }

    std::vector<double> eigenvalues(size);
    char vectors = 'N';
    char upper = 'U';
    int stride = matrix.leading_dimension();
    int workspace_size = -1;
    double workspace_query = 0.0;
    int integer_workspace_size = -1;
    int integer_workspace_query = 0;
    int info = 0;

    dsyevd_(
        &vectors,
        &upper,
        &size,
        matrix.data(),
        &stride,
        eigenvalues.data(),
        &workspace_query,
        &workspace_size,
        &integer_workspace_query,
        &integer_workspace_size,
        &info);
    check_eigen_info(info);

    workspace_size = std::max(1, static_cast<int>(std::ceil(workspace_query)));
    integer_workspace_size = std::max(1, integer_workspace_query);
    std::vector<double> workspace(workspace_size);
    std::vector<int> integer_workspace(integer_workspace_size);
    dsyevd_(
        &vectors,
        &upper,
        &size,
        matrix.data(),
        &stride,
        eigenvalues.data(),
        workspace.data(),
        &workspace_size,
        integer_workspace.data(),
        &integer_workspace_size,
        &info);
    check_eigen_info(info);
    return eigenvalues[size - 1];
}

TallSvdResult stable_tall_svd_in_place(Matrix& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    if (rows < cols) {
        throw std::invalid_argument("Tall SVD needs at least as many rows as columns");
    }

    TallSvdResult result;
    result.selected_rank = cols;
    result.singular_values.assign(cols, 0.0);
    result.vt = Matrix(cols, cols);
    if (cols == 0) {
        return result;
    }

    // LAPACK leaves the left singular vectors in the input matrix.
    char vectors = 'O';
    int stride = matrix.leading_dimension();
    int left_stride = 1;
    int right_stride = result.vt.leading_dimension();
    int workspace_size = -1;
    double workspace_query = 0.0;
    double unused_left_vector = 0.0;
    std::vector<int> integer_workspace(8 * cols);
    int info = 0;

    dgesdd_(
        &vectors,
        &rows,
        &cols,
        matrix.data(),
        &stride,
        result.singular_values.data(),
        &unused_left_vector,
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
    dgesdd_(
        &vectors,
        &rows,
        &cols,
        matrix.data(),
        &stride,
        result.singular_values.data(),
        &unused_left_vector,
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

enum class ExactBudgetState {
    feasible,
    certificate_miss,
    inconsistent_norm
};

static ExactBudgetState set_exact_frobenius_budget(
    double matrix_norm_squared,
    double projected_norm_squared,
    double tolerance_squared,
    TallSvdResult& result) {
    if (!std::isfinite(projected_norm_squared)) {
        return ExactBudgetState::inconsistent_norm;
    }

    double residual = matrix_norm_squared - projected_norm_squared;
    double scale = std::max(matrix_norm_squared, projected_norm_squared);
    double roundoff =
        512.0 * std::numeric_limits<double>::epsilon() * scale;
    if (residual < 0.0 && residual >= -roundoff) {
        residual = 0.0;
    }
    if (residual < 0.0) {
        return ExactBudgetState::inconsistent_norm;
    }

    result.exact_frobenius_budget_used = true;
    result.projected_norm_squared = projected_norm_squared;
    result.residual_norm_squared = residual;
    result.tail_budget_squared =
        std::max(0.0, tolerance_squared - residual);
    result.exact_frobenius_budget_feasible =
        residual <= tolerance_squared;
    return result.exact_frobenius_budget_feasible
        ? ExactBudgetState::feasible
        : ExactBudgetState::certificate_miss;
}

static bool try_safe_gram_svd(
    Matrix& matrix,
    bool use_tail_budget,
    double tail_budget,
    bool use_exact_frobenius_budget,
    double matrix_norm_squared,
    double tolerance_squared,
    TallSvdResult& result) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    if (cols == 0) {
        if (use_exact_frobenius_budget) {
            ExactBudgetState state = set_exact_frobenius_budget(
                matrix_norm_squared, 0.0, tolerance_squared, result);
            if (state == ExactBudgetState::inconsistent_norm) {
                return false;
            }
        }
        result.selected_rank = 0;
        result.used_gram = true;
        return true;
    }
    if (static_cast<long long>(rows) < 8LL * cols) {
        return false;
    }
    if (use_tail_budget && !std::isfinite(tail_budget)) {
        return false;
    }

    Matrix gram(cols, cols);
    char upper = 'U';
    char transpose = 'T';
    double alpha = 1.0;
    double beta = 0.0;
    int matrix_stride = matrix.leading_dimension();
    int gram_stride = gram.leading_dimension();
    dsyrk_(
        &upper,
        &transpose,
        &cols,
        &rows,
        &alpha,
        matrix.data(),
        &matrix_stride,
        &beta,
        gram.data(),
        &gram_stride);

    if (use_exact_frobenius_budget) {
        long double projected_norm_squared = 0.0L;
        for (int index = 0; index < cols; ++index) {
            projected_norm_squared += gram(index, index);
        }
        ExactBudgetState state = set_exact_frobenius_budget(
            matrix_norm_squared,
            static_cast<double>(projected_norm_squared),
            tolerance_squared,
            result);
        if (state == ExactBudgetState::inconsistent_norm) {
            return false;
        }
        tail_budget = result.tail_budget_squared;
        if (state == ExactBudgetState::certificate_miss) {
            result.selected_rank = 0;
            result.used_gram = true;
            return true;
        }
    }

    std::vector<double> eigenvalues(cols);
    char vectors = 'V';
    int workspace_size = -1;
    double workspace_query = 0.0;
    int integer_workspace_size = -1;
    int integer_workspace_query = 0;
    int info = 0;
    dsyevd_(
        &vectors,
        &upper,
        &cols,
        gram.data(),
        &gram_stride,
        eigenvalues.data(),
        &workspace_query,
        &workspace_size,
        &integer_workspace_query,
        &integer_workspace_size,
        &info);
    if (info != 0) {
        return false;
    }

    workspace_size = std::max(1, static_cast<int>(std::ceil(workspace_query)));
    std::vector<double> workspace(workspace_size);
    integer_workspace_size = std::max(1, integer_workspace_query);
    std::vector<int> integer_workspace(integer_workspace_size);
    dsyevd_(
        &vectors,
        &upper,
        &cols,
        gram.data(),
        &gram_stride,
        eigenvalues.data(),
        workspace.data(),
        &workspace_size,
        integer_workspace.data(),
        &integer_workspace_size,
        &info);
    if (info != 0) {
        return false;
    }

    for (double value : eigenvalues) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    double largest_eigenvalue = eigenvalues[cols - 1];
    if (largest_eigenvalue <= 0.0) {
        return false;
    }

    result.singular_values.resize(cols);
    result.vt = Matrix(cols, cols);
    std::vector<long double> tail(cols + 1, 0.0L);
    long double negative_mass = 0.0L;
    for (int index = cols - 1; index >= 0; --index) {
        int source = cols - 1 - index;
        double eigenvalue = eigenvalues[source];
        if (eigenvalue < 0.0) {
            negative_mass -= eigenvalue;
            eigenvalue = 0.0;
        }
        result.singular_values[index] = std::sqrt(eigenvalue);
        tail[index] = tail[index + 1] + eigenvalue;
        for (int row = 0; row < cols; ++row) {
            result.vt(index, row) = gram(row, source);
        }
    }

    int selected_rank = cols;
    if (use_tail_budget) {
        selected_rank = 0;
        long double budget = tail_budget;
        while (selected_rank <= cols && tail[selected_rank] > budget) {
            ++selected_rank;
        }
        if (selected_rank > cols) {
            return false;
        }
    }
    result.selected_rank = selected_rank;

    if (selected_rank > 0) {
        int retained_index = cols - selected_rank;
        double retained_eigenvalue = eigenvalues[retained_index];
        if (retained_eigenvalue / largest_eigenvalue < 1e-7) {
            return false;
        }
    }

    long double total_energy = tail[0];
    long double relative_guard = std::max(
        1e-10L,
        64.0L * std::numeric_limits<double>::epsilon() * rows);
    long double tail_guard = relative_guard * total_energy;
    if (negative_mass > tail_guard) {
        return false;
    }

    if (use_tail_budget) {
        long double budget = tail_budget;
        long double lower_margin = budget - tail[selected_rank];
        long double upper_margin = std::numeric_limits<long double>::infinity();
        if (selected_rank > 0) {
            upper_margin = tail[selected_rank - 1] - budget;
        }
        if (std::min(lower_margin, upper_margin) < tail_guard) {
            return false;
        }
    }

    Matrix coefficients(cols, selected_rank);
    for (int col = 0; col < selected_rank; ++col) {
        double singular_value = result.singular_values[col];
        for (int row = 0; row < cols; ++row) {
            coefficients(row, col) =
                result.vt(col, row) / singular_value;
        }
    }
    Matrix left_vectors = multiply(matrix, coefficients);

    if (selected_rank > 0) {
        Matrix left_gram(selected_rank, selected_rank);
        int left_stride = left_vectors.leading_dimension();
        int left_gram_stride = left_gram.leading_dimension();
        dsyrk_(
            &upper,
            &transpose,
            &selected_rank,
            &rows,
            &alpha,
            left_vectors.data(),
            &left_stride,
            &beta,
            left_gram.data(),
            &left_gram_stride);

        long double orthogonality_squared = 0.0L;
        for (int col = 0; col < selected_rank; ++col) {
            for (int row = 0; row <= col; ++row) {
                long double expected = row == col ? 1.0L : 0.0L;
                long double difference = left_gram(row, col) - expected;
                long double weight = row == col ? 1.0L : 2.0L;
                orthogonality_squared += weight * difference * difference;
            }
        }
        if (std::sqrt(orthogonality_squared) > 1e-8L) {
            return false;
        }
    }

    left_vectors.give_values_to(matrix);
    result.used_gram = true;
    return true;
}

TallSvdResult tall_svd_in_place(Matrix& matrix) {
    if (matrix.rows() < matrix.cols()) {
        throw std::invalid_argument("Tall SVD needs at least as many rows as columns");
    }

    TallSvdResult result;
    if (try_safe_gram_svd(
            matrix, false, 0.0, false, 0.0, 0.0, result)) {
        return result;
    }
    return stable_tall_svd_in_place(matrix);
}

TallSvdResult tall_svd_in_place(Matrix& matrix, double tail_budget) {
    if (matrix.rows() < matrix.cols()) {
        throw std::invalid_argument("Tall SVD needs at least as many rows as columns");
    }

    TallSvdResult result;
    if (try_safe_gram_svd(
            matrix, true, tail_budget, false, 0.0, 0.0, result)) {
        return result;
    }

    result = stable_tall_svd_in_place(matrix);
    result.selected_rank = smallest_rank_for_tail(
        result.singular_values, tail_budget);
    matrix.truncate_columns(result.selected_rank);
    return result;
}

TallSvdResult tall_svd_exact_frobenius_in_place(
    Matrix& matrix,
    double matrix_norm_squared,
    double tolerance_squared) {
    if (matrix.rows() < matrix.cols()) {
        throw std::invalid_argument("Tall SVD needs at least as many rows as columns");
    }
    if (!std::isfinite(matrix_norm_squared) || matrix_norm_squared < 0.0 ||
        !std::isfinite(tolerance_squared) || tolerance_squared < 0.0) {
        throw std::invalid_argument("Exact Frobenius budget must be finite");
    }

    TallSvdResult result;
    if (try_safe_gram_svd(
            matrix,
            true,
            0.0,
            true,
            matrix_norm_squared,
            tolerance_squared,
            result)) {
        return result;
    }

    result = stable_tall_svd_in_place(matrix);
    long double projected_norm_squared = 0.0L;
    for (double singular_value : result.singular_values) {
        projected_norm_squared +=
            static_cast<long double>(singular_value) * singular_value;
    }
    ExactBudgetState state = set_exact_frobenius_budget(
        matrix_norm_squared,
        static_cast<double>(projected_norm_squared),
        tolerance_squared,
        result);
    if (state == ExactBudgetState::inconsistent_norm) {
        throw std::invalid_argument(
            "Exact Frobenius norm is below the captured energy");
    }
    if (state == ExactBudgetState::certificate_miss) {
        result.selected_rank = 0;
        return result;
    }

    result.selected_rank = smallest_rank_for_tail(
        result.singular_values, result.tail_budget_squared);
    matrix.truncate_columns(result.selected_rank);
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

Matrix column_space(
    const Matrix& matrix,
    double relative_tolerance,
    double absolute_tolerance) {
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

    double cutoff = std::max(
        tolerance * svd.singular_values[0],
        absolute_tolerance);
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           svd.singular_values[rank] > cutoff) {
        ++rank;
    }
    return copy_block(svd.u, 0, 0, matrix.rows(), rank);
}

}  // namespace math
}  // namespace ac_rsvd
