#include <cmath>
#include <iostream>
#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/projection.hpp"
#include "algorithms/mathematics/linalg/qr.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/linalg/truncation.hpp"
#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"

using ac_rsvd::math::BlockOrthogonalizationResult;
using ac_rsvd::math::Matrix;
using ac_rsvd::math::QrResult;
using ac_rsvd::math::SvdResult;

static double difference_norm(const Matrix& left, const Matrix& right) {
    Matrix difference = left;
    for (int index = 0; index < difference.size(); ++index) {
        difference.data()[index] -= right.data()[index];
    }
    return ac_rsvd::math::frobenius_norm(difference);
}

static Matrix test_matrix() {
    Matrix matrix(5, 3);
    matrix(0, 0) = 1.0;
    matrix(1, 0) = 2.0;
    matrix(2, 0) = -1.0;
    matrix(3, 0) = 0.5;
    matrix(4, 0) = 3.0;

    matrix(0, 1) = -2.0;
    matrix(1, 1) = 1.0;
    matrix(2, 1) = 4.0;
    matrix(3, 1) = 1.5;
    matrix(4, 1) = 0.0;

    matrix(0, 2) = 0.5;
    matrix(1, 2) = -3.0;
    matrix(2, 2) = 2.0;
    matrix(3, 2) = 5.0;
    matrix(4, 2) = 1.0;
    return matrix;
}

static void check_qr(const Matrix& matrix) {
    QrResult qr = ac_rsvd::math::thin_qr(matrix);
    Matrix reconstructed = ac_rsvd::math::multiply(qr.q, qr.r);
    if (difference_norm(matrix, reconstructed) > 1e-12) {
        throw std::runtime_error("QR reconstruction is wrong");
    }

    Matrix gram = ac_rsvd::math::transpose_multiply(qr.q, qr.q);
    for (int col = 0; col < gram.cols(); ++col) {
        for (int row = 0; row < gram.rows(); ++row) {
            double expected = row == col ? 1.0 : 0.0;
            if (std::abs(gram(row, col) - expected) > 1e-12) {
                throw std::runtime_error("Q is not orthonormal");
            }
        }
        if (qr.diagonal[col] < 0.0) {
            throw std::runtime_error("QR diagonal is negative");
        }
    }

    SvdResult qb_svd = ac_rsvd::math::compact_qb_svd(qr.q, qr.r);
    Matrix scaled_u = qb_svd.u;
    for (int col = 0; col < scaled_u.cols(); ++col) {
        for (int row = 0; row < scaled_u.rows(); ++row) {
            scaled_u(row, col) *= qb_svd.singular_values[col];
        }
    }
    reconstructed = ac_rsvd::math::multiply(scaled_u, qb_svd.vt);
    if (difference_norm(matrix, reconstructed) > 1e-11) {
        throw std::runtime_error("QB SVD reconstruction is wrong");
    }
}

static void check_svd(const Matrix& matrix) {
    SvdResult svd = ac_rsvd::math::thin_svd(matrix);
    Matrix scaled_u = svd.u;
    for (int col = 0; col < scaled_u.cols(); ++col) {
        for (int row = 0; row < scaled_u.rows(); ++row) {
            scaled_u(row, col) *= svd.singular_values[col];
        }
    }
    Matrix reconstructed = ac_rsvd::math::multiply(scaled_u, svd.vt);
    if (difference_norm(matrix, reconstructed) > 1e-12) {
        throw std::runtime_error("SVD reconstruction is wrong");
    }

    std::vector<double> tail =
        ac_rsvd::math::squared_tail_sums(svd.singular_values);
    int rank = ac_rsvd::math::smallest_rank_for_tail(
        svd.singular_values, tail[2]);
    if (rank != 2) {
        throw std::runtime_error("Tail rank is wrong");
    }
}

static void check_ordered_block(const Matrix& matrix) {
    Matrix empty_basis(matrix.rows(), 0);
    BlockOrthogonalizationResult block =
        ac_rsvd::math::orthogonalize_block(matrix, empty_basis);

    for (int col = 0; col < matrix.cols(); ++col) {
        Matrix old_q = ac_rsvd::math::copy_block(
            block.q, 0, 0, block.q.rows(), col);
        Matrix current = ac_rsvd::math::copy_block(
            matrix, 0, col, matrix.rows(), 1);
        ac_rsvd::math::ColumnOrthogonalizationResult sequential =
            ac_rsvd::math::orthogonalize_column(current, old_q, 0.0);
        if (std::abs(sequential.norm - block.diagonal[col]) > 1e-12) {
            throw std::runtime_error("Block QR changed the prefix norm");
        }
    }

    QrResult qr;
    qr.q = block.q;
    qr.r = block.r;
    qr.diagonal = block.diagonal;
    Matrix trailing = ac_rsvd::math::trailing_basis(qr, 1);
    Matrix suffix = ac_rsvd::math::copy_block(
        matrix, 0, 1, matrix.rows(), matrix.cols() - 1);
    Matrix first_q = ac_rsvd::math::copy_block(
        block.q, 0, 0, block.q.rows(), 1);
    ac_rsvd::math::project_out(first_q, suffix);
    ac_rsvd::math::project_out(trailing, suffix);
    if (ac_rsvd::math::frobenius_norm(suffix) > 1e-11) {
        throw std::runtime_error("Trailing basis missed part of the suffix");
    }
}

int main() {
    try {
        Matrix matrix = test_matrix();
        Matrix rebuilt(matrix.rows(), 0);
        rebuilt.append_columns(ac_rsvd::math::copy_block(
            matrix, 0, 0, matrix.rows(), 1));
        rebuilt.append_columns(ac_rsvd::math::copy_block(
            matrix, 0, 1, matrix.rows(), matrix.cols() - 1));
        if (difference_norm(matrix, rebuilt) != 0.0) {
            throw std::runtime_error("Column append is wrong");
        }
        check_qr(matrix);
        check_qr(ac_rsvd::math::transpose(matrix));
        check_svd(matrix);
        check_svd(ac_rsvd::math::transpose(matrix));
        check_ordered_block(matrix);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
