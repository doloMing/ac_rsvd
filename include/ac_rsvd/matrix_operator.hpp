#pragma once

namespace ac_rsvd {

class MatrixOperator {
public:
    virtual ~MatrixOperator() = default;

    virtual int rows() const = 0;
    virtual int cols() const = 0;

    // X is n by block_cols. Y is m by block_cols.
    virtual void apply(const double* x, int block_cols, double* y) const = 0;

    // Y is m by block_cols. X is n by block_cols.
    virtual void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const = 0;
};

}  // namespace ac_rsvd
