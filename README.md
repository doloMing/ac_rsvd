# AC-RSVD: Anytime-Certified Randomized SVD

AC-RSVD computes a compact low-rank approximation of a large real matrix from
products with the matrix and its transpose. The user supplies an absolute
Frobenius-error tolerance instead of a target rank. The algorithm grows the
sampled range, decides when the range is accurate enough, and then chooses the
smallest output rank supported by the remaining error allowance.

For a matrix $A\in\mathbb{R}^{m\times n}$, a tolerance $\tau>0$, and a
failure probability $0<\delta<1$, AC-RSVD returns

$$
A \approx U_k\mathrm{diag}(s_k)V_k^\top
$$

with the exact-arithmetic guarantee

$$
\Pr(\|A-U_k\mathrm{diag}(s_k)V_k^\top\|_F\leq\tau)
\geq 1-\delta.
$$

The output rank $k$ is determined during the calculation. No target rank or
model for the singular-value decay is required.

The repository contains:

- **AC-RSVD**, the default fixed-tolerance algorithm;
- **AC-RSVD-Fro**, which uses an available exact value of $\|A\|_F^2$ to
  reduce the final rank without changing the sampled range or matrix products;
- **randQB-EI**, a blocked adaptive QB baseline using an incremental error
  indicator; and
- **randQB-MF-Fro**, a matrix-free adaptive QB baseline using Gaussian
  Frobenius-residual estimates.

The implementation is written in C++20 and uses BLAS, LAPACK, and OpenMP. A
PyTorch CPU extension calls the same C++ algorithm core.

**Release:** [v0.1.0](https://github.com/doloMing/ac_rsvd/releases/tag/v0.1.0)
&nbsp; | &nbsp; **License:** MIT &nbsp; | &nbsp; **Language:** C++20 / PyTorch CPU

## Contents

- [1. Repository structure](#1-repository-structure)
- [2. Choosing and calling an algorithm](#2-choosing-and-calling-an-algorithm)
- [3. Inputs, outputs, counters, and diagnostics](#3-inputs-outputs-counters-and-diagnostics)
- [4. Build and deployment](#4-build-and-deployment)
- [5. C++ and PyTorch examples](#5-c-and-pytorch-examples)
- [6. Algorithm structure and pseudocode](#6-algorithm-structure-and-pseudocode)
- [7. Mathematical principles](#7-mathematical-principles)
- [8. Why AC-RSVD is faster and gives reliable accuracy](#8-why-ac-rsvd-is-faster-and-gives-reliable-accuracy)
- [9. Paper, citation, author, and license](#9-paper-citation-author-and-license)

## 1. Repository structure

```text
ac_rsvd/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/ac_rsvd/
│   ├── ac_rsvd.hpp                 # AC-RSVD and AC-RSVD-Fro options/API
│   ├── factorization_result.hpp     # factors, status, counters, diagnostics
│   ├── matrix_operator.hpp          # matrix-free A and A^T interface
│   ├── randqb_ei.hpp                # randQB-EI API
│   └── randqb_mf_fro.hpp            # randQB-MF-Fro API
├── src/algorithms/
│   ├── ac_rsvd/                     # AC-RSVD state machine
│   ├── randqb_ei/                    # randQB-EI implementation
│   ├── randqb_mf_fro/                # randQB-MF-Fro implementation
│   └── mathematics/
│       ├── analysis/                 # analytic rank and product bounds
│       ├── certificate/              # e-process updates and inversion
│       ├── linalg/                   # BLAS/LAPACK matrix, QR, and SVD code
│       ├── operators/                # internal dense and Hadamard operators
│       ├── orthogonalization/        # blocked and sequential QR paths
│       └── random/                   # counter-based Gaussian generator
└── bindings/torch/
    └── torch_binding.cpp             # torch.ops.ac_rsvd entry points
```

### Public C++ interface

The public headers under `include/ac_rsvd/` are sufficient for applications
that provide their own matrix operator. The algorithm only requires four
operations:

```cpp
class MatrixOperator {
public:
    virtual int rows() const = 0;
    virtual int cols() const = 0;
    virtual void apply(const double* x, int block_cols, double* y) const = 0;
    virtual void apply_transpose(
        const double* y, int block_cols, double* x) const = 0;
};
```

For `apply`, `x` is an $n\times b$ column-major block and `y` receives the
$m\times b$ block $Ax$. For `apply_transpose`, `y` is an $m\times b$
column-major block and `x` receives $A^\top y$. The operator may represent a
stored dense or sparse matrix, a streamed matrix, a fast transform, or a
discretized operator. AC-RSVD does not form the full residual matrix.

### Internal components

The C++ AC-RSVD state machine makes every stopping decision. Matrix products,
fixed-order QR, certificate updates, the optional diagnostic, the terminal
projected SVD, and work counters are also executed in C++. The PyTorch binding
copies a CPU tensor into the C++ matrix representation and returns the same
result fields as tensors.

## 2. Choosing and calling an algorithm

### 2.1 Which method should be used?

| Method | Required information | Output rule | Recommended role |
|---|---|---|---|
| **AC-RSVD** | `A`, `A^T`, $\tau$, and $\delta$ | Certified range residual plus projected-SVD tail | Default fixed-tolerance factorization |
| **AC-RSVD-Fro** | AC-RSVD inputs and exact $\|A\|_F^2$ | Exact projection residual plus projected-SVD tail | Use when the exact norm is already stored or can be accumulated while forming the matrix |
| **randQB-EI** | `A`, `A^T`, $\tau$, and $\|A\|_F$ | Incremental captured-energy estimate | Reproduction and comparison with the EI baseline |
| **randQB-MF-Fro** | `A`, `A^T`, and $\tau$ | Gaussian residual estimate and final-block pruning | Reproduction and comparison with the matrix-free baseline |

Use **AC-RSVD** when the matrix is available only through operator products or
when an exact Frobenius norm would require an additional full pass over the
matrix. Use **AC-RSVD-Fro** when $\|A\|_F^2$ is reliable side information.
For an explicitly stored or streamed matrix, this scalar can be accumulated as
the entries are read. For a black-box operator, obtaining the exact norm can
require applying $A$ to every coordinate vector, so AC-RSVD is usually the
appropriate interface.

The two `randQB` routines are included as baseline implementations. Their
residual estimates are not AC-RSVD certificates and their result fields should
be interpreted according to their own stopping rules.

### 2.2 AC-RSVD C++ options

```cpp
ac_rsvd::AcRsvdOptions options;
options.tolerance = 1e-8;
options.failure_probability = 1e-6;
options.block_size = 16;
options.use_enhanced_mode = true;
options.diagnostic_test_size = 512;
options.seed = 0;
options.stream = 0;
options.exact_frobenius_norm_squared = std::nullopt;
options.use_sequential_orthogonalization = false;
```

| Option | Valid values and meaning |
|---|---|
| `tolerance` | Absolute Frobenius tolerance $\tau>0$. The returned approximation targets error at most $\tau$. |
| `failure_probability` | Total failure probability $0<\delta<1$. Smaller values require stronger statistical evidence and can increase the sampled range. |
| `block_size` | Positive block width for products and QR. The default value `16` is an empirical performance choice. Fixed-order processing keeps the stopping decisions in column order. |
| `use_enhanced_mode` | `true` uses the ordinary e-process and the optional fixed-basis diagnostic described in the paper. `false` uses the basic ordered-frame path. |
| `diagnostic_test_size` | Maximum number of held-out diagnostic directions in enhanced mode. It must be between `0` and `512`; `0` disables the diagnostic. The default ceiling `512` is empirical. |
| `seed` | Seed for the counter-based Gaussian generator. Use the same seed and stream to replay the same randomized path. |
| `stream` | Independent random stream identifier. It separates runs without changing the global seed. |
| `exact_frobenius_norm_squared` | Leave empty for AC-RSVD. Set it to the exact scalar $\|A\|_F^2$ for AC-RSVD-Fro. It is used only after the range and terminal projected matrix have been fixed. |
| `use_sequential_orthogonalization` | `false` uses blocked Householder QR. `true` retains block operator calls but changes QR to the column-by-column ablation path. It is intended for algorithm checks rather than normal deployment. |

The implementation uses empirical mixture weights, scale grids, diagnostic
sizes, trigger values, and a failure-probability allocation. These values
control efficiency. The validity proof requires predictable nonnegative
mixture components and a failure allocation whose total is at most $\delta$;
it does not require the particular empirical values used in this release.

### 2.3 Fixed empirical parameters

The enhanced implementation fixes the following values so that runs can be
reproduced from the public options and random seed:

| Component | Value used in version 0.1.0 |
|---|---|
| Fixed-scale grid | $J=1+\lceil2\log_2 n\rceil$ scales, with $\gamma_j=2^{-(j-1)}/3$ |
| Start-time weights | Harmonic weights $a_t=1/((t+1)H_n)$, where $H_n=\sum_{j=1}^n1/j$ |
| Ordinary mixture | Weight `0.10` for the start-time component and weight `0.225` for each of four history-based components |
| History windows | `8`, `16`, `32`, and `64` ordinary observations |
| History scale limits | Empirical mean floor `0.05`, available-residual floor `0.05`, and scale cap $1/3$ |
| Diagnostic trigger history | Mean of the latest `32` ordinary observations divided by $\tau^2$ |
| Diagnostic trigger | `0.925` for range rank at most `128`, `1.0` for ranks `129`–`256`, and `0.90` above `256` |
| Diagnostic refinement target | $0.95\tau^2$ for range rank at most `128` and $0.98\tau^2$ above `128` |
| Diagnostic blocks | `32` pilot directions and held-out blocks of width `32` |
| Failure-probability allocation | $0.05\delta$ for the ordinary process, $0.25\delta$ for the pilot spectral cap, and $0.70\delta$ for held-out testing |

The weights, windows, scale values, trigger thresholds, refinement targets,
block sizes, and probability fractions in this table are empirical choices.
The theorem does not require these particular numbers. It requires each scale
used at a round to be determined before that round's observation, all mixture
weights to be nonnegative and sum to one, and the failure allowances to sum to
at most $\delta$.

### 2.4 Absolute and relative tolerances

The public API accepts an absolute tolerance. If an application requests a
relative Frobenius tolerance $\varepsilon$, convert it before the call:

$$
\tau=\varepsilon\|A\|_F.
$$

This conversion needs a known or separately computed norm. It does not require
AC-RSVD-Fro: the default algorithm can still be used after $\tau$ has been
formed. Supplying `exact_frobenius_norm_squared` additionally changes the final
rank rule and selects the AC-RSVD-Fro path.

### 2.5 Baseline options

`RandQbEiOptions` accepts `tolerance`, the exact Frobenius norm itself in
`frobenius_norm` (not its square), `block_size`, `seed`, and `stream`.

`RandQbMfFroOptions` accepts `tolerance`, `block_size`, `seed`, and `stream`.

Both baselines return `FactorizationResult`, so factors, ranks, work counters,
and timings use the same data layout as AC-RSVD.

## 3. Inputs, outputs, counters, and diagnostics

### 3.1 Compact factorization

The C++ call

```cpp
ac_rsvd::FactorizationResult result =
    ac_rsvd::compute_ac_rsvd(matrix_operator, options);
```

returns the following main fields:

| Field | Interpretation |
|---|---|
| `rows`, `cols` | Shape $m\times n$ of the input matrix. |
| `rank` | Returned factor rank $k$. |
| `range_rank` | Dimension of the final sampled range before projected-SVD truncation. |
| `u` | Column-major $m\times k$ matrix $U_k$. |
| `singular_values` | Length-$k$ vector of retained singular values. |
| `v` | Column-major $n\times k$ matrix $V_k$. Reconstruct with $U_k\mathrm{diag}(s_k)V_k^\top$. |
| `status` | `success = 0` or, for AC-RSVD-Fro, `certificate_miss = 1`. Always inspect this field before using exact-Fro factors. |
| `stop_reason` | Why range construction ended. The enum values are listed below. |
| `residual_bound_source` | Source of the residual value used at termination. |
| `residual_bound_squared` | AC-RSVD scalar upper bound $U_{\rm res}$ for the squared range residual. The subscript distinguishes it from the left factor $U_k$. |
| `truncation_budget_squared` | Squared projected-SVD tail allowance used to choose $k$. |
| `residual_estimate_squared` | Exact terminal projection residual on AC-RSVD-Fro; method-specific estimate on a baseline. |
| `direct_error_squared` | Independent post-run error when a built-in known-spectrum operator computes it; otherwise `-1`. |

The approximation can be applied without forming a dense $m\times n$
matrix:

$$
x\longmapsto U_k\bigl(s_k\odot(V_k^\top x)\bigr).
$$

It stores $O((m+n)k)$ numbers and applies the approximation in
$O((m+n)k)$ arithmetic operations.

### 3.2 AC-RSVD-Fro status

AC-RSVD-Fro uses the same random samples, stopping history, final range, and
products with $A$ and $A^\top$ as the paired AC-RSVD run.

- `status == 0`: the exact projection residual is no larger than $\tau^2$,
  and the returned factor arrays contain the requested approximation.
- `status == 1`: the sampled range is insufficient under the supplied exact
  norm. The factor arrays are empty, while the trace, stop reason, range rank,
  counters, and timings remain available.

The supplied scalar must equal the mathematical $\|A\|_F^2$. The algorithm
rejects nonfinite or negative values and values smaller than the energy already
captured by the projected matrix.

### 3.3 Stop reasons

| Integer | C++ enum | Meaning |
|---:|---|---|
| `0` | `certificate` | The ordinary e-process or diagnostic certified the current range. |
| `1` | `tolerance_met` | A baseline residual indicator met its tolerance. |
| `2` | `full_output_space` | The basis spans the full output space. |
| `3` | `full_rank` | A baseline reached the maximum possible rank. |
| `4` | `final_input_direction` | All input directions have been processed. |
| `5` | `zero_residual` | The new residual direction is exactly zero. |

### 3.4 Residual-bound sources

| Integer | C++ enum | Meaning |
|---:|---|---|
| `0` | `none` | No AC-RSVD residual certificate is attached, as on a baseline result. |
| `1` | `global_certificate` | The ordinary e-process supplied the bound. |
| `2` | `diagnostic_certificate` | Held-out diagnostic observations supplied the bound. |
| `3` | `diagnostic_spectral_cap` | The pilot spectral cap alone certified the range. |
| `4` | `exact_residual` | Range exhaustion or an exact-zero terminal state gives zero residual. |

### 3.5 Work counters

`result.statistics` separates vector-equivalent products, block calls, and
algorithm stages.

| Field | Meaning |
|---|---|
| `directions_processed` | Total ordinary and diagnostic random directions processed. |
| `a_columns` | Number of vector-equivalent products with $A$. A block of width $b$ counts as $b$ columns. |
| `at_columns` | Number of vector-equivalent products with $A^\top$. |
| `a_block_calls`, `at_block_calls` | Actual calls to the block operator. |
| `ordinary_columns`, `ordinary_block_calls` | Forward work used by ordinary range construction. |
| `ordinary_observations` | Ordinary residual observations read by the stopping rule. |
| `ordinary_assimilated` | Ordinary directions added to the accepted range. |
| `ordinary_discarded` | Computed ordinary columns not included in the returned range. |
| `diagnostic_columns`, `diagnostic_block_calls` | Total diagnostic forward work. |
| `diagnostic_pilot_columns` | Pilot directions used to bound the largest residual eigenvalue. |
| `diagnostic_heldout_columns` | Fresh held-out directions used by the fixed-basis diagnostic. |
| `certificate_bound_round` | Ordinary trace prefix that gives `residual_bound_squared`. |

Timing fields are `total_seconds`, `a_seconds`, `at_seconds`,
`orthogonalization_seconds`, `certificate_seconds`, and `svd_seconds`.
`total_seconds` is the complete C++ algorithm time. The stage times can overlap
with work included in the total and should be read as a breakdown, not added to
reconstruct a different total.

### 3.6 PyTorch return tuple

The dense AC-RSVD entry returns

```text
(U, S, V, counters, diagnostics, timings, trace, stop_reason)
```

AC-RSVD-Fro appends `status` as the ninth item. The tensor layouts are:

| Item | Shape | Contents |
|---|---:|---|
| `U` | `(m, k)` | Left singular vectors. |
| `S` | `(k,)` | Singular values. |
| `V` | `(n, k)` | Right singular vectors. |
| `counters` | `(26,)` | Integer work and stopping data. |
| `diagnostics` | `(16,)` | Residual, truncation, and fixed-basis diagnostic values. |
| `timings` | `(6,)` | Total, $A$, $A^\top$, QR, certificate, and SVD times. |
| `trace` | `(rounds, 7)` | Ordinary observation, active dimension, leakage, and four predictable scales. |
| `stop_reason` | scalar integer | Stop-reason code from the table above. |
| `status` | scalar integer | Present only for exact-Fro calls. |

The `counters` indices are:

```text
 0 directions_processed       13 ordinary_observations
 1 A columns                  14 ordinary_assimilated
 2 A^T columns                15 ordinary_discarded
 3 A block calls              16 diagnostic_columns
 4 A^T block calls            17 diagnostic_block_calls
 5 range_rank                 18 diagnostic_pilot_columns
 6 boundary_only_rank         19 diagnostic_heldout_columns
 7 assimilated_directions     20 diagnostic trigger range rank
 8 validation_directions      21 held-out directions
 9 certificate_bound_round    22 held-out endpoints
10 certificate horizon        23 first diagnostic crossing
11 ordinary_columns           24 diagnostic bound prefix
12 ordinary_block_calls       25 residual-bound source
```

The `diagnostics` indices are:

```text
 0 residual_estimate_squared   8 diagnostic crossed
 1 residual_bound_squared      9 refinement target reached
 2 truncation_budget_squared  10 trigger mean ratio
 3 residual_decrease_squared  11 pilot mean
 4 direct_error_squared       12 spectral cap
 5 base truncation budget     13 diagnostic scale gamma
 6 diagnostic attempted      14 diagnostic bound sum
 7 held-out activated        15 first-crossing sum
```

The `timings` indices are:

```text
0 total   1 A products   2 A^T products   3 orthogonalization
4 certificate and inversion                5 terminal SVD
```

### 3.7 Complete PyTorch operator list

All registered functions are available below `torch.ops.ac_rsvd` after the
shared library is loaded.

| Function | Main inputs | Purpose and output |
|---|---|---|
| `run_ac_rsvd` | dense matrix, $\tau$, $\delta$, block and random options | Dense AC-RSVD; returns the eight-item factorization tuple |
| `run_ac_rsvd_fro` | dense matrix, $\tau$, $\delta$, exact $\|A\|_F^2$, block and random options | Dense AC-RSVD-Fro; returns the factorization tuple followed by `status` |
| `run_randqb_ei` | dense matrix, $\tau$, exact $\|A\|_F$, block and random options | Dense randQB-EI baseline |
| `run_randqb_mf_fro` | dense matrix, $\tau$, block and random options | Dense randQB-MF-Fro baseline |
| `run_ac_rsvd_hadamard` | singular-value vector, $\tau$, $\delta$, matrix seed, algorithm seed, and block options | AC-RSVD on the structured known-spectrum matrix used by the experiments |
| `run_ac_rsvd_fro_hadamard` | singular-value vector, $\tau$, $\delta$, matrix seed, algorithm seed, and block options | Exact-Fro variant on the same structured matrix; returns `status` |
| `run_randqb_ei_hadamard` | singular-value vector, $\tau$, matrix seed, algorithm seed, and block options | Structured-matrix randQB-EI run |
| `run_randqb_mf_fro_hadamard` | singular-value vector, $\tau$, matrix seed, algorithm seed, and block options | Structured-matrix randQB-MF-Fro run |
| `replay_certificate` | saved trace, input dimension, $\tau$, $\delta$, candidate residual, horizon | Recomputes the ordinary e-process value at a candidate squared residual |
| `invert_certificate` | saved trace, input dimension, $\tau$, $\delta$, horizon | Returns the continuous ordinary residual inverse from a saved trace |
| `replay_raw_gaussian_certificate` | held-out count and sum, dimension, spectral cap, scale, candidate residual | Recomputes a fixed-basis diagnostic value |
| `invert_raw_gaussian_certificate` | held-out data, spectral cap, scale, failure probability, upper endpoint | Inverts the fixed-basis diagnostic |
| `theory_bounds_e5_analytic` | singular values, matrix shape, tolerance, failure parameters, rank parameters, block size | Returns deterministic and high-probability direction, column, and block-call bounds |

The four dense factorization functions are the normal application interface.
The structured-Hadamard, replay, and theory-bound functions support experiment
reproduction and inspection of the mathematical certificates.

## 4. Build and deployment

### 4.1 Requirements

- CMake 3.24 or newer;
- a C++20 compiler;
- an LP64 BLAS implementation;
- LAPACK;
- OpenMP; and
- Python and a CPU PyTorch installation when building the Torch extension.

The CMake build first searches for standard BLAS and LAPACK packages and then
checks common OpenBLAS and LAPACK library locations. The public numeric type is
FP64 and the current backend uses 32-bit BLAS/LAPACK integers.

### 4.2 Linux dependencies

On Debian or Ubuntu, a typical core build uses:

```bash
sudo apt update
sudo apt install build-essential cmake libopenblas-dev liblapack-dev
```

For the PyTorch extension, create or activate the Python environment that owns
the desired CPU PyTorch installation before running CMake:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

### 4.3 Build the C++ core

```bash
git clone https://github.com/doloMing/ac_rsvd.git
cd ac_rsvd

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAC_RSVD_BUILD_TORCH=OFF \
  -DAC_RSVD_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF

cmake --build build --parallel
```

The main artifact is `build/libac_rsvd_core.a`.

For a CMake application, add this repository as a subdirectory:

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_ac_rsvd_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(AC_RSVD_BUILD_TORCH OFF CACHE BOOL "" FORCE)
set(AC_RSVD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

add_subdirectory(external/ac_rsvd)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ac_rsvd_core)
```

### 4.4 Build the PyTorch extension

Run CMake from the Python environment containing PyTorch:

```bash
cmake -S . -B build-torch \
  -DCMAKE_BUILD_TYPE=Release \
  -DAC_RSVD_BUILD_TORCH=ON \
  -DAC_RSVD_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DPython3_EXECUTABLE="$(command -v python)"

cmake --build build-torch --parallel
```

The extension is `build-torch/ac_rsvd_torch.so`. It is a registered Torch
operator library and is loaded with `torch.ops.load_library`; it is not
imported as an independent Python implementation.

```python
from pathlib import Path
import torch

torch.ops.load_library(str(Path("build-torch/ac_rsvd_torch.so").resolve()))
print(torch.ops.ac_rsvd.run_ac_rsvd)
```

### 4.5 Thread control

AC-RSVD uses OpenMP and the selected BLAS library. Set thread counts before
starting the process when reproducible performance is important:

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

Avoid running several outer workers while each worker also uses all BLAS and
OpenMP threads. Select either parallel runs with small per-run thread counts or
one run with the full thread budget.

### 4.6 Numerical model

The mathematical probability and rank results are exact-arithmetic statements.
The released CPU implementation uses FP64 BLAS/LAPACK, fixed-order QR,
log-domain certificate updates, and continuous scalar inversion. As with other
FP64 matrix factorizations, input scaling and the numerical behavior of the
operator affect the computed result.

## 5. C++ and PyTorch examples

### 5.1 Matrix-free C++ example

This example factors a diagonal operator without storing a dense matrix. The
input and output blocks use column-major storage.

```cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "ac_rsvd/ac_rsvd.hpp"

class DiagonalOperator final : public ac_rsvd::MatrixOperator {
public:
    explicit DiagonalOperator(std::vector<double> diagonal)
        : diagonal_(std::move(diagonal)) {}

    int rows() const override {
        return static_cast<int>(diagonal_.size());
    }

    int cols() const override {
        return static_cast<int>(diagonal_.size());
    }

    void apply(
        const double* x,
        int block_cols,
        double* y) const override {
        const int n = cols();
        for (int column = 0; column < block_cols; ++column) {
            for (int row = 0; row < n; ++row) {
                y[row + column * n] =
                    diagonal_[row] * x[row + column * n];
            }
        }
    }

    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override {
        apply(y, block_cols, x);
    }

private:
    std::vector<double> diagonal_;
};

int main() {
    const int n = 256;
    std::vector<double> diagonal(n);
    for (int i = 0; i < n; ++i) {
        diagonal[i] = std::exp(-0.04 * i);
    }

    DiagonalOperator matrix(std::move(diagonal));

    ac_rsvd::AcRsvdOptions options;
    options.tolerance = 1e-2;
    options.failure_probability = 0.01;
    options.block_size = 16;
    options.seed = 2026;
    options.use_enhanced_mode = true;

    ac_rsvd::FactorizationResult result =
        ac_rsvd::compute_ac_rsvd(matrix, options);

    if (result.status != ac_rsvd::FactorizationStatus::success) {
        std::cerr << "The sampled range was not accepted.\n";
        return 1;
    }

    std::cout << "output rank: " << result.rank << '\n';
    std::cout << "sampled range rank: " << result.range_rank << '\n';
    std::cout << "squared residual bound: "
              << result.residual_bound_squared << '\n';
    std::cout << "A block calls: "
              << result.statistics.a_block_calls << '\n';
    std::cout << "A^T block calls: "
              << result.statistics.at_block_calls << '\n';
}
```

With the core already built, save the example as `main.cpp` and compile it
through a small CMake project using the `add_subdirectory` configuration in
[Section 4.3](#43-build-the-c-core).

### 5.2 Dense PyTorch example

The dense binding requires a two-dimensional CPU tensor with `torch.float64`
dtype.

```python
from pathlib import Path
import torch

torch.ops.load_library(str(Path("build-torch/ac_rsvd_torch.so").resolve()))

torch.manual_seed(7)
left = torch.randn(600, 50, dtype=torch.float64)
right = torch.randn(50, 400, dtype=torch.float64)
A = left @ right

relative_tolerance = 1e-6
A_fro = torch.linalg.matrix_norm(A, ord="fro").item()
tau = relative_tolerance * A_fro
delta = 0.01

result = torch.ops.ac_rsvd.run_ac_rsvd(
    A,
    tau,
    delta,
    16,       # block_size
    2026,     # seed
    0,        # stream
    False,    # sequential_orthogonalization
    True,     # enhanced_mode
    512,      # diagnostic_test_size
)

U, S, V, counters, diagnostics, timings, trace, stop_reason = result
A_hat = (U * S) @ V.T
absolute_error = torch.linalg.matrix_norm(A - A_hat, ord="fro").item()

print(f"rank              = {S.numel()}")
print(f"range rank        = {int(counters[5])}")
print(f"absolute error    = {absolute_error:.6e}")
print(f"requested tau     = {tau:.6e}")
print(f"residual bound^2  = {float(diagnostics[1]):.6e}")
print(f"total time        = {float(timings[0]):.6f} s")
print(f"stop reason       = {stop_reason}")
```

`A_fro` is used above only to convert a relative request into the absolute
input `tau`. The default AC-RSVD call does not receive `A_fro`.

### 5.3 AC-RSVD-Fro in PyTorch

Use the exact-Fro entry when the squared norm is available as input data:

```python
frobenius_norm_squared = torch.sum(A * A).item()

result_fro = torch.ops.ac_rsvd.run_ac_rsvd_fro(
    A,
    tau,
    delta,
    frobenius_norm_squared,
    16,
    2026,
    0,
    False,
    True,
    512,
)

U_f, S_f, V_f, counters_f, diagnostics_f, timings_f, trace_f, stop_f, status = (
    result_fro
)

if status == 0:
    A_hat_fro = (U_f * S_f) @ V_f.T
    error_fro = torch.linalg.matrix_norm(A - A_hat_fro, ord="fro").item()
    print(f"AC-RSVD-Fro rank  = {S_f.numel()}")
    print(f"absolute error    = {error_fro:.6e}")
else:
    print("The fixed sampled range is insufficient under the exact norm.")
```

With identical `seed`, `stream`, and range-construction options, the AC-RSVD
and AC-RSVD-Fro calls use the same random directions and matrix products. Their
final ranks can differ because the second call replaces the certified upper
bound by the exact projection residual.

### 5.4 Baseline calls in PyTorch

```python
ei = torch.ops.ac_rsvd.run_randqb_ei(
    A, tau, A_fro, 16, 2026, 0
)

mf = torch.ops.ac_rsvd.run_randqb_mf_fro(
    A, tau, 16, 2026, 0
)

U_ei, S_ei, V_ei = ei[:3]
U_mf, S_mf, V_mf = mf[:3]
```

The EI argument is $\|A\|_F$, not its square. The two baseline calls return
the same eight-item tuple shape as `run_ac_rsvd`.

### 5.5 Applying the factors to vectors or blocks

For a vector `x` or a block `X`, use:

```python
y = U @ (S * (V.T @ x))
Y = U @ (S[:, None] * (V.T @ X))
```

There is no need to construct `torch.diag(S)`.

## 6. Algorithm structure and pseudocode

The source tree has three algorithm cores: AC-RSVD, randQB-EI, and
randQB-MF-Fro. AC-RSVD-Fro uses the AC-RSVD range construction and replaces
only its terminal rank rule.

### 6.1 AC-RSVD

```text
Input: matrix products with A and A^T, tolerance tau,
       failure probability delta, and block parameters
Output: U_k, s_k, V_k with an adaptively selected rank k

1. Set Q to an empty orthonormal basis.
2. Draw a Gaussian block G and compute Y = A G.
3. Project Y against Q and compute fixed-order unpivoted block QR.
4. Read the QR columns in their original order.
5. Before accepting column t, form the scalar residual observation
       X_t = n ||(I - Q Q^T) A g_t||_2^2 / ||g_t||_2^2.
6. Update the ordinary e-process at the candidate c = tau^2.
7. If the process has not crossed its threshold:
       accept the nonzero direction into Q and continue.
8. When recent observations approach the tolerance, the enhanced mode may:
       hold Q fixed;
       use an independent pilot block to bound the residual spectrum;
       use fresh held-out vectors to test the fixed residual;
       return to ordinary range growth if this diagnostic does not certify Q.
9. When either stopping path certifies Q, set Q_star = Q and record a
   simultaneous upper bound U_res on ||(I - Q_star Q_star^T) A||_F^2.
10. Form B_star = Q_star^T A with one block application of A^T.
11. Compute the SVD of B_star.
12. Choose the smallest k such that
       sum_{i > k} sigma_i(B_star)^2 <= tau^2 - U_res.
13. Return the leading k factors of Q_star B_star.
```

#### AC-RSVD-Fro terminal variant

```text
Input: all AC-RSVD inputs and the exact scalar f_A = ||A||_F^2
Output: a successful compact SVD or certificate_miss

1. Run the same range construction and stopping calculation as AC-RSVD.
2. Form the same terminal matrix B_star = Q_star^T A.
3. Compute the exact projection residual
       mu_F = f_A - ||B_star||_F^2.
4. If mu_F > tau^2:
       return certificate_miss and keep the trace, counters, and diagnostics.
5. Otherwise choose the smallest k such that
       sum_{i > k} sigma_i(B_star)^2 <= tau^2 - mu_F.
6. Return the leading k factors.
```

### 6.2 randQB-EI baseline

```text
Input: A, A^T, tau, exact ||A||_F, block width b
Output: a blocked QB-based compact SVD

1. Initialize Q and B as empty and E = ||A||_F^2.
2. Draw a Gaussian block Omega.
3. Form the residual sample (A - Q B) Omega.
4. Orthogonalize the sample twice to obtain a new basis block Q_b.
5. Form B_b = Q_b^T A with an application of A^T.
6. Read the rows of B_b in order and subtract ||B_b(i,:)||_2^2 from E.
7. Keep the shortest block prefix for which E < tau^2.
8. If the tolerance has not been reached, append the full block and repeat.
9. Compute the compact SVD of Q B and return its factors.
```

### 6.3 randQB-MF-Fro baseline

```text
Input: A, A^T, tau, block width b
Output: a matrix-free adaptive QB factorization

1. Initialize Q and B as empty.
2. Draw a Gaussian block Omega with entry variance 1/b.
3. Form Y = (A - Q B) Omega.
4. Use ||Y||_F^2 as the current squared residual estimate.
5. If the estimate is no larger than tau^2:
       remove low-energy directions from the most recently accepted block
       while the adjusted estimate remains below tau^2;
       compute and return the compact SVD.
6. Otherwise orthogonalize Y to form Q_b.
7. Form B_b = Q_b^T A, append Q_b and B_b, and repeat.
```

## 7. Mathematical principles

### 7.1 Fixed-tolerance low-rank approximation

For each $\varepsilon\geq0$, define the tolerance rank

$$
r_\star(\varepsilon)=
\min\{k:\sum_{i>k}\sigma_i(A)^2\leq\varepsilon^2\}.
$$

The Frobenius Eckart–Young theorem shows that no rank below
$r_\star(\tau)$ can meet error $\tau$. AC-RSVD does not know this rank in
advance. It learns a sufficient range from matrix products and then selects an
output rank from the projected singular values.

### 7.2 One residual observation

Let $Q_t$ be the accepted orthonormal basis before direction $g_t$, and
define

$$
E_t=(I-Q_tQ_t^\top)A,
\qquad
\mu_t=\|E_t\|_F^2.
$$

For an independent Gaussian vector $g_t$, enhanced AC-RSVD observes

$$
X_t=n\frac{\|E_tg_t\|_2^2}{\|g_t\|_2^2}.
$$

Conditional on the complete past of the calculation,
$g_t/\|g_t\|_2$ is uniform on the unit sphere and

$$
\mathbb{E}[X_t\mid Q_t]=\mu_t.
$$

The observation is computed before its direction is added to $Q_t$. If the
algorithm continues, the new basis reduces the residual and changes the
distribution of the next observation. A fixed-round estimator alone does not
control the first successful round selected from this sequence.

### 7.3 The ordinary e-process

Let $B_{n,1}$ have the beta distribution with parameters
$1/2$ and $(n-1)/2$, and define

$$
\Phi_{n,1}(s)=\mathbb{E}\exp(s n B_{n,1}).
$$

For a proposed squared residual $c>0$ and a scale $\gamma\geq0$, one
update factor is

$$
\ell_t(c;\gamma)=
\frac{\exp(-\gamma X_t/c)}{\Phi_{n,1}(-\gamma)}.
$$

Whenever the current residual satisfies $\mu_t\geq c$, the conditional mean
of this factor is at most one. AC-RSVD combines these factors over several
fixed scales, start times, and scales selected from earlier observations. The
resulting nonnegative process $W_t(c)$ obeys the supermartingale condition
needed for Ville's inequality.

The online calculation tests $c=\tau^2$. When

$$
W_t(\tau^2)\geq\frac{1}{\delta_{\mathrm{glob}}},
$$

the same stored trace is inverted to obtain

$$
U_t=\inf\{c>0:
W_t(c)\geq\frac{1}{\delta_{\mathrm{glob}}}\}.
$$

This gives a residual upper bound valid at the stopping round selected by the
observations. Continuous inversion changes no matrix products and usually
leaves more error allowance for the final rank truncation than the boundary
value $\tau^2$.

### 7.4 Fixed-basis diagnostic

The enhanced mode can test the current basis without adding diagnostic vectors
to it. With $Q_b$ fixed, define

$$
M_b=A^\top(I-Q_bQ_b^\top)A.
$$

An independent pilot block first gives an upper bound $L$ on
$\lambda_{\max}(M_b)$. Fresh held-out Gaussian vectors then provide a
time-uniform upper bound on $\mathrm{tr}(M_b)=\|(I-Q_bQ_b^\top)A\|_F^2$.
If the diagnostic succeeds, it returns the fixed basis and its residual bound.
If it does not succeed, ordinary range construction resumes with the same
basis and ordinary trace.

The implementation uses an empirical allocation of the requested failure
probability among the ordinary process, pilot spectral cap, and held-out
diagnostic. The proof combines their failure events with the standard union
bound.

### 7.5 Blocked execution preserves ordered decisions

The implementation computes $AG$ for several Gaussian vectors at once, but
the stopping rule reads the resulting QR columns in their original order. For
an unpivoted fixed-order QR factorization, every accepted prefix spans the same
space as sequential orthogonalization of that prefix. Therefore the algorithm
obtains BLAS-3 block performance without replacing the ordered stopping rule by
a block-level approximation.

### 7.6 Projection and truncation are both included

After the final basis $Q_\star$ is fixed, AC-RSVD forms

$$
B_\star=Q_\star^\top A.
$$

For any approximation $C$ to $B_\star$, orthogonality gives the exact
identity

$$
\|A-Q_\star C\|_F^2
=\|(I-Q_\star Q_\star^\top)A\|_F^2
+\|B_\star-C\|_F^2.
$$

If $U_{\rm res}$ is the certified upper bound on the first term, the algorithm chooses
the smallest $k$ satisfying

$$
\sum_{i>k}\sigma_i(B_\star)^2\leq\tau^2-U_{\rm res}.
$$

Thus the certificate applies to the final truncated factorization, not only to
the sampled range.

For AC-RSVD-Fro, orthogonality also gives

$$
\mu_F=\|(I-Q_\star Q_\star^\top)A\|_F^2
=\|A\|_F^2-\|B_\star\|_F^2.
$$

Replacing $U_{\rm res}$ by $\mu_F$ can only enlarge the valid truncation allowance
when the AC-RSVD certificate covers the residual. This explains why the
exact-Fro variant can return a smaller factor with the same sampled range and
the same matrix products.

## 8. Why AC-RSVD is faster and gives reliable accuracy

### Reliable fixed-tolerance accuracy

AC-RSVD addresses the complete error of the returned factorization:

1. the ordinary e-process remains valid over all candidate stopping rounds;
2. the diagnostic uses independent data while the tested basis is fixed;
3. the standard union bound combines the stopping paths within $\delta$; and
4. the projection–truncation identity includes the singular values discarded
   from the projected matrix.

The guarantee therefore follows the same adaptive decisions made by the
algorithm. It is not obtained by applying a fixed-round residual estimate after
the stopping round has already been selected.

### Lower output rank

Range construction and output-rank selection solve different tasks. The range
must capture enough of $A$ to meet the tolerance, but every captured
direction does not need to appear in the returned factorization. AC-RSVD
computes the projected SVD after the range is fixed and spends the remaining
error allowance on truncation. A smaller $k$ directly reduces factor storage
and the cost of later products with the approximation.

AC-RSVD-Fro sharpens this step by replacing the residual upper bound with the
exact projection residual. It does not resample the range and does not add a
product with $A$ or $A^\top$.

### Faster execution

Three implementation choices account for the main speed gain:

- Gaussian directions are applied in blocks and orthogonalized with
  fixed-order blocked QR;
- after the range is fixed, $A^\top Q_\star$ is formed in one block call
  instead of forming a new adjoint panel after every accepted range block; and
- the final truncation returns fewer columns, reducing the terminal factor work
  and every later use of the approximation.

The paper reports the following setup-inclusive results:

| Comparison | Reported result |
|---|---:|
| AC-RSVD output rank relative to randQB-EI | about **5% smaller** at the median |
| AC-RSVD excess rank relative to randQB-EI | about **8% lower** at the median |
| AC-RSVD time relative to randQB-EI | about **23% faster** at the median |
| AC-RSVD time relative to randQB-MF-Fro | about **18% faster** at the median |
| AC-RSVD-Fro output rank relative to AC-RSVD | about **35% smaller** with the same matrix products |
| AC-RSVD-Fro time relative to AC-RSVD | about **9% faster** at the median |

In the reported accuracy comparison, AC-RSVD and randQB-EI have a zero
tolerance-violation rate, while randQB-MF-Fro exceeds the tolerance in about
46% of its runs. Blocked AC-RSVD preserves the stopping rounds and output ranks
of column-by-column processing while substantially reducing
orthogonalization and total time.

## 9. Paper, citation, author, and license

### Paper

**Anytime-Certified Randomized SVD**<br>
Yang Tian<br>
Manuscript submitted to the *SIAM Journal on Scientific Computing*.


If AC-RSVD supports your research, please cite the paper:

```bibtex
@unpublished{Tian2026ACRSVD,
  author = {Tian, Yang},
  title  = {Anytime-Certified Randomized SVD},
  note   = {Manuscript submitted to the SIAM Journal on Scientific Computing},
  year   = {2026}
}
```

The software can be cited separately:

```bibtex
@misc{Tian2026ACRSVDSoftware,
  author       = {Tian, Yang},
  title        = {{AC-RSVD}: Anytime-Certified Randomized {SVD}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/doloMing/ac_rsvd}
}
```

If the algorithms or implementation are useful to you, please **star this
repository** and cite the paper. A GitHub star helps other researchers find the
project, and a citation records its use in scientific work.

### Author

**Yang Tian**<br>
Infplane Computing Technologies Ltd<br>
[tyanyang04@gmail.com](mailto:tyanyang04@gmail.com) &
[yang.tian@infplane.com](mailto:yang.tian@infplane.com)
