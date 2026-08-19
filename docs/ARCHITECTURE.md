# Library Architecture

## 1. Design principles

The implementation will follow five constraints:

1. There is one algorithm core, written in C++20. The CLI, PyTorch binding, and
   experiments are adapters, not independent implementations.
2. The matrix-free operator boundary is smaller than the algorithm boundary.
   Operator implementations know how to apply `A` and `A^T`; they do not own
   stopping, QR, truncation, or accounting policy.
3. Every randomized and numerical decision is observable through diagnostics.
4. Exact-model logic and finite-precision policy are separate components.
5. Public headers expose stable data contracts, while BLAS/LAPACK, PyTorch, and
   special-function details remain behind internal adapters.

## 2. Planned repository layout

The following tree is a design target, not a directory scaffold to create all
at once:

```text
ac_rsvd/
  CMakeLists.txt
  pyproject.toml
  cmake/
  include/ac_rsvd/
    operator.hpp
    options.hpp
    result.hpp
    diagnostics.hpp
    ac_rsvd.hpp
  src/
    core/
    certificate/
    linalg/
    operators/
    analysis/
    baselines/
    cli/
  bindings/torch/
  python/ac_rsvd/
  tests/cpp/
  tests/python/
  experiments/
    configs/
    runners/
    analysis/
    plots/
  docs/
  CITATION.cff
  CONTRIBUTING.md
  LICENSE
  README.md
```

Paper manuscripts and generated PDFs remain outside the source repository.
The repository will contain citations and implementation-facing specifications,
not duplicate manuscript working files.

## 3. Public C++ boundary

### Matrix operator

The core operator abstraction supplies:

- row and column dimensions;
- maximum supported forward block width;
- `apply_A` for column-major FP64 blocks;
- `apply_AT` for column-major FP64 blocks;
- an optional exact Frobenius norm and optional known singular values for tests
  and analysis, never for the AC-RSVD stopping decision.

The initial dense view type will be a small non-owning column-major matrix view
with pointer, row count, column count, and leading dimension. Owning workspaces
are internal. This avoids forcing Eigen or PyTorch types into the public C++
API and maps directly to BLAS/LAPACK.

### Options

Options are grouped rather than flattened:

- problem: tolerance and failure probability;
- blocking: maximum block width and workspace policy;
- randomness: master seed, stream identifier, and replay mode;
- certificate: default or explicit scales and weights;
- numerics: orthogonality thresholds, retry limits, inversion tolerances, and
  enclosure policy;
- diagnostics: event trace level and optional snapshot recording.

All defaults are serialized into run metadata. An experiment result must remain
interpretable without consulting the source version that created it.

### Result

The result contains compact factors, output rank, terminal reason, residual
upper bound, truncation budget, product counters, timings, orthogonality
metrics, numerical retries, certificate status, seed metadata, and a structured
status code. Invalid input, nonfinite operator output, QR/SVD failure, inversion
failure, and exhausted retries are distinct statuses.

## 4. Internal components

### Algorithm state machine

One state machine owns block generation, ordered column revelation, certificate
updates, terminal conditions, final-block augmentation, the terminal adjoint
call, and final truncation. It is the only component allowed to decide whether
a direction is processed or a basis column is retained.

### Linear algebra layer

A thin internal layer wraps column-major BLAS and LAPACK operations. Required
operations include projections, GEMM, fixed-order Householder QR, explicit-Q
formation, small-core SVD/RRQR, and final compact SVD. The wrappers normalize
integer sizes, workspace queries, error reporting, and backend differences.

The first supported backend is a system BLAS/LAPACK installation, normally
OpenBLAS on Linux. OpenMP is optional and algorithm code must not assume a
particular BLAS thread runtime.

### Randomness

Randomness is a named service rather than direct calls to standard-library
distributions. It records a master seed and independent stream/counter IDs.
Paired comparison methods receive replayable common random panels. The chosen
normal generator must be specified and tested; `std::normal_distribution` is
not sufficient for cross-standard-library reproducibility.

### Certificate engine

The certificate engine owns start weights, scale mixtures, log-domain component
updates, replay, continuous inversion, and evaluation of `log Phi(d, s)`. It
does not know about QR or matrix storage. A high-precision oracle used in tests
is separate from the production evaluator.

### Numerical policy

Numerical policy supplies two-pass projections, diagonal thresholds, leakage
bounds, lower/upper scalar enclosures, retry rules, and conservative final-tail
tests. This policy emits audit events for every conservative substitution.

### Operators

Initial operators are:

- a non-owning dense matrix operator for correctness tests and small examples;
- a known-spectrum structured Hadamard operator for the paper experiments;
- a counting/timing decorator used around every operator;
- a fault-injection operator for nonfinite and error-path tests.

Generic user operators enter through the C++ interface. The Python binding may
later provide a callback adapter, but built-in dense and structured operators
remain the performance and experiment paths.

### Analysis and baselines

The theorem-bound evaluator consumes singular values and fixed analysis
parameters; it is not linked into the stopping rule. Comparison algorithms are
isolated under `baselines` and use the same operator, workspace, RNG, counters,
and timing conventions. Their outputs identify the source paper and all tuning
parameters.

## 5. C++/Python boundary

The PyTorch extension is a coarse-grained adapter: one call runs one complete
C++ algorithm configuration. Hadamard transforms, QR, certificate updates,
data-dependent stopping, inversion, and final SVD remain inside C++.

Planned binding paths are:

- a dense CPU tensor convenience operation;
- construction and execution of the built-in structured-spectrum operator;
- conversion of result factors and diagnostics to tensors and Python objects;
- an optional Python callback operator for usability, documented as slower and
  excluded from benchmark claims.

The Python package performs configuration generation, multiprocessing, result
aggregation, confidence intervals, and plotting. It must not contain a second
AC-RSVD implementation.

## 6. Build and dependencies

Planned required dependencies are CMake, a C++20 compiler, BLAS, and LAPACK.
OpenMP, PyTorch, and a rigorous high-precision/enclosure backend are optional
build features. Python packaging will use `pyproject.toml` with
`scikit-build-core`, so the CLI-only C++ build does not require PyTorch.

Dependency versions and discovery rules will be fixed during the scaffold
milestone. No network fetch is allowed in a default reproducible build; CI may
exercise an explicitly enabled dependency bootstrap path.

## 7. Threading and timing

The core is reentrant: one run owns all mutable state. It does not mutate global
BLAS or OpenMP settings. Experiment workers set thread counts before entering
the core.

Internal timing starts after validated configuration and workspace allocation,
and ends after factors and counters are complete. It separately records forward
operator work, adjoint work, orthogonalization, certificate work, inversion,
and final SVD. Python dispatch, DataFrame creation, and plotting are excluded.
