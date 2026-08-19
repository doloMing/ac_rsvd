# Implementation and Verification Plan

No phase advances merely because its code compiles. Each phase has an explicit
acceptance gate tied to the paper's invariants or to reproducible software
behavior.

## Phase 0: freeze the specification

Deliverables:

- review the algorithm and architecture documents against the manuscript;
- resolve every open decision listed at the end of this document;
- assign equation-to-test identifiers for the stopping recurrence, final-block
  augmentation, truncation budget, and product counts;
- choose supported compiler, BLAS/LAPACK, PyTorch, and Python version ranges.

Acceptance gate: the exact state transition for every terminal route can be
described without implementation-specific ambiguity.

## Phase 1: minimal build and data contracts

Deliverables:

- CMake project with a core library, CLI option, test option, and optional
  PyTorch option;
- public operator, option, result, diagnostic, status, and matrix-view types;
- BLAS/LAPACK discovery and a backend information report;
- deterministic RNG service and seed/stream metadata;
- dense, counting, and fault-injection operators.

Tests:

- matrix-view stride and alias tests;
- exact operator counter tests for variable block widths;
- RNG replay and stream-separation tests;
- invalid-dimension, nonfinite, and backend-error tests.

Acceptance gate: an operator block can pass through the library and return
correct data and counters without any AC-RSVD logic.

## Phase 2: linear algebra and ordered block invariants

Deliverables:

- two-pass block projection;
- fixed-order unpivoted Householder QR with a documented diagonal convention;
- small trailing-core SVD/RRQR;
- orthogonality metrics, thresholds, refactor, and retry hooks.

Tests:

- blocked QR prefix projectors against a scalar sequential oracle;
- every prefix residual norm and diagonal against the sequential oracle;
- rank-deficient and nearly dependent panels;
- preservation of column order and sign convention;
- reconstruction of the final-block projected span from the trailing core;
- sanitizer builds for workspace and leading-dimension errors.

Acceptance gate: for deterministic small matrices, every block prefix produces
the same mathematical state as sequential processing within stated tolerances.

## Phase 3: certificate engine

Deliverables:

- start-time and scale mixtures;
- stable `log Phi(d, s)` evaluation;
- log-domain online recurrence at `tau^2`;
- full scalar trace recording and replay at arbitrary positive `c`;
- monotone bracketing and continuous inversion;
- optional high-precision oracle and lower/upper enclosure interface.

Tests:

- `Phi` against numerical quadrature/high-precision values over all required
  dimensions and arguments;
- mixture recurrence against a direct product-sum definition;
- monotonicity in `c` and agreement between online state and replay;
- inversion endpoints around known roots and near-zero bounds;
- overflow, underflow, final-direction, and `n = 1` cases;
- fixed-seed Monte Carlo checks of one-step moment inequalities, reported as
  diagnostics rather than used as substitutes for deterministic tests.

Acceptance gate: every certificate scalar in a saved trace can be independently
recomputed and the same crossing round and inverse interval recovered.

## Phase 4: exact-model AC-RSVD state machine

Deliverables:

- block generation and ordered column revelation;
- all four terminal routes;
- final-block suffix utilization;
- single terminal adjoint call;
- compact-core SVD and smallest valid truncation rank;
- full diagnostic trace and exact product accounting.

Tests:

- dense matrices covering zero, rank-zero, rank-one, square, rectangular,
  rank-deficient, repeated-singular-value, smooth-tail, and clustered-tail cases;
- exact projection/truncation error identity;
- deterministic caps on directions and block-rounded products;
- output-rank lower and upper bounds when singular values are known;
- proof that no `A` call occurs after the final range is fixed;
- proof that a nonempty result uses exactly one `A^T` block call;
- paired block-width-one and block-width-many paths with the same direction
  sequence and sequential stopping round.

Acceptance gate: all deterministic statements in Theorem 1 are executable as
tests, and small known-spectrum runs meet the end-to-end error accounting.

## Phase 5: finite-precision safeguards

Deliverables:

- input/output diagonal thresholds and redraw/refactor policies;
- directional-energy and used-subspace leakage bounds;
- lower and upper log-certificate enclosures;
- certified crossing and inverse interval policy;
- conservative residual-decrease and SVD-tail tests;
- retry-at-tighter-accuracy and explicit failure behavior.

Tests:

- adversarial nearly dependent blocks;
- very small and very large tolerances;
- clustered singular values at the truncation boundary;
- injected special-function, QR, product, and SVD perturbations;
- comparisons of every enclosure against higher-precision calculations;
- negative truncation budget prevention and retry exhaustion.

Acceptance gate: no successful run can result from an uncertified comparison
when strict enclosure mode is enabled. The documentation states precisely what
the normal FP64 mode does and does not certify.

## Phase 6: bindings and CLI

Deliverables:

- coarse-grained PyTorch CPU extension over the C++ core;
- Python configuration/result types without algorithm duplication;
- CLI accepting a versioned JSON configuration;
- machine-readable JSON result and trace output;
- build metadata and runtime environment capture.

Tests:

- C++/CLI/Python result equivalence for the same configuration and seed;
- tensor dtype, device, shape, stride, and ownership validation;
- exception translation and subprocess exit statuses;
- import without PyTorch for the C++/CLI-only build.

Acceptance gate: the same C++ path produces the same counters and factors when
invoked through each supported front end.

## Phase 7: comparison methods and theorem bounds

Deliverables:

- independently implemented `randQB_MF_Fro` and `randQB_EI` baselines from
  their published descriptions;
- shared operator, counter, RNG replay, thread, and timing infrastructure;
- full-spectrum Theorem 2 bound evaluator;
- tolerance-rank Corollary 2.1 bound evaluator;
- documented analysis parameters and finite minimizations.

Acceptance gate: each baseline passes its own reconstruction tests, and every
reported work comparison uses identical counter semantics and explicitly stored
tuning parameters.

## Phase 8: paper experiments and release hardening

Deliverables:

- immutable experiment manifests and seed registry;
- small-matrix reliability, main comparison, and paired ablation runners;
- Clopper--Pearson intervals, tables, and deterministic plotting scripts;
- CI, sanitizer jobs, release packaging, API documentation, `CITATION.cff`, and
  contribution guidance.

Acceptance gate: a clean checkout can reproduce all tables and figures from
saved manifests, while generated data and build artifacts remain outside Git.

## Testing policy

Tests are divided into four layers:

1. deterministic unit tests for algebra, recurrences, counters, and errors;
2. property/differential tests against scalar and high-precision oracles;
3. deterministic integration tests on small known-spectrum matrices;
4. statistical validation suites with fixed seeds and confidence intervals.

Ordinary CI runs layers 1--3 and a small statistical smoke test. Large
reliability experiments are scheduled or release tests, not flaky per-commit
gates. Test data records the RNG algorithm, seed, compiler, BLAS, CPU thread
count, and library version.

## Review rules

- Algorithm changes require an update to the mathematical contract and a test
  tied to the affected invariant.
- Counter or timing changes require a schema-version update if saved results
  change meaning.
- Baseline-specific shortcuts cannot enter the AC-RSVD core.
- Python may orchestrate or analyze runs but may not decide AC-RSVD state.
- Performance work follows correctness profiling; it may change kernels but not
  ordered decisions.

## Decisions to close before Phase 1

1. Production `log Phi` backend: Boost.Math, an internal approximation, or an
   optional MPFR-backed enclosure path.
2. Reproducible Gaussian generator and its cross-platform compatibility target.
3. Required BLAS integer ABI support, including whether ILP64 is in version 0.1.
4. Exact minimum supported PyTorch and Python versions.
5. Whether a generic Python callback operator belongs in version 0.1 or a later
   usability release.
6. Normal FP64 mode versus strict enclosure mode naming and result flags.
7. Exact formulas for the smooth and clustered spectra and all fixed experiment
   values of `tau`, `delta`, `rho`, `eta_r`, `eta_c`, and `alpha`.
8. Baseline parameter matching rules, especially stopping tolerances and final
   rank reduction.
