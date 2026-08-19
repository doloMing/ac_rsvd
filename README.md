# AC-RSVD

AC-RSVD is a planned C++20 library for anytime-certified randomized singular
value decomposition of matrix-free operators. Given an absolute Frobenius
tolerance and a failure probability, it will adaptively select the output rank
and return a compact SVD.

The repository is currently in the design phase. No implementation has been
started. The design intentionally separates the mathematical contract, the
numerical realization, language bindings, and reproducible experiments before
the first source file is written.

## Planned scope

The first release will provide:

- one C++20 algorithm core for AC-RSVD;
- a matrix-free interface based on block applications of `A` and `A^T`;
- an FP64 CPU implementation using BLAS/LAPACK, with optional OpenMP;
- ordered blocked QR with sequentially equivalent stopping decisions;
- log-domain certificate updates and continuous post-stop inversion;
- use of all useful products from the block in which stopping occurs;
- one terminal block application of `A^T` after the final range is fixed;
- a CLI and a thin PyTorch/Python binding over the same C++ core;
- deterministic diagnostics, product counters, and experiment manifests;
- reference implementations of the comparison methods used in the paper;
- tests for the mathematical invariants, numerical safeguards, and final error.

The initial release will support absolute Frobenius tolerance only. Relative
tolerance, spectral-norm stopping, GPU kernels, distributed execution, and
power iterations are outside the first-release scope.

## Design documents

- [Mathematical and numerical contract](docs/ALGORITHM_CONTRACT.md)
- [Library architecture](docs/ARCHITECTURE.md)
- [Implementation and verification plan](docs/IMPLEMENTATION_PLAN.md)
- [Experiment and reproducibility plan](docs/EXPERIMENTS.md)

## Status

The next milestone is to review and freeze these design documents. Source-code
scaffolding begins only after the interfaces, numerical policies, dependency
choices, and acceptance tests have been agreed upon.

## License

AC-RSVD is licensed under the [MIT License](LICENSE).
