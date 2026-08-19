# Experiment and Reproducibility Plan

## 1. Questions

The numerical suite answers only four implementation-facing questions:

1. Is the empirical failure rate of the final compact SVD compatible with the
   user-specified failure level after adaptive repeated checking?
2. How do final error, rank, operator work, block calls, and CPU time compare
   with `randQB_MF_Fro` and `randQB_EI` under one absolute Frobenius task?
3. How conservative are Theorem 2, Corollary 2.1, and the deterministic
   dimension cap relative to actual processed directions?
4. What changes are caused by true block orthogonalization, continuous
   inversion, and one terminal adjoint block call?

## 2. Execution model

All algorithms execute in one C++20 core. PyTorch/Python assigns configurations
and seeds, starts coarse-grained runs, collects structured results, computes
confidence intervals, and plots. It does not execute Hadamard transforms,
certificate steps, stopping decisions, continuous inversion, or matrix algebra.

The production timing interval begins after configuration validation and
workspace initialization and ends after factors and counters are finalized.
Every result also records phase timings and environmental metadata.

## 3. Operators and spectra

The main operator has the known-spectrum form `A = U Sigma V^T`, where `U` and
`V` are compositions of signs, permutations, and fast Hadamard transforms. It
stores only singular values and transform metadata. This supplies exact tail
sums and tolerance ranks while retaining dense singular vectors.

Two spectrum families are required:

- a smooth polynomial decay;
- a spectrum clustered around the target tolerance boundary.

Their formulas, normalization, dimensions, tolerance values, and target ranks
must be written into versioned JSON manifests before implementation timing is
accepted. Main comparisons use `n = 2^17`, target tolerance ranks 64 and 192,
FP64, block width 32, one BLAS/LAPACK backend, and ten CPU threads.

## 4. Experiment groups

### Reliability

Use two fixed `256 x 256` matrices. For `delta = 0.1`, run 500 seeds per
matrix. For `delta = 0.01`, run 2,000 seeds per matrix. The total is 5,000
AC-RSVD runs. Eight worker processes each use a single-threaded C++ core.

Failure means only that the directly evaluated final compact-SVD Frobenius
error exceeds `tau`. Report failure counts, empirical rates, and one-sided 95%
Clopper--Pearson upper bounds separately for both spectra.

### Main comparison

Use two spectra, two target ranks, five paired seeds, and three algorithms for
60 total runs. Report all five paired values and their median. Each seed is
timed once after a separate warmup protocol.

Required outputs are error divided by tolerance, output rank divided by optimal
tolerance rank, forward and adjoint vector-product counts, forward and adjoint
block-call counts, computed columns absent from the final range, total time, and
phase times.

### Bounds

Reuse the 20 AC-RSVD paths from the main comparison. Compute the full-spectrum
Theorem 2 bound, the finite Corollary 2.1 minimum, and the deterministic cap
without new randomized algorithm runs. Store the minimizing analysis parameters
and report each bound divided by actual processed directions.

### Paired ablations

Use both spectra, target rank 192, and the first three main-comparison seeds.
Only the scalar sequential-orthogonalization ablation adds six complete runs.
Continuous inversion versus grid rounding replays the same saved certificate
trace. Terminal adjoint calls are measured from the existing full runs.

## 5. Result schema

Every run emits one append-only JSON record containing:

- schema and software versions;
- complete serialized algorithm and operator configuration;
- RNG algorithm, seed, stream, and pairing ID;
- dimensions, spectrum ID, tolerance, and known optimal ranks/tails;
- terminal reason, certificate bound, truncation budget, and output rank;
- all five operator counters and unused computed columns;
- total and phase timings;
- final error when independently measurable;
- orthogonality errors, retries, thresholds, and enclosure status;
- compiler, build type, BLAS/LAPACK identity, CPU, thread counts, and host OS;
- success or a structured failure status.

Raw records are immutable. Derived tables and figures carry the input-record
hash and plotting-script version.

## 6. Reproducibility controls

- A checked-in seed registry prevents accidental seed reuse or cherry-picking.
- Paired methods consume the same named random panels where their algorithms
  permit a meaningful pairing.
- Warmup is separate from timed runs and recorded in the manifest.
- BLAS and OpenMP thread counts are fixed before process launch.
- CPU affinity and frequency controls are recorded when available, not silently
  assumed.
- Generated raw data, plots, and build products are not committed; release
  artifacts include checksums and exact commands.
- A small CI profile validates the pipeline. The full approximately three-hour
  profile is an explicit release workflow.

## 7. Parameters still requiring a paper-level decision

Implementation must not invent the following values:

- exact smooth-spectrum formula and exponent;
- exact clustered-spectrum construction and cluster width;
- tolerance for every target-rank configuration;
- `delta` used in the main comparison and ablations;
- Theorem 2 and Corollary 2.1 values of `rho`, `eta_r`, `eta_c`, and `alpha`;
- baseline block, oversampling, stopping, and final-truncation parameters;
- compiler/BLAS versions for the designated paper machine.

These values will be frozen in manifests before the full implementation is
considered experiment-ready.
