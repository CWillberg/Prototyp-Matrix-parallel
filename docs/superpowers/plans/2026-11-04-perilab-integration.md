# PD_Matrix → PeriLab Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PeriLab's matrix-based solvers run distributed (MPI), on the GPU and mixed CPU+GPU by
replacing PeriLab's serial `SparseMatrixCSC` path with the PD_Matrix prototype, use case by use case,
while the current serial path stays the default and keeps its reference results.

**Architecture:** The prototype modules move unchanged into `PeriLab.jl/src/Core/Solver/Matrix/`
(module `PD_Matrix`, no PeriLab imports, README A0.0/A0.3). One adapter module
(`Matrix_Adapter`) is the only code that reads `Data_Manager`: it builds the halo exchange from
PeriLab's overlap map, the element data from PeriLab fields, and converts between PeriLab's
`nnodes × dof` fields and the prototype's interleaved local vectors. Solvers choose the backend with
a new input key `Matrix Backend`; `"SparseArrays"` (default) is today's code path. CUDA is a weak
dependency (`ext/PeriLabCUDAExt.jl`).

**Tech Stack:** Julia ≥ 1.12, MPI.jl 0.20, KernelAbstractions, Adapt, ArraysOfArrays,
StaticArrays, TimerOutputs, CUDA.jl (extension only).

**Spec:** Requirements A0–A8 in `/localdata/hess_ja/Prototyp-Matrix-parallel/README.md` (section
"Anforderungen"), plus the measured comparison of 2026-11-04 (prototype vs. PeriLab 3.0.0 LU:
3.4× faster at 1M DOFs on 56 ranks, slower on 1 core; Matrix Verlet is the best fit because it only
needs SpMV).

## Phases and use cases

| Phase | Tasks | Use case delivered | Hardware |
|---|---|---|---|
| 1 Foundation | 1–3 | library inside PeriLab, halo from PeriLab, PeriLab correspondence element | CPU, MPI |
| 2 Matrix Verlet | 4–7 | explicit dynamics, DCB 100k points | 1 CPU → 56 ranks, 1 GPU, mixed, cluster |
| 3 Implicit | 8 | Linear Static and Newmark with `BiCGStab` | same |
| 4 Changing matrices | 9–10 | damage (DCB crack growth), additive (node activation) | same |

Not in this plan (non-goals): model reduction (Guyan, Craig-Bampton stay serial LU), stronger
preconditioners (README F7, separate plan), materials other than Correspondence Elastic,
CUDA-aware MPI tuning (F6), multi-node GPU scheduling beyond "device = local rank mod #GPUs".

## Global Constraints

- PeriLab `Project.toml`: `julia = "1.12"`; new hard deps only `KernelAbstractions`, `Adapt`; `CUDA` only under `[weakdeps]` + `[extensions] PeriLabCUDAExt = "CUDA"`.
- Code in `src/Core/Solver/Matrix/` (except `Matrix_Adapter.jl`) must not `using` any PeriLab module (README A0.3).
- Style: SciML JuliaFormatter (`.JuliaFormatter.toml`), SPDX/REUSE header on every new file (`SPDX-FileCopyrightText: 2026 Christian Willberg`, `BSD-3-Clause`), `@timeit` names as in the prototype.
- `Matrix Backend` values: `"SparseArrays"` (default, current code), `"CPU"`, `"GPU"`, `"Mixed"`. With `"SparseArrays"` every existing fullscale reference must pass unchanged.
- Results must not depend on rank count or backend beyond rounding (README A7.2): compare with `exodiff -p` and the folder's `.cmd` tolerances; matrices with `rtol = 1e-12`.
- Prototype matrices use the stiffness sign (`K u = −f_int`) and interleaved local DOFs `(i−1)·dof + c`; PeriLab's `get_stiffness_matrix()` uses the force sign and component-major DOFs `(c−1)·nnodes + i`. Conversion happens only in `Matrix_Adapter`.
- Local node order is PeriLab's: own nodes `1:get_nnodes()`, then ghosts (`loc_to_glob(get_nnodes()+1:end)`).

## Review Focus

1. **Ghosts not grouped by owner rank** (neighbor-based distribution) — halo exchange must still deliver the right values. Test in Task 2 (`ghosts in arbitrary owner order`).
2. **A rank without ghosts or neighbours on one side** (3 ranks on a narrow mesh, end ranks) — exchange and accumulate are no-ops, no hang. Test in Task 2 (`three ranks, end ranks`).
3. **Points with zero bonds** (bond filter / pre-crack, damage) — element writes nothing, row stays zero, no NaN in acceleration or Jacobi. Test in Task 3 (`isolated point`).
4. **Non-uniform density** (multi-block DCB with different densities) — per-node mass must be used; the existing serial path indexes `density_mass` interleaved while the state is component-major. Test in Task 4 (`two densities`); do not fix the serial path here, report it.
5. **GPU requested without functional CUDA, or more GPU ranks than GPUs** — `"GPU"` aborts with a clear message, `"Mixed"` falls back to CPU ranks. Test in Task 5.

---

## File Structure (in `/localdata/hess_ja/PeriLab.jl`)

| File | Responsibility |
|---|---|
| `src/Core/Solver/Matrix/PD_Matrix.jl` | module root, copied from prototype `src/PD_Matrix.jl` (without `Linear_Solver`, `Guyan`) |
| `src/Core/Solver/Matrix/{Core,MPI_communication,Models,Support}/…` | prototype files, unchanged paths below `src/` |
| `src/Core/Solver/Matrix/Backend.jl` | `select_backend`, GPU hook filled by the extension (from prototype `test/mpi/backend.jl`) |
| `src/Core/Solver/Matrix/Models/Elements/Correspondence/PeriLab_Correspondence.jl` | element reproducing `assemble_stiffness_with_zero_energy` |
| `src/Core/Solver/Matrix/Matrix_Adapter.jl` | the only PeriLab-aware file: halo, element data, field↔vector, `MatrixSystem` |
| `ext/PeriLabCUDAExt.jl` | registers `CuArray` and device selection |
| `src/Core/Solver/Matrix_Verlet.jl` | `Matrix Backend` option, distributed force path |
| `src/Core/Solver/Matrix_linear_static.jl`, `Newmark.jl` | `Linear Solver` option (Task 8) |
| `src/IO/mesh_data.jl` | weighted `create_distribution` (Task 6) |
| `test/unit_tests/Core/Solver/Matrix/…` | prototype unit tests + adapter tests |
| `test/multifunctional_test/test_MPI_matrix.jl` | MPI tests (2–3 ranks) via `run_mpi_test` |
| `test/fullscale_tests/test_matrix_backend/` | decks + runner comparing backends against existing references |
| `test/benchmarks/DCB_matrix/` | 100k DCB generator, decks, benchmark + SLURM script |

---

## Phase 1 — Foundation

### Task 1: Move the prototype into PeriLab

**Files:**
- Create: `src/Core/Solver/Matrix/**` (copy of prototype `src/` minus `Core/Solver/Linear_Solver.jl`, `Core/Model_reduction/Guyan.jl`), `src/Core/Solver/Matrix/Backend.jl`
- Modify: `Project.toml` (deps, compat, weakdeps, extensions), `src/Core/Solver/Solver_manager.jl` (include `Matrix/PD_Matrix.jl` before the solvers)
- Test: `test/unit_tests/Core/Solver/Matrix/ut_*.jl` (prototype `test/` unit tests for CSR, assembly, dof sets, operators, bond-based, correspondence, BiCGStab), registered in `test/runtests.jl` under `"Core"`

**Interfaces:**
- Produces: `PeriLab.Solver_Manager.PD_Matrix` with the prototype exports (`create_assembly_pattern`, `create_matrix`, `create_element_buffer`, `compute_elements!`, `assemble!`, `create_dof_set`, `ConstrainedOperator`, `constrained_rhs!`, `apply_prescribed!`, `create_halo_exchange`, `exchange!`, `accumulate_ghosts!`, `DistributedOperator`, `MPISpace`, `SerialSpace`, `inner`, `BiCGStabWorkspace`, `jacobi!`, `solve_bicgstab!`, `to_sparse`, `data_bytes`).
- Produces: `select_backend(mode::String, comm::MPI.Comm) -> (array_type, on_gpu::Bool, device::Int)`; `GPU_HOOK::Ref{Any}` (nothing until the extension sets `(array_type = CuArray, n_devices = () -> Int, device! = Int -> nothing)`).

- [ ] **Step 1:** Copy the prototype unit tests, change `using PD_Matrix` to `using PeriLab.Solver_Manager.PD_Matrix`, add them to `runtests.jl`.
- [ ] **Step 2:** Run `julia --project -e 'using Pkg; Pkg.test()'` filtered to the new testsets → FAIL (module missing).
- [ ] **Step 3:** Copy the sources, add deps/compat (`KernelAbstractions = "0.9, 0.10"`, `Adapt = "4"`), weakdep `CUDA = "5"`, include in `Solver_manager.jl`. `Backend.jl`: prototype `select_backend` with `Main.CUDA` replaced by `GPU_HOOK[]`.
- [ ] **Step 4:** Run the new testsets and `Aqua` → PASS; run full `Pkg.test()` → all existing tests PASS.
- [ ] **Step 5:** Commit `feat(matrix): add PD_Matrix library`.

### Task 2: Halo exchange from PeriLab's distribution

**Files:**
- Modify: `src/Core/Solver/Matrix/MPI_communication/Halo_Exchange.jl` (drop the "grouped by owner" requirement)
- Create: `src/Core/Solver/Matrix/Matrix_Adapter.jl` (first part)
- Test: `test/multifunctional_test/test_MPI_matrix.jl` (run by `run_mpi_test(..., 2 / 3, true)` from `runtests.jl` MPI testset), `test/unit_tests/Core/Solver/Matrix/ut_matrix_adapter.jl`

**Interfaces:**
- Produces: `create_halo_exchange(comm, own_points, ghost_points, ghost_owner, dof; array_type, cuda_aware)` — same signature, ghosts may be in any owner order (receive into a buffer, scatter with a `recv_idx` kernel when not grouped).
- Produces (adapter): `perilab_halo(dof::Int64; array_type = Array) -> Union{HaloExchange,Nothing}` — own = `loc_to_glob(1:get_nnodes())`, ghosts = the remaining local nodes, owner from `get_overlap_map()[rank+1][j]["Controller"]`; `nothing` for one rank.

- [ ] **Step 1:** MPI test `ghosts in arbitrary owner order`: 3 ranks, each owns 4 points, ghost list shuffled; `x[own] = global_id`; after `exchange!` every ghost entry equals its global id.
- [ ] **Step 2:** MPI test `three ranks, end ranks`: 1D chain, rank 0 and 2 have one neighbour; `exchange!` and `accumulate_ghosts!` (ghost entries = 1 → owners get +1 per requesting rank) give the expected sums.
- [ ] **Step 3:** MPI test `perilab_halo`: run PeriLab's mesh distribution on `test/fullscale_tests/test_DCB` mesh with 2 ranks (`PeriLab.IO.initialize_data` as in `test_MPI_data_manager.jl`), then `exchange!` of coordinates (dof = 2) reproduces the ghost coordinates in `Data_Manager.get_field("Coordinates")`.
- [ ] **Step 4:** Run `mpiexec -n 3 julia --project test/multifunctional_test/test_MPI_matrix.jl` → FAIL.
- [ ] **Step 5:** Implement; run again → PASS; prototype halo unit tests still PASS.
- [ ] **Step 6:** Commit `feat(matrix): halo exchange from PeriLab overlap map`.

### Task 3: PeriLab correspondence element

**Files:**
- Create: `src/Core/Solver/Matrix/Models/Elements/Correspondence/PeriLab_Correspondence.jl`
- Modify: `src/Core/Solver/Matrix/Matrix_Adapter.jl` (element data, system setup, conversion)
- Test: `test/unit_tests/Core/Solver/Matrix/ut_perilab_correspondence.jl`

**Interfaces:**
- Consumes: `perilab_halo` (Task 2).
- Produces: `PeriLabCorrespondence{D}(zero_energy::Bool) <: AbstractElement`; element of own point `i` writes rows of `i` and its neighbours, columns `i` and its neighbours (same `element_dofs` as `CorrespondenceElastic`). Values equal `−` PeriLab's per-node contribution in `assemble_stiffness_with_zero_energy` (normal part via `CB` tensors and `k_loop!`, plus zero-energy part from `zStiff`), bonds with `bond_damage == 0` skipped.
- Produces: `correspondence_data() -> NamedTuple` (flat arrays: `nlist_ptr`, `nlist`, `volume`, `omega`, `bond_damage`, `bond_geometry` (D × n_bonds), `inverse_shape_tensor`, `CB` per bond, `zStiff`), built from `Data_Manager` fields after `init_matrix` prerequisites (`_apply_rotation`, `precompute_CB_all_nodes!`, `_setup_zero_energy`) on own nodes.
- Produces: `MatrixSystem` with fields `pattern, K, ke, data, element, halo, op::DistributedOperator, n_own::Int64, dof::Int64, x, y` (local vectors on the backend) and
  - `init_matrix_system(backend::String) -> MatrixSystem` (symbolic phase on host, `adapt`, first assembly)
  - `assemble_system!(sys) -> sys` (`compute_elements!` + `assemble!`)
  - `field_to_vector!(x, field::AbstractMatrix, n::Int)` / `vector_to_field!(field, x, n::Int)`: `nnodes × dof` ↔ interleaved `x[(i−1)·dof + c]` for `i ∈ 1:n` (host↔device copy inside when `x` is a device vector)
  - `apply_stiffness!(f_field, u_field, sys)`: `f_field[own, :] = −K u` (force sign), the only call solvers need for SpMV.

- [ ] **Step 1:** Test `matches PeriLab matrix`: for `test_linear_static_matrix_based_solver/test_mesh_2D.txt`, `test_mesh_3D.txt` and `DCBmodel.txt` (1 rank) run PeriLab init up to `init_matrix()`, then `to_sparse(sys.K) ≈ −P' * get_stiffness_matrix() * P` with `rtol = 1e-12`, `P` = interleaved→component-major permutation.
- [ ] **Step 2:** Test `isolated point`: mesh where one point has all bonds filtered → its row in `sys.K` is all zero and `apply_stiffness!` returns finite forces.
- [ ] **Step 3:** Test `apply_stiffness! = −K u`: random `u`, compare to `get_stiffness_matrix() * vec(u)` reshaped (`rtol = 1e-12`).
- [ ] **Step 4:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 5:** Commit `feat(matrix): PeriLab correspondence element and MatrixSystem`.

---

## Phase 2 — Matrix Verlet on all hardware

### Task 4: Matrix Verlet with `Matrix Backend: CPU` (MPI)

**Files:**
- Modify: `src/Core/Solver/Matrix_Verlet.jl` (`VerletMatrixParams`, `init_solver`, `MatrixVerletWorkspace`, `time_step!`), `src/Models/Material/Material_Models/Correspondence/Correspondence_matrix_based.jl:628` (abort only when the backend is `"SparseArrays"`)
- Create: `test/fullscale_tests/test_matrix_backend/` (`verlet_full_matrix_cpu.yaml` = copy of `test_model_reduction_Verlet_matrix/verlet_full_matrix.yaml` + `Matrix Backend: CPU`, `two_densities.yaml`, `.cmd` copied, `Reference/verlet_full_matrix_cpu.e` = copy of `test_model_reduction_Verlet_matrix/Reference/verlet_full_matrix.e`), `test_matrix_backend.jl`; register in `runtests.jl`

**Interfaces:**
- Consumes: `init_matrix_system`, `apply_stiffness!` (Task 3).
- Produces: `VerletMatrixParams.matrix_backend::String = opt("Matrix Backend"; default = "SparseArrays")`, `gpu_weight::Float64 = opt("GPU Weight"; default = 1.0, min = 0)`; `check!` rejects unknown backends and `Model Reduction` together with a non-`SparseArrays` backend (message: `"Model Reduction requires Matrix Backend: SparseArrays"`).
- Distributed step: acceleration `a[own, :] = f[own, :] ./ density[own]` (per node, no `lu` of the mass); `compute_models` is not called for the matrix part (same as today without reduction).

- [ ] **Step 1:** Runner: `run_perilab("verlet_full_matrix_cpu", 1, true, folder)` and `run_perilab("verlet_full_matrix_cpu", 2, true, folder)` and `(…, 3, …)` against the reference of the `SparseArrays` run.
- [ ] **Step 2:** `two densities`: two-block deck with densities 1.0 and 2.0; CPU backend on 1 and 2 ranks equals its own 1-rank `Reference` (generated once from the CPU backend and checked by hand against `a = f/ρ` at one node).
- [ ] **Step 3:** Run → FAIL. Implement. Run → PASS; `test_model_reduction_Verlet_matrix` unchanged → PASS.
- [ ] **Step 4:** Commit `feat(matrix-verlet): distributed CPU backend`.

### Task 5: GPU backend via CUDA extension

**Files:**
- Create: `ext/PeriLabCUDAExt.jl`
- Modify: `test/fullscale_tests/test_matrix_backend/` (`verlet_full_matrix_gpu.yaml`)

**Interfaces:**
- Consumes: `GPU_HOOK`, `select_backend` (Task 1).
- Produces: with `using CUDA` loaded and `Matrix Backend: GPU` every rank uses device `local_rank % n_devices`; `"Mixed"`: one rank per GPU on the GPU, other ranks on CPU.

- [ ] **Step 1:** Tests (skipped with `@test_skip` when `CUDA.functional()` is false): `verlet_full_matrix_gpu` on 1 rank equals the reference; `Matrix Backend: GPU` without CUDA loaded aborts with `"Matrix Backend GPU requires CUDA.jl (using CUDA) and a functional GPU"`; `Mixed` without CUDA runs as CPU and logs one warning.
- [ ] **Step 2:** Run → FAIL. Implement. Run on this machine (1× RTX A6000) → PASS.
- [ ] **Step 3:** Commit `feat(matrix): CUDA extension`.

### Task 6: Weighted distribution for `Mixed`

**Files:**
- Modify: `src/IO/mesh_data.jl` (`create_distribution`, its caller)
- Test: `test/unit_tests/IO/ut_mesh_data.jl` (existing file, add testset)

**Interfaces:**
- Produces: `create_distribution(nnodes::Int64, size::Int64; weights::Vector{Float64} = ones(size))` — contiguous chunks with lengths `round(nnodes · w_r / Σw)`, at least one node each, last chunk takes the rest; `rank_weights(comm, backend, gpu_weight) -> Vector{Float64}` (`gpu_weight` for ranks that `select_backend` puts on a GPU, else 1.0; one `Allgather`).

- [ ] **Step 1:** Unit test: `create_distribution(100, 3; weights = [16.0, 1.0, 1.0])` → chunk lengths `[89, 6, 5]`; default weights reproduce today's result for `(10, 3)`.
- [ ] **Step 2:** Run → FAIL. Implement; the caller passes weights only when the solver is `Verlet Matrix Based` with `Matrix Backend: Mixed`. Run → PASS.
- [ ] **Step 3:** Mixed fullscale run (2 ranks, `GPU Weight: 16`, skipped without CUDA) equals the reference.
- [ ] **Step 4:** Commit `feat(io): weighted node distribution for mixed CPU/GPU`.

### Task 7: DCB benchmark (100k points) and cluster script

**Files:**
- Create: `test/benchmarks/DCB_matrix/create_dcb_mesh.jl`, `dcb_matrix_verlet.yaml`, `run_benchmark.jl`, `slurm_dcb.sh`, `README.md`

**Interfaces:**
- `create_dcb_mesh.jl n_points=100000` writes `DCB_<n>.txt` (`header: x y block_id volume`) and node sets with the geometry and blocks of `test_linear_static_matrix_based_solver/DCBmodel.txt` (refined grid, same bond filter `bf_1`).
- `run_benchmark.jl` runs `{SparseArrays 1 rank, CPU 1/8/28/56 ranks (-bind-to core), GPU 1 rank, Mixed 56 ranks with GPU Weight 55}` for a fixed number of steps, writes one CSV line per run (`backend, ranks, n_points, t_init, t_step_mean, max |u_y|`).
- `slurm_dcb.sh`: `MPIPreferences.use_system_binary()` note, `srun` launcher, `--gpus-per-node`, `Matrix Backend: GPU` one rank per GPU.

- [ ] **Step 1:** Run the benchmark at 10k points: all backends give the same `max |u_y|` (rtol 1e-10).
- [ ] **Step 2:** Run at 100k points; record the table in `README.md`.
- [ ] **Step 3:** Commit `bench: DCB matrix verlet across backends`.

---

## Phase 3 — Implicit solvers

### Task 8: `Linear Solver: BiCGStab` for Linear Static and Newmark

**Files:**
- Modify: `src/Core/Solver/Matrix_linear_static.jl`, `src/Core/Solver/Newmark.jl`, their `@params`; `src/Core/Solver/Matrix/Core/Operators.jl` (shifted operator), `Core/Solver/BiCGStab.jl` (`jacobi!` with shift, start vector)
- Test: `test/fullscale_tests/test_matrix_backend/` (`linear_static_bicgstab.yaml`, `newmark_bicgstab.yaml` from `test_linear_static_matrix_based_solver` / `test_Newmark` decks)

**Interfaces:**
- Produces: `linear_solver::String = opt("Linear Solver"; default = "LU")` (`"LU"`, `"BiCGStab"`), `rtol::Float64 = opt("Linear Solver Tolerance"; default = 1e-10)`, `itmax::Int64 = opt("Linear Solver Maximum Iterations"; default = 20000)`; `BiCGStab` requires a non-`SparseArrays` backend.
- Produces: `ShiftedOperator(A::ConstrainedOperator, shift::AbstractVector)` with `mul!(y, S, x) = A x + shift .* x` on free own DOFs (`K + a0·M` for Newmark); `jacobi!(ws, K, prescribed; shift = nothing, …)`; `solve_bicgstab!(…; x0 = nothing)` starts from `x0` when given.
- BCs: `DofSet` from `get_bc_free_dof()` complement plus inactive nodes; RHS via `constrained_rhs!`. A non-converged solve returns `StepFailed` (existing fallback mechanism).

- [ ] **Step 1:** Tests: BiCGStab (CPU, 1 and 2 ranks) results equal the LU references with the folder `.cmd` tolerances; a deck with `Linear Solver Maximum Iterations: 1` returns `StepFailed` and logs the residual.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS; LU references unchanged.
- [ ] **Step 3:** Commit `feat(solvers): BiCGStab option for linear static and Newmark`.

---

## Phase 4 — Changing matrices

### Task 9: Damage (DCB crack growth) in Matrix Verlet

**Files:**
- Modify: `src/Core/Solver/Matrix_Verlet.jl` (`update_system!` call; stop overwriting `Active` at line 228 when no model reduction is configured), `Matrix_Adapter.jl` (`update_system!`)
- Test: `test/fullscale_tests/test_matrix_backend/dcb_damage.yaml` (DCB deck with `Damage Model: Damage` enabled in blocks 1–2)

**Interfaces:**
- Produces: `update_system!(sys, update::AbstractVector{Bool})` — copies `Bond Damage NP1` of flagged own nodes into `sys.data`, recomputes `compute_elements!` and `assemble!` (pattern unchanged, README A6a.1); called after `compute_models` when any `Update` flag is set (one `MPI.Allreduce(|)`); damage models run on own nodes via the existing `compute_models` (not empty anymore when damage is on).
- Reference for the matrix: PeriLab's `rebuild_matrix!` (`Correspondence_matrix_based.jl:809`).

- [ ] **Step 1:** Unit test: set damage of 3 bonds to 0, `update_system!` → `to_sparse(sys.K) ≈ −P' K_rebuilt P` (rtol 1e-12).
- [ ] **Step 2:** Fullscale: `dcb_damage` CPU 1 rank vs 2 ranks equal (exodiff `-p`), damage field non-zero at the end.
- [ ] **Step 3:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 4:** Commit `feat(matrix-verlet): damage updates`.

### Task 10: Additive (node activation) with status mask

**Files:**
- Modify: `src/Core/Solver/Matrix/Core/Matrix_Assembly.jl` (gather with status mask, README A6a.2), `Matrix_Adapter.jl`, `Matrix_Verlet.jl` (call `update_system!` when `Active` changes)
- Test: `test/fullscale_tests/test_matrix_backend/additive_verlet.yaml` (mesh from `test_matrix_based_additive/mesh.txt`, `Additive Models: true`, mechanical only)

**Interfaces:**
- Produces: `assemble!(K, pattern, ke; active = nothing)` — with `active` (per-DOF `Bool` mask on the backend) rows and columns of inactive DOFs are 0, diagonal 1; bonds to inactive points get `bond_damage = 0` in the element data; `update_system!` is called when the `Active` field changed (count via `Allreduce`).
- The prefix-view variant (README A6a.3, numbering by activation time) is not part of this plan.

- [ ] **Step 1:** Unit test: 4-point chain, point 4 inactive → row/column 4 zero except diagonal 1, `K[1:3,1:3]` equals the matrix of the 3-point chain.
- [ ] **Step 2:** Fullscale: CPU 1 vs 2 ranks equal; inactive nodes keep zero displacement.
- [ ] **Step 3:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 4:** Commit `feat(matrix): node activation for additive`.
