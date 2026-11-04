# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Distributed test scenarios. Run with any number of ranks, e.g.
#
#   mpiexecjl -n 4 julia --project=test/mpi test/mpi/scenarios.jl             # CPU, all
#   mpiexecjl -n 2 julia --project=test/mpi test/mpi/scenarios.jl gpu         # GPU ranks
#   mpiexecjl -n 2 julia --project=test/mpi test/mpi/scenarios.jl bar_nonsym memory timer
#
# Arguments (any order): `cpu` | `gpu` | `mixed` (one rank per GPU on the GPU, the others on
# the CPU, coupled through the same halo exchange), scenario names (default: all), `timer` (prints the
# TimerOutputs table of rank 0). Exit code 1 if any check fails.

using MPI
using PD_Matrix
using Adapt: adapt
using LinearAlgebra: norm, mul!, I
using Random: MersenneTwister
using SparseArrays: nnz
using Printf: @sprintf
using TimerOutputs: print_timer, reset_timer!

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const N_RANKS = MPI.Comm_size(COMM)

const BACKEND = "mixed" in ARGS ? "mixed" : ("gpu" in ARGS ? "gpu" : "cpu")
if BACKEND != "cpu"
    using CUDA
end
include(joinpath(@__DIR__, "backend.jl"))
const ARRAY_TYPE, ON_GPU, DEVICE = select_backend(BACKEND, COMM)
const N_GPU_RANKS = MPI.Allreduce(Int(ON_GPU), +, COMM)

include(joinpath(@__DIR__, "..", "helper.jl"))
include(joinpath(@__DIR__, "scenario_helper.jl"))

const FAILURES = Ref(0)
const QUIET = Ref(false)   # warm-up run before timing

"Collective check: passes only if it passes on every rank."
function check(scenario::String, what::String, ok::Bool, detail::String = "")
    all_ok = MPI.Allreduce(ok, &, COMM)
    all_ok || (FAILURES[] += 1)
    if RANK == 0 && !QUIET[]
        println(all_ok ? "[PASS] " : "[FAIL] ", rpad(scenario, 12), " ranks=", N_RANKS,
                " ", BACKEND, BACKEND == "mixed" ? "($(N_GPU_RANKS) gpu)" : "", "  ", what,
                isempty(detail) ? "" : "  ($detail)")
    end
    return all_ok
end
global_max(x) = MPI.Allreduce(x, max, COMM)
global_sum(x) = MPI.Allreduce(x, +, COMM)

function bar_problem(n; volume = fill(1.0, n))
    coordinates = reshape(collect(0.0:(n - 1)), 1, n)
    return coordinates, volume, 3.0
end

function plate_problem(nx, ny)
    coordinates = zeros(2, nx * ny)
    for j in 1:ny, i in 1:nx
        coordinates[:, (j - 1) * nx + i] = [i - 1.0, j - 1.0]
    end
    return coordinates, fill(1.0, nx * ny), 2.0
end

"Serial reference on every rank (small problems only)."
function reference(coordinates, volume, horizon, D, prescribed, values, f;
                   element = nothing)
    nlist = neighbor_list(coordinates, horizon)
    data = create_bondbased_data(coordinates, volume, nlist, unit_damage(nlist),
                                 fill(1.0, size(coordinates, 2)))
    K = isnothing(element) ? reference_bondbased(data, D) :
        reference_correspondence(data, D, element.lambda, element.mu, element.stabilization)
    u = reference_solve(K, prescribed, values, f)
    return K, u, K * u - f
end

function compare_with_reference(name, result, K_ref, u_ref, r_ref, prescribed)
    own = result.l2g[1:(result.n_own)]
    check(name, "BiCGStab converged", result.stats.solved,
          "$(result.stats.niter) iterations")
    if !result.accumulate
        K_local = Matrix(to_sparse(result.K))[1:(result.n_own), :]
        err_K = global_max(maximum(abs, K_local - K_ref[own, result.l2g]; init = 0.0))
        check(name, "own rows of K equal rows of serial K", err_K < 1e-12,
              @sprintf("max diff %.1e", err_K))
    end
    x_global = rand(MersenneTwister(7), size(K_ref, 1))       # same on every rank
    y = distributed_matvec(result, x_global, ARRAY_TYPE)
    err_y = global_max(maximum(abs, y - (K_ref * x_global)[own]; init = 0.0))
    check(name,
          "distributed K x equals serial K x" *
          (result.accumulate ? " (with ghost-row accumulation)" : ""),
          err_y < 1e-10 * max(1.0, maximum(abs, K_ref)), @sprintf("max diff %.1e", err_y))
    scale = maximum(abs, u_ref)
    err_u = global_max(maximum(abs, result.u - u_ref[own]; init = 0.0)) / scale
    check(name, "u equals serial reference", err_u < 1e-8, @sprintf("rel. %.1e", err_u))
    exact = all(result.u[i] == u_ref[g] for (i, g) in enumerate(own) if g in prescribed)
    check(name, "prescribed values exact", exact)
    mine = [i for (i, g) in enumerate(own) if g in prescribed]
    err_r = global_max(maximum(abs, result.r[mine] - r_ref[own[mine]]; init = 0.0))
    check(name, "reactions equal serial reference", err_r < 1e-6 * max(1.0, scale),
          @sprintf("max diff %.1e", err_r))
end

function scenario_halo()
    for (D, problem) in ((1, bar_problem(40)), (2, plate_problem(12, 5)))
        coordinates, volume, horizon = problem
        part = distribute_points(coordinates, volume, horizon, RANK, N_RANKS)
        l2g = local_to_global_dofs(part, D)
        n_own = D * length(part.own)
        x_host = zeros(length(l2g))
        x_host[1:n_own] = l2g[1:n_own]
        x = adapt(ARRAY_TYPE, x_host)
        halo = create_halo_exchange(COMM, part.own, part.ghosts, part.ghost_owner, D;
                                    array_type = ARRAY_TYPE)
        exchange!(halo, x)
        check("halo", "ghost values from owners (D = $D)", Array(x) == Float64.(l2g),
              "$(length(l2g) - n_own) ghost dofs on rank 0")
    end
end

function scenario_bar(name; volume_fn = n -> fill(1.0, n))
    n = 100
    coordinates, volume, horizon = bar_problem(n; volume = volume_fn(n))
    prescribed = [1, n]
    values = [0.0, 100.0]
    f = zeros(n)
    f[50] = 10.0
    K_ref, u_ref, r_ref = reference(coordinates, volume, horizon, 1, prescribed, values, f)
    part = distribute_points(coordinates, volume, horizon, RANK, N_RANKS)
    result = solve_distributed(part, 1, prescribed, values, f; array_type = ARRAY_TYPE)
    compare_with_reference(name, result, K_ref, u_ref, r_ref, prescribed)
    if name == "bar_nonsym"
        check(name, "matrix is not symmetric", !(K_ref ≈ K_ref'))
    else
        total = global_sum(sum(result.r))
        check(name, "global equilibrium: sum of reactions = -F", abs(total + 10.0) < 1e-6,
              @sprintf("sum %.6f", total))
    end
    return result
end

function scenario_plate()
    nx, ny, D = 24, 8, 2
    coordinates, volume, horizon = plate_problem(nx, ny)
    left = [(j - 1) * nx + 1 for j in 1:ny]
    right = [(j - 1) * nx + nx for j in 1:ny]
    prescribed = vcat([2 * (p - 1) + d for p in left for d in 1:2],
                      [2 * (p - 1) + 1 for p in right])
    values = vcat(zeros(2 * ny), fill(1.0, ny))
    f = zeros(D * nx * ny)
    K_ref, u_ref, r_ref = reference(coordinates, volume, horizon, D, prescribed, values, f)
    part = distribute_points(coordinates, volume, horizon, RANK, N_RANKS)
    result = solve_distributed(part, D, prescribed, values, f; array_type = ARRAY_TYPE)
    compare_with_reference("plate_2d", result, K_ref, u_ref, r_ref, prescribed)
end

function scenario_bc_values()
    n = 100
    coordinates, volume, horizon = bar_problem(n)
    prescribed = [1, n]
    part = distribute_points(coordinates, volume, horizon, RANK, N_RANKS)
    f = zeros(n)
    f[50] = 10.0
    first_result = solve_distributed(part, 1, prescribed, [0.0, 100.0], f;
                                     array_type = ARRAY_TYPE)
    nzval_before = Array(first_result.K.nzval)

    # new values and load, same matrix, same workspace: no new assembly
    values = [0.0, 50.0]
    f2 = zeros(n)
    f2[30] = -4.0
    l2g = first_result.l2g
    u_host = zeros(first_result.n_local)
    for (i, g) in enumerate(l2g)
        g == n && (u_host[i] = 50.0)
    end
    f_host = zeros(first_result.n_local)
    f_host[1:(first_result.n_own)] = f2[l2g[1:(first_result.n_own)]]
    u = adapt(ARRAY_TYPE, u_host)
    f_vec = adapt(ARRAY_TYPE, f_host)
    b = fill!(similar(f_vec), 0.0)
    constrained_rhs!(b, first_result.K, first_result.bc, u, f_vec)
    x, stats = solve_bicgstab!(first_result.ws, first_result.A, b, first_result.space;
                               rtol = 1e-12, itmax = 50_000)
    apply_prescribed!(x, first_result.bc, u)

    _, u_ref, _ = reference(coordinates, volume, horizon, 1, prescribed, values, f2)
    own = l2g[1:(first_result.n_own)]
    err = global_max(maximum(abs, Array(x)[1:(first_result.n_own)] - u_ref[own];
                             init = 0.0)) / 50.0
    check("bc_values", "new BC values and load without reassembly",
          stats.solved && err < 1e-8,
          @sprintf("rel. %.1e, %d iterations", err, stats.niter))
    check("bc_values", "matrix values unchanged",
          Array(first_result.K.nzval) == nzval_before)
end

function scenario_memory()
    n = 10_000
    coordinates, volume, horizon = bar_problem(n)
    part = distribute_points(coordinates, volume, horizon, RANK, N_RANKS)
    f = zeros(n)
    f[div(n, 2)] = 10.0
    result = solve_distributed(part, 1, [1, n], [0.0, 100.0], f; array_type = ARRAY_TYPE,
                               rtol = 1e-6, itmax = 200)
    nnz_local = length(result.K.nzval)
    sizes = ["own rows" => result.n_own, "ghost cols" => result.n_local - result.n_own,
        "nnz" => nnz_local, "elem. values" => length(result.ke)]
    RANK == 0 && !QUIET[] && println("\nMemory per rank (n = $n, $(BACKEND)):")
    memory_report(COMM, sizes, result.memory; io = QUIET[] ? devnull : stdout)
    RANK == 0 && !QUIET[] && println()

    nlist = part.global_nlist
    nnz_global = n + sum(length, nlist)
    check("memory", "sum of local nnz = nnz of the global matrix",
          global_sum(nnz_local) == nnz_global, "$nnz_global")
    if N_RANKS > 1
        check("memory", "no rank holds the whole matrix",
              global_max(nnz_local) < nnz_global,
              "largest rank $(global_max(nnz_local)) of $nnz_global")
    end
end

function scenario_corr(name, D)
    if D == 1
        n = 60
        coordinates, volume, horizon = bar_problem(n; volume = [0.5 + 0.01i for i in 1:n])
        prescribed = [1, n]
        values = [0.0, 1.0]
        f = zeros(n)
        f[30] = 0.1
        element = CorrespondenceElastic(1; lambda = 0.0, mu = 0.5, stabilization = 0.1)
    else
        nx, ny = 16, 6
        coordinates, volume, horizon = plate_problem(nx, ny)
        left = [(j - 1) * nx + 1 for j in 1:ny]
        right = [(j - 1) * nx + nx for j in 1:ny]
        prescribed = vcat([2 * (p - 1) + d for p in left for d in 1:2],
                          [2 * (p - 1) + 1 for p in right])
        values = vcat(zeros(2 * ny), fill(0.5, ny))
        f = zeros(2 * nx * ny)
        element = CorrespondenceElastic(2; lambda = 0.6, mu = 0.4, stabilization = 0.1)
    end
    K_ref, u_ref, r_ref = reference(coordinates, volume, horizon, D, prescribed, values, f;
                                    element = element)
    part = distribute_points(coordinates, volume, horizon, RANK, N_RANKS)
    result = solve_distributed(part, D, prescribed, values, f; element = element,
                               array_type = ARRAY_TYPE)
    check(name, "ghost rows are accumulated", result.accumulate || N_RANKS == 1)
    compare_with_reference(name, result, K_ref, u_ref, r_ref, prescribed)
end

const SCENARIOS = Dict("corr_bar" => () -> scenario_corr("corr_bar", 1),
                       "corr_plate" => () -> scenario_corr("corr_plate", 2),
                       "halo" => scenario_halo,
                       "bar_uniform" => () -> scenario_bar("bar_uniform"),
                       "bar_nonsym" =>
                           () -> scenario_bar("bar_nonsym";
                                              volume_fn = n -> [0.5 + 0.01i
                                                                for i in 1:n]),
                       "plate_2d" => scenario_plate,
                       "bc_values" => scenario_bc_values,
                       "memory" => scenario_memory)
const ORDER = ["halo", "bar_uniform", "bar_nonsym", "plate_2d", "bc_values", "corr_bar",
    "corr_plate", "memory"]

selected = filter(a -> haskey(SCENARIOS, a), ARGS)
isempty(selected) && (selected = ORDER)
if "timer" in ARGS
    # warm-up: compile everything once, so the timer shows run time and allocations only
    QUIET[] = true
    for name in selected
        SCENARIOS[name]()
    end
    QUIET[] = false
end
reset_timer!()
for name in selected
    SCENARIOS[name]()
end
if "timer" in ARGS && RANK == 0
    print_timer(; sortby = :firstexec)
    println()
end
RANK == 0 && println(FAILURES[] == 0 ? "ALL PASSED" : "$(FAILURES[]) CHECK(S) FAILED",
        " (ranks = $N_RANKS, $BACKEND)")
MPI.Barrier(COMM)
exit(FAILURES[] == 0 ? 0 : 1)
