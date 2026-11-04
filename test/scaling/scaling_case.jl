# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# One scaling run (one MPI job). Every rank creates only its own points and ghosts of a
# structured grid (no global data, no global reference), assembles, solves and appends
# one line to a CSV file. Started by run_scaling.jl or a batch script:
#
#   mpiexecjl -n 4 julia --project=test/scaling test/scaling/scaling_case.jl \
#       dim=2 dofs=10000 backend=cpu element=bb csv=results/scaling.csv
#
# Keys: dim=1|2|3, dofs (target, global), backend=cpu|gpu|mixed, element=bb|corr,
# horizon=m (δ = m·Δx; default 3 in 1D/2D, 2 in 3D), gpu_weight=w (points of a GPU rank
# relative to a CPU rank, mixed only), rtol, itmax, csv=path.

using MPI
using PD_Matrix
using Adapt: adapt
using ArraysOfArrays: VectorOfVectors
using LinearAlgebra: mul!, norm
using Printf: @printf, @sprintf
using Dates: now, format
using TimerOutputs: TimerOutputs, reset_timer!, get_defaulttimer

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const N_RANKS = MPI.Comm_size(COMM)

options = Dict(split(a, "="; limit = 2)[1] => split(a, "="; limit = 2)[2]
               for a in ARGS if occursin("=", a))
function get_option(key, default)
    haskey(options, key) ? parse(typeof(default), options[key]) :
    default
end
const DIM = get_option("dim", 1)
const TARGET_DOFS = get_option("dofs", 10_000)
const BACKEND = get(options, "backend", "cpu")
const ELEMENT = get(options, "element", "bb")
const M = get_option("horizon", DIM == 3 ? 2 : 3)
const GPU_WEIGHT = get_option("gpu_weight", 1.0)
const RTOL = get_option("rtol", 1e-8)
const ITMAX = get_option("itmax", 20_000)
const CSV_PATH = get(options, "csv", joinpath("results", "scaling.csv"))

if BACKEND != "cpu"
    using CUDA
end
include(joinpath(@__DIR__, "..", "mpi", "backend.jl"))
const ARRAY_TYPE, ON_GPU, DEVICE = select_backend(BACKEND, COMM)

"Grid with about `dofs` degrees of freedom, elongated in x (slab decomposition along x)."
function grid_size(dofs, dim)
    n = max(dofs ÷ dim, 1)
    dim == 1 && return (n, 1, 1)
    if dim == 2
        nx = round(Int, sqrt(4n))
        return (nx, max(round(Int, n / nx), 1), 1)
    end
    nx = round(Int, cbrt(16n))
    nyz = max(round(Int, sqrt(n / nx)), 1)
    return (nx, nyz, nyz)
end

"Slab bounds along x, weighted by backend (GPU ranks can take more points)."
function slab_bounds(nx, weights)
    total = sum(weights)
    bounds = [0; round.(Int, cumsum(weights) ./ total .* nx)]
    bounds[end] = nx
    for r in 2:length(bounds)                      # at least one layer per rank
        bounds[r] = max(bounds[r], bounds[r - 1] + 1)
    end
    bounds[end] > nx &&
        error("Grid with nx = $nx is too small for $(length(weights)) ranks.")
    return bounds
end

function stencil(m, dim)
    offsets = NTuple{3,Int}[]
    rz = dim == 3 ? m : 0
    ry = dim >= 2 ? m : 0
    for dx in (-m):m, dy in (-ry):ry, dz in (-rz):rz
        (dx, dy, dz) == (0, 0, 0) && continue
        dx^2 + dy^2 + dz^2 <= m^2 + 1e-9 && push!(offsets, (dx, dy, dz))
    end
    return offsets
end

"""
Local part of the grid: own slab `ix0:ix1-1`, then left ghosts, then right ghosts
(ghosts grouped by owner rank). Global id = ix·ny·nz + iy·nz + iz + 1 (rank-sorted).
"""
function local_part(dims, bounds, rank, m, dim)
    nx, ny, nz = dims
    nyz = ny * nz
    ix0, ix1 = bounds[rank + 1], bounds[rank + 2]
    lo, hi = max(0, ix0 - m), min(nx, ix1 + m)
    slabs = vcat(collect(ix0:(ix1 - 1)), collect(lo:(ix0 - 1)), collect(ix1:(hi - 1)))
    slot = zeros(Int, hi - lo)
    for (k, ix) in enumerate(slabs)
        slot[ix - lo + 1] = k - 1
    end
    local_index(ix, iy, iz) = slot[ix - lo + 1] * nyz + iy * nz + iz + 1
    owner(ix) = searchsortedlast(bounds, ix) - 1

    n_local = length(slabs) * nyz
    n_own_points = (ix1 - ix0) * nyz
    coordinates = zeros(dim, n_local)
    global_id = zeros(Int, n_local)
    point_owner = zeros(Int, n_local)
    ix_of = zeros(Int, n_local)
    for ix in slabs, iy in 0:(ny - 1), iz in 0:(nz - 1)
        l = local_index(ix, iy, iz)
        c = (ix, iy, iz)
        for a in 1:dim
            coordinates[a, l] = c[a]
        end
        global_id[l] = ix * nyz + iy * nz + iz + 1
        point_owner[l] = owner(ix)
        ix_of[l] = ix
    end
    offsets = stencil(m, dim)
    nlist = VectorOfVectors{Int64}()
    neighbors = Int64[]
    for ix in ix0:(ix1 - 1), iy in 0:(ny - 1), iz in 0:(nz - 1)
        empty!(neighbors)
        for (dx, dy, dz) in offsets
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            (0 <= jx < nx && 0 <= jy < ny && 0 <= jz < nz) || continue
            push!(neighbors, local_index(jx, jy, jz))
        end
        push!(nlist, neighbors)
    end
    damage = VectorOfVectors([ones(length(l)) for l in nlist])
    data = create_bondbased_data(coordinates, ones(n_local), nlist, damage, ones(n_local))
    ghosts = (n_own_points + 1):n_local
    return (data = data, n_own_points = n_own_points, n_local = n_local,
            own = global_id[1:n_own_points], ghosts = global_id[ghosts],
            ghost_owner = point_owner[ghosts], ix_of = ix_of)
end

function timed(f)
    (MPI.Barrier(COMM); t = MPI.Wtime(); r = f(); MPI.Barrier(COMM);
     (r, MPI.Allreduce(MPI.Wtime() - t, max, COMM)))
end

function section_time(path...)
    t = get_defaulttimer()
    for p in path
        haskey(t.inner_timers, p) || return 0.0
        t = t.inner_timers[p]
    end
    return TimerOutputs.time(t) / 1e9
end

function run_case()
    dims = grid_size(TARGET_DOFS, DIM)
    weights = [w for w in MPI.Allgather(ON_GPU ? GPU_WEIGHT : 1.0, COMM)]
    bounds = slab_bounds(dims[1], weights)
    element = ELEMENT == "corr" ?
              CorrespondenceElastic(DIM; lambda = 0.5, mu = 0.5, stabilization = 0.1) :
              BondbasedElastic(DIM)

    # --- setup: local data, symbolic phase, halo pattern, transfer to the backend -------
    (setup, t_setup) = timed() do
        part = local_part(dims, bounds, RANK, M, DIM)
        n_own = DIM * part.n_own_points
        n_local = DIM * part.n_local
        rows, cols = element_dofs(element, part.data)
        host_pattern = create_assembly_pattern(rows, cols, n_local, n_local)
        accumulate = needs_accumulate(COMM, host_pattern.rowptr, n_own)
        pattern = adapt(ARRAY_TYPE, host_pattern)
        data = adapt(ARRAY_TYPE, part.data)
        K = create_matrix(pattern)
        ke = create_element_buffer(pattern)
        halo = create_halo_exchange(COMM, part.own, part.ghosts, part.ghost_owner, DIM;
                                    array_type = ARRAY_TYPE)
        # BCs: x = 0 clamped, x = L: u_x = 1
        prescribed = Int64[]
        u_host = zeros(n_local)
        for l in 1:(part.n_local)
            ix = part.ix_of[l]
            if ix == 0
                append!(prescribed, ((l - 1) * DIM + 1):(l * DIM))
            elseif ix == dims[1] - 1
                push!(prescribed, (l - 1) * DIM + 1)
                u_host[(l - 1) * DIM + 1] = 1.0
            end
        end
        bc = adapt(ARRAY_TYPE, create_dof_set(prescribed, n_local))
        u = adapt(ARRAY_TYPE, u_host)
        f = adapt(ARRAY_TYPE, zeros(n_local))
        b = fill!(similar(f), 0.0)
        ws = BiCGStabWorkspace(b)
        A = DistributedOperator(ConstrainedOperator(K, bc; n_own = n_own), halo;
                                accumulate = accumulate)
        (; part, n_own, n_local, pattern, data, K, ke, halo, bc, u, f, b, ws, A,
         accumulate, space = MPISpace(COMM, n_own))
    end
    s = setup

    assemble() = (compute_elements!(s.ke, s.pattern, element, s.data);
                  assemble!(s.K, s.pattern, s.ke))
    function solve(itmax)
        constrained_rhs!(s.b, s.K, s.bc, s.u, s.f)
        s.accumulate && accumulate_ghosts!(s.halo, s.b)
        jacobi!(s.ws, s.K, s.bc; halo = s.halo, accumulate = s.accumulate, n_own = s.n_own)
        x, stats = solve_bicgstab!(s.ws, s.A, s.b, s.space; rtol = RTOL, itmax = itmax)
        apply_prescribed!(x, s.bc, s.u)
        return x, stats
    end

    assemble()                       # warm-up (compilation)
    solve(3)
    reset_timer!()
    _, t_assembly = timed(assemble)
    (result, t_solve) = timed(() -> solve(ITMAX))
    x, stats = result
    t_comm = MPI.Allreduce(section_time("solver (BiCGStab)", "halo exchange") +
                           section_time("solver (BiCGStab)", "halo accumulate"), max, COMM)

    # --- checks without a global reference -------------------------------------------------
    y = fill!(similar(s.b), 0.0)
    mul!(y, s.A, x)
    rhs_norm = sqrt(inner(s.space, s.b, s.b))
    y .-= s.b
    y .= ifelse.(s.bc.mask, 0.0, y)      # residual of the free degrees of freedom
    residual = sqrt(inner(s.space, y, y)) / max(rhs_norm, eps())
    x_host = Array(x)
    u_host = Array(s.u)
    mask = Array(s.bc.mask)
    exact = MPI.Allreduce(all(x_host[i] == u_host[i] for i in 1:(s.n_own) if mask[i]), &,
                          COMM)

    memory = data_bytes((s.K, s.pattern.ke_ptr, s.pattern.src_ptr, s.pattern.src, s.ke,
                         s.data, s.bc, s.u, s.f, s.b, s.ws, s.halo))
    mem_max = MPI.Allreduce(memory, max, COMM) / 2^20
    mem_total = MPI.Allreduce(memory, +, COMM) / 2^20
    dofs = MPI.Allreduce(s.n_own, +, COMM)
    dofs_max = MPI.Allreduce(s.n_own, max, COMM)
    n_gpu = MPI.Allreduce(Int(ON_GPU), +, COMM)

    if RANK == 0
        header = "timestamp,dim,element,backend,ranks,gpu_ranks,threads,dofs," *
                 "dofs_per_rank_max,nx,ny,nz,horizon,t_setup,t_assembly,t_solve," *
                 "iterations,t_iteration,t_comm,mem_rank_max_MB,mem_total_MB,residual," *
                 "prescribed_exact,solved"
        line = join(Any[format(now(), "yyyy-mm-ddTHH:MM:SS"), DIM, ELEMENT, BACKEND,
                        N_RANKS, n_gpu, Threads.nthreads(), dofs, dofs_max, dims..., M,
                        t_setup, t_assembly, t_solve, stats.niter,
                        t_solve / max(stats.niter, 1), t_comm, mem_max, mem_total,
                        residual, exact, stats.solved], ",")
        mkpath(dirname(abspath(CSV_PATH)))
        new_file = !isfile(CSV_PATH)
        open(CSV_PATH, "a") do io
            new_file && println(io, header)
            println(io, line)
        end
        @printf("%dD %-4s %-5s ranks=%d (gpu %d) dofs=%d | setup %.3fs assembly %.4fs solve %.3fs (%d it, %.3f ms/it, comm %.3fs) | %.2f MB/rank | res %.1e %s\n",
                DIM, ELEMENT, BACKEND, N_RANKS, n_gpu, dofs, t_setup, t_assembly,
                t_solve, stats.niter, 1e3 * t_solve / max(stats.niter, 1), t_comm,
                mem_max, residual,
                stats.solved && exact && residual < 10 * RTOL ? "OK" : "CHECK")
    end
    return stats.solved && exact && residual < 10 * RTOL
end

ok = run_case()
MPI.Barrier(COMM)
exit(ok ? 0 : 1)
