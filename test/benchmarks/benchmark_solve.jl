# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Lösen mit Dirichlet-RB am 1D-bond-based-Stab (u = 0 links, u = 100 rechts, F Mitte).
#   Referenz:  b = f − K[:,p] u_p;  K_ff = K[free, free] (Kopie);  u_f = K_ff \ b_f (UMFPACK)
#   PD_Matrix: maskierter Operator + GMRES, keine Teilmatrix (CPU und GPU)
#
#   julia --project=test/benchmarks -t 1 (bzw. -t 4, Maximum in WSL) test/benchmarks/benchmark_solve.jl

using PD_Matrix
using Adapt: adapt
using ArraysOfArrays: VectorOfVectors
using LinearAlgebra: mul!, norm
using SparseArrays: sparse, nnz
using Printf: @printf
using CUDA

include(joinpath(@__DIR__, "benchmark_bar.jl"))   # bar_data, reference_assembly, (Tabelle 1)

function reference_solve(K, p, u_p, f)
    n = size(K, 1)
    free = setdiff(1:n, p)
    b = f[free] - K[free, p] * u_p
    K_ff = K[free, free]
    u = zeros(n)
    u[p] = u_p
    u[free] = K_ff \ b
    return u
end

function own_solve!(workspace, K, bc, u_full, f, b, r; rtol)
    constrained_rhs!(b, K, bc, u_full, f)
    u, stats = solve_linear!(workspace, ConstrainedOperator(K, bc), b; rtol = rtol,
                             itmax = 20_000)
    apply_prescribed!(u, bc, u_full)
    return u, stats
end

function run_solve(n; memory = 50, rtol = 1e-10)
    data = bar_data(n)
    p = [1, n]; u_p = [0.0, 100.0]
    f = zeros(n); f[div(n, 2)] = 10.0
    Kref = reference_assembly(data, n)

    reference_solve(Kref, p, u_p, f)
    ref = @timed reference_solve(Kref, p, u_p, f)
    free = setdiff(1:n, p)
    slice_bytes = @allocated Kref[free, free]

    el = BondbasedElastic(1)
    rows, cols = element_dofs(el, data)
    host_pattern = create_assembly_pattern(rows, cols, n)
    results = Dict{String,Any}()
    for (name, AT) in (("CPU", Array), ("GPU", CuArray))
        AT === CuArray && !CUDA.functional() && continue
        pat = adapt(AT, host_pattern); d = adapt(AT, data)
        K = create_matrix(pat); ke = create_element_buffer(pat)
        compute_elements!(ke, pat, el, d); assemble!(K, pat, ke)
        bc = adapt(AT, create_dof_set(p, n))
        u_full = zeros(n); u_full[p] = u_p
        u_full = adapt(AT, u_full); fd = adapt(AT, f)
        b = similar(fd); r = similar(fd)
        ws = LinearSolverWorkspace(n, typeof(b); memory = memory)
        own_solve!(ws, K, bc, u_full, fd, b, r; rtol = rtol)
        stat = AT === CuArray ? CUDA.@timed(own_solve!(ws, K, bc, u_full, fd, b, r; rtol = rtol)) :
               @timed(own_solve!(ws, K, bc, u_full, fd, b, r; rtol = rtol))
        u, stats = stat.value
        err = norm(Array(u) - ref.value) / norm(ref.value)
        bytes = AT === CuArray ? stat.cpu_bytes + stat.gpu_bytes : stat.bytes
        results[name] = (t = stat.time * 1e3, it = stats.niter, ok = stats.solved, err = err,
                         mb = bytes / 2^20)
    end
    @printf("%8d | %9.2f %9.2f %8.2f | ", n, ref.time * 1e3, ref.bytes / 2^20,
            slice_bytes / 2^20)
    for name in ("CPU", "GPU")
        haskey(results, name) || continue
        s = results[name]
        @printf("%s %9.1f ms %7d it %5s err %.1e %7.3f MB | ", name, s.t, s.it, s.ok, s.err,
                s.mb)
    end
    println()
    flush(stdout)
end

println("\nLösen mit RB, Threads: ", Threads.nthreads())
println("       n | Referenz ms  alloc MB  K_ff MB | PD_Matrix (GMRES(50), rtol 1e-10, ohne Vorkonditionierer)")
for n in (100, 1_000, 10_000)
    run_solve(n)
end
