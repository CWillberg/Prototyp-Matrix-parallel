# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Zeit und Speicherallokation je Abschnitt (TimerOutputs) für den 1D-bond-based-Stab.
# Alle Puffer und der Krylov-Workspace werden vorab angelegt; gemessen wird nur die
# numerische Phase und das Lösen (zweiter Durchlauf, nach dem Kompilieren).
#
#   julia --project=test/benchmarks -t 1 test/benchmarks/profile_bar.jl [n] [gpu]

using PD_Matrix
using Adapt: adapt
using ArraysOfArrays: VectorOfVectors
using TimerOutputs: reset_timer!, print_timer, @timeit
using CUDA

include(joinpath(@__DIR__, "benchmark_bar.jl"))   # bar_data

n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1_000
array_type = ("gpu" in ARGS && CUDA.functional()) ? CuArray : Array

data = bar_data(n)
el = BondbasedElastic(1)
p = [1, n]
u_full = zeros(n); u_full[p] = [0.0, 100.0]
f = zeros(n); f[div(n, 2)] = 10.0

# --- Vorbereitung (einmal, darf allozieren) --------------------------------------
rows, cols = element_dofs(el, data)
host_pattern = create_assembly_pattern(rows, cols, n)
pattern = adapt(array_type, host_pattern)
d = adapt(array_type, data)
K = create_matrix(pattern)
ke = create_element_buffer(pattern)
bc = adapt(array_type, create_dof_set(p, n))
u_dev = adapt(array_type, u_full); f_dev = adapt(array_type, f)
b = similar(f_dev); r = similar(f_dev)
workspace = LinearSolverWorkspace(n, typeof(b); memory = 50)

function step!()
    @timeit "numerical phase" begin
        compute_elements!(ke, pattern, el, d)
        assemble!(K, pattern, ke)
    end
    @timeit "solve with BCs" begin
        constrained_rhs!(b, K, bc, u_dev, f_dev)
        u, stats = solve_linear!(workspace, ConstrainedOperator(K, bc), b; rtol = 1e-8,
                                 itmax = 2_000)
        apply_prescribed!(u, bc, u_dev)
        reactions!(r, K, bc, u, f_dev)
    end
    return stats
end

step!()                 # kompilieren
reset_timer!()
step!()                 # legt die Timer-Abschnitte an (TimerOutputs alloziert dabei selbst)
bytes_first = @allocated step!()
bytes_second = @allocated step!()
stats = step!()
println("n = $n, backend = $array_type, threads = $(Threads.nthreads()), ",
        "GMRES: solved = $(stats.solved), iterations = $(stats.niter)")
println("@allocated für einen kompletten Schritt (Assemblierung + Lösen): ",
        bytes_first, " B bzw. ", bytes_second, " B")
print_timer(; sortby = :firstexec)
println()
