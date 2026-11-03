# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Linear_Solver

using Krylov: GmresWorkspace, krylov_solve!, solution, statistics
using TimerOutputs: @timeit

export LinearSolverWorkspace, solve_linear!

"""
    LinearSolverWorkspace(n, vector_type; memory = 50)

Preallocated Krylov workspace (GMRES, README A5.1). `vector_type` decides the backend,
e.g. `Vector{Float64}` or `CuVector{Float64}`; the workspace is reused for every solve.
"""
struct LinearSolverWorkspace{W}
    gmres::W
end
function LinearSolverWorkspace(n::Int64, vector_type::Type; memory::Int64 = 50)
    return LinearSolverWorkspace(GmresWorkspace(n, n, vector_type; memory = memory))
end

"""
    solve_linear!(workspace, A, b; rtol, atol, itmax, preconditioner)

Solves `A x = b` with restarted GMRES. `A` is any operator with `size`, `eltype` and
`mul!` (e.g. `ConstrainedOperator`). `preconditioner` is passed to Krylov as left
preconditioner `M` (README A5.3, selection postponed, F7).

# Returns
- `x`: Solution vector (owned by the workspace, copy it to keep it)
- `stats`: Krylov statistics (`stats.solved`, `stats.niter`)
"""
function solve_linear!(workspace::LinearSolverWorkspace, A, b::AbstractVector;
                       rtol::Float64 = 1e-10, atol::Float64 = 0.0, itmax::Int64 = 0,
                       restart::Bool = true, preconditioner = nothing)
    @timeit "solver (Krylov GMRES)" begin
        if isnothing(preconditioner)
            krylov_solve!(workspace.gmres, A, b; rtol = rtol, atol = atol, itmax = itmax,
                          restart = restart)
        else
            krylov_solve!(workspace.gmres, A, b; M = preconditioner, rtol = rtol,
                          atol = atol, itmax = itmax, restart = restart)
        end
    end
    return solution(workspace.gmres), statistics(workspace.gmres)
end

end
