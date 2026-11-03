# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Guyan

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: LinearAlgebra, lu, ldiv!, mul!, Diagonal
using SparseArrays: SparseMatrixCSC, sparse
using ..Sparse_CSR: CSRMatrix, launch!
using ..Dof_Sets: DofSet, dof_indices
using ..Operators: ConstrainedOperator, SubOperator
using ..Linear_Solver: LinearSolverWorkspace, solve_linear!

export model_reduction_name, extract, reduce_matrices, SchurOperator

model_reduction_name() = "Static Condensation"

"""
    extract(K, rows, cols)

Explicit copy of the block `K[rows, cols]` into a host `SparseMatrixCSC` (README A4a.4).
This is the only function that allocates a sub-matrix. It is used where a method needs
one (factorisation of `K_ss`, eigenvalue problem of Craig-Bampton) and is never called by
boundary conditions.

# Arguments
- `K::CSRMatrix`: Full matrix (host or device)
- `rows::AbstractVector{Int64}`, `cols::AbstractVector{Int64}`: Index lists
"""
function extract(K::CSRMatrix, rows::AbstractVector{Int64}, cols::AbstractVector{Int64})
    rowptr = Array(K.rowptr)
    colval = Array(K.colval)
    nzval = Array(K.nzval)
    new_col = zeros(Int64, K.n_cols)
    new_col[cols] = 1:length(cols)
    I = Int64[]
    J = Int64[]
    V = Float64[]
    for (ir, r) in enumerate(rows), k in rowptr[r]:(rowptr[r + 1] - 1)
        jc = new_col[colval[k]]
        jc == 0 && continue
        push!(I, ir)
        push!(J, jc)
        push!(V, nzval[k])
    end
    return sparse(I, J, V, length(rows), length(cols))
end

"""
    reduce_matrices(K, M_diag, masters)

Guyan reduction (static condensation) with the interface of PeriLab
`Model_reduction/Guyan.jl`, without assuming symmetry:

    T = −K_ss⁻¹ K_sm,   K_r = K_mm + K_ms T,   M_r = M_mm + Tᵀ M_ss T

Uses `extract` (explicit sub-matrices) and a direct factorisation of `K_ss`.

# Returns
- `K_r::Matrix{Float64}`, `M_r::Matrix{Float64}`: in the order of `dof_indices(masters)`
"""
function reduce_matrices(K::CSRMatrix, M_diag::AbstractVector{Float64}, masters::DofSet)
    m = dof_indices(masters)
    s = setdiff(1:(K.n_rows), m)
    K_ss = extract(K, s, s)
    T = Matrix(extract(K, s, m))
    ldiv!(lu(K_ss), T)
    T .*= -1.0
    K_r = Matrix(extract(K, m, m)) + extract(K, m, s) * T
    M_host = Array(M_diag)
    M_r = Matrix(Diagonal(M_host[m])) + T' * Diagonal(M_host[s]) * T
    return K_r, M_r
end

"""
    SchurOperator(K, masters; memory = 50, rtol = 1e-12)

Matrix-free Guyan operator on full-length vectors (README A4a.3):

    y_m = (K_mm − K_ms K_ss⁻¹ K_sm) x_m,   y_s = x_s

`K_ss⁻¹` is applied by an inner GMRES on `ConstrainedOperator(K, masters)`, which is
`K_ss` with the identity on the masters. No sub-matrix is formed; the work vectors and
the inner Krylov workspace are allocated once.
"""
struct SchurOperator{M<:CSRMatrix,S<:DofSet,V<:AbstractVector,W}
    K::M
    masters::S
    t::V
    z::V
    w::V
    inner::W
    rtol::Float64
end
function SchurOperator(K::CSRMatrix, masters::DofSet; memory::Int64 = 50,
                       rtol::Float64 = 1e-12)
    t = similar(K.nzval, K.n_rows)
    z = similar(t)
    w = similar(t)
    inner = LinearSolverWorkspace(K.n_rows, typeof(t); memory = memory)
    return SchurOperator(K, masters, t, z, w, inner, rtol)
end

Base.size(A::SchurOperator) = size(A.K)
Base.size(A::SchurOperator, d::Integer) = size(A.K, d)
Base.eltype(A::SchurOperator) = eltype(A.K)

@kernel function schur_combine_kernel!(y, @Const(masters), @Const(x), @Const(kmm_x),
                                       @Const(kms_z))
    i = @index(Global)
    @inbounds y[i] = masters[i] ? kmm_x[i] - kms_z[i] : x[i]
end

function LinearAlgebra.mul!(y::AbstractVector, A::SchurOperator, x::AbstractVector)
    m = A.masters
    mul!(A.t, SubOperator(A.K, m, m; complement_rows = true), x)        # K_sm x
    z, _ = solve_linear!(A.inner, ConstrainedOperator(A.K, m), A.t; rtol = A.rtol)
    mul!(A.w, SubOperator(A.K, m, m; complement_cols = true), z)        # K_ms z
    mul!(A.z, SubOperator(A.K, m, m), x)                                # K_mm x
    launch!(schur_combine_kernel!, y, length(y), y, m.mask, x, A.z, A.w)
    return y
end

end
