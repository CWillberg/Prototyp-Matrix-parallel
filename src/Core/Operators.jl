# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Operators

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: LinearAlgebra
using TimerOutputs: @timeit
using ..Sparse_CSR: CSRMatrix, launch!
using ..Dof_Sets: DofSet

export ConstrainedOperator, SubOperator, constrained_rhs!, apply_prescribed!, reactions!

"""
    ConstrainedOperator(K, prescribed)

Operator of the system with prescribed degrees of freedom `p` and free ones `f`, applied
directly on the full matrix without forming sub-matrices (README A6.1, A6.4):

    y = M_f K M_f x + M_p x

i.e. `K_ff` on the free part and the identity on the prescribed part. The matrix stays
unchanged, so the prescribed set can change between load steps (README A6.3).

The same operator restricted to the complement of a master set is `K_ss` of a model
reduction (README A4a.3).
"""
struct ConstrainedOperator{M<:CSRMatrix,S<:DofSet}
    K::M
    prescribed::S
end

Base.size(A::ConstrainedOperator) = size(A.K)
Base.size(A::ConstrainedOperator, d::Integer) = size(A.K, d)
Base.eltype(A::ConstrainedOperator) = eltype(A.K)

@kernel function constrained_spmv_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval),
                                          @Const(prescribed), @Const(x))
    i = @index(Global)
    if prescribed[i]
        @inbounds y[i] = x[i]
    else
        s = zero(eltype(y))
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
            c = colval[k]
            s += ifelse(prescribed[c], zero(s), nzval[k] * x[c])
        end
        @inbounds y[i] = s
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::ConstrainedOperator, x::AbstractVector)
    K = A.K
    @timeit "SpMV constrained" launch!(constrained_spmv_kernel!, y, K.n_rows, y, K.rowptr,
                                       K.colval, K.nzval, A.prescribed.mask, x)
    return y
end

"""
    SubOperator(K, rows, cols)

Applies the block `K_ab` of the full matrix on full-length vectors without copying it
(README A4a.3):

    y_i = Σ_{j ∈ b} K_ij x_j   for i ∈ a,   y_i = 0 otherwise

`rows`/`cols` are `DofSet`s; `complement_rows`/`complement_cols` use the complement of the
set instead, so `K_ms`, `K_sm`, `K_mm` of a reduction need only the master set.
"""
struct SubOperator{M<:CSRMatrix,S<:DofSet}
    K::M
    rows::S
    cols::S
    complement_rows::Bool
    complement_cols::Bool
end
function SubOperator(K::CSRMatrix, rows::DofSet, cols::DofSet;
                     complement_rows::Bool = false, complement_cols::Bool = false)
    return SubOperator(K, rows, cols, complement_rows, complement_cols)
end

Base.size(A::SubOperator) = size(A.K)
Base.size(A::SubOperator, d::Integer) = size(A.K, d)
Base.eltype(A::SubOperator) = eltype(A.K)

@kernel function sub_spmv_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval),
                                  @Const(rows), @Const(cols), flip_rows::Bool,
                                  flip_cols::Bool, @Const(x))
    i = @index(Global)
    s = zero(eltype(y))
    if rows[i] != flip_rows
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
            c = colval[k]
            s += ifelse(cols[c] != flip_cols, nzval[k] * x[c], zero(s))
        end
    end
    @inbounds y[i] = s
end

function LinearAlgebra.mul!(y::AbstractVector, A::SubOperator, x::AbstractVector)
    K = A.K
    @timeit "SpMV sub-block" launch!(sub_spmv_kernel!, y, K.n_rows, y, K.rowptr, K.colval,
                                     K.nzval, A.rows.mask, A.cols.mask, A.complement_rows,
                                     A.complement_cols, x)
    return y
end

@kernel function rhs_kernel!(b, @Const(rowptr), @Const(colval), @Const(nzval),
                             @Const(prescribed), @Const(u), @Const(f))
    i = @index(Global)
    if prescribed[i]
        @inbounds b[i] = zero(eltype(b))
    else
        s = f[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
            c = colval[k]
            s -= ifelse(prescribed[c], nzval[k] * u[c], zero(s))
        end
        @inbounds b[i] = s
    end
end

"""
    constrained_rhs!(b, K, prescribed, u, f)

Right-hand side for `ConstrainedOperator` (README A6.4), homogeneous form:

    b = M_f (f − K M_p u)

`u` holds the prescribed values at the prescribed degrees of freedom (other entries are
ignored). One fused kernel, no temporary vector. Because `b_p = 0` and the operator is the
identity on the prescribed part, every Krylov vector is exactly zero there; the prescribed
values are inserted exactly afterwards with `apply_prescribed!`.
"""
function constrained_rhs!(b::AbstractVector, K::CSRMatrix, prescribed::DofSet,
                          u::AbstractVector, f::AbstractVector)
    @timeit "rhs" launch!(rhs_kernel!, b, K.n_rows, b, K.rowptr, K.colval, K.nzval,
                          prescribed.mask, u, f)
    return b
end

@kernel function apply_prescribed_kernel!(x, @Const(prescribed), @Const(u))
    i = @index(Global)
    @inbounds x[i] = prescribed[i] ? u[i] : x[i]
end

"""
    apply_prescribed!(x, prescribed, u)

Inserts the prescribed values `u` into the solution `x` (exact, no rounding).
"""
function apply_prescribed!(x::AbstractVector, prescribed::DofSet, u::AbstractVector)
    @timeit "apply prescribed" launch!(apply_prescribed_kernel!, x, length(x), x,
                                       prescribed.mask, u)
    return x
end

@kernel function reaction_kernel!(r, @Const(rowptr), @Const(colval), @Const(nzval),
                                  @Const(prescribed), @Const(u), @Const(f))
    i = @index(Global)
    s = zero(eltype(r))
    if prescribed[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
            s += nzval[k] * u[colval[k]]
        end
        @inbounds s -= f[i]
    end
    @inbounds r[i] = s
end

"""
    reactions!(r, K, prescribed, u, f)

Reaction forces at the prescribed degrees of freedom from the full matrix
(README A6.5): `r = M_p (K u − f)`.
"""
function reactions!(r::AbstractVector, K::CSRMatrix, prescribed::DofSet,
                    u::AbstractVector, f::AbstractVector)
    @timeit "reactions" launch!(reaction_kernel!, r, K.n_rows, r, K.rowptr, K.colval,
                                K.nzval, prescribed.mask, u, f)
    return r
end

end
