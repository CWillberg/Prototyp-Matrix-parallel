# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Correspondence_Elastic

using ArraysOfArrays: VectorOfVectors
using LinearAlgebra: dot
using StaticArrays: SVector, SMatrix, @SMatrix
using ..Element_Interface: AbstractElement
import ..Element_Interface: element_name, element_dofs, compute_element!, n_elements

export CorrespondenceElastic

"""
    CorrespondenceElastic{D}(lambda, mu, stabilization)

Linear elastic correspondence element in `D` dimensions (small strains). The element of
point `i` (README A3.6):

    K_i = Σ_j ω_ij V_j ξ_ij ⊗ ξ_ij                     shape tensor
    g_j = ω_ij V_j K_i⁻¹ ξ_ij,   G = Σ_j g_j
    H_i = Σ_j (u_j − u_i) ⊗ g_j                         displacement gradient
    σ_i = λ tr(ε) I + 2μ ε,  ε = sym(H_i)
    bond force  t_j = ω_ij σ_i K_i⁻¹ ξ_ij :   f_i += t_j V_j,   f_j −= t_j V_i

plus a zero-energy-mode stabilisation on the non-affine part of every bond

    z_j = u_j − u_i − H_i ξ_ij,   p_j = β ω_ij z_j :   f_i += p_j V_j,   f_j −= p_j V_i

(`β = stabilization`, zero for affine fields). The element matrix is the stiffness
`K = −∂f/∂u` (same sign convention as `BondbasedElastic`, `K u = −f_int`). It writes into the rows of `i` **and of its
neighbours**, columns are `i` and its neighbours. Assembled, a row couples over two
horizons; with one-horizon ghosts the neighbour rows on other ranks are partial sums that
are added to their owners in the solver (README A4.0b).

Uses the same data as `create_bondbased_data` (`bond_constant` is not used).
"""
struct CorrespondenceElastic{D} <: AbstractElement
    lambda::Float64
    mu::Float64
    stabilization::Float64
end
function CorrespondenceElastic(dim::Int64; lambda::Float64, mu::Float64,
                               stabilization::Float64 = 0.0)
    return CorrespondenceElastic{dim}(lambda, mu, stabilization)
end

element_name(::Type{<:CorrespondenceElastic}) = "Correspondence Elastic"

n_elements(::CorrespondenceElastic, data) = length(data.nlist_ptr) - 1

point_dofs(i::Int64, D::Int64) = ((i - 1) * D + 1):(i * D)

function element_dofs(::CorrespondenceElastic{D}, data) where {D}
    nlist_ptr = Array(data.nlist_ptr)
    nlist = Array(data.nlist)
    rows = VectorOfVectors{Int64}()
    for i in 1:(length(nlist_ptr) - 1)
        dofs = collect(point_dofs(i, D))
        for b in nlist_ptr[i]:(nlist_ptr[i + 1] - 1)
            append!(dofs, point_dofs(nlist[b], D))
        end
        push!(rows, dofs)
    end
    return rows, rows
end

@inline function bond_vector(x, i, j, ::Val{D}) where {D}
    return SVector{D,Float64}(ntuple(a -> x[a, j] - x[a, i], Val(D)))
end

@inline function compute_element!(el::CorrespondenceElastic{D}, ke, start, data,
                                  i) where {D}
    x = data.coordinates
    first_bond = data.nlist_ptr[i]
    n_bonds = data.nlist_ptr[i + 1] - first_bond
    n_r = D * (n_bonds + 1)
    for q in 0:(n_r * n_r - 1)
        ke[start + q] = 0.0
    end
    shape = zero(SMatrix{D,D,Float64})
    for b in first_bond:(first_bond + n_bonds - 1)
        xi = bond_vector(x, i, data.nlist[b], Val(D))
        shape += data.bond_damage[b] * data.volume[data.nlist[b]] * (xi * xi')
    end
    shape_inv = inv(shape)
    G = zero(SVector{D,Float64})
    for b in first_bond:(first_bond + n_bonds - 1)
        j = data.nlist[b]
        G += data.bond_damage[b] * data.volume[j] *
             (shape_inv * bond_vector(x, i, j, Val(D)))
    end
    V_i = data.volume[i]
    identity = SMatrix{D,D,Float64}(ntuple(k -> ((k - 1) % D == (k - 1) ÷ D) ? 1.0 : 0.0,
                                           Val(D * D)))
    # column block cb = 0 is point i, cb = k the k-th neighbour; column component c
    for cb in 0:n_bonds, c in 1:D
        g_k = if cb == 0
            -G
        else
            k = data.nlist[first_bond + cb - 1]
            data.bond_damage[first_bond + cb - 1] * data.volume[k] *
            (shape_inv * bond_vector(x, i, k, Val(D)))
        end
        e_c = SVector{D,Float64}(ntuple(a -> a == c ? 1.0 : 0.0, Val(D)))
        dH = e_c * g_k'
        d_eps = 0.5 * (dH + dH')
        d_sigma = el.lambda * g_k[c] * identity + 2.0 * el.mu * d_eps
        column = start + (cb * D + c - 1) * n_r - 1
        df_i = d_sigma * G
        for jj in 1:n_bonds
            b = first_bond + jj - 1
            j = data.nlist[b]
            omega = data.bond_damage[b]
            xi = bond_vector(x, i, j, Val(D))
            g_j = omega * data.volume[j] * (shape_inv * xi)
            df_j = -(V_i / data.volume[j]) * (d_sigma * g_j)
            # stabilisation: dz_j = e_c (δ_jk − δ_ik) − dH ξ_j
            s = (cb == jj ? 1.0 : 0.0) - (cb == 0 ? 1.0 : 0.0)
            dz = s * e_c - dH * xi
            p = el.stabilization * omega * dz
            df_i += p * data.volume[j]
            df_j -= p * V_i
            for a in 1:D
                ke[column + jj * D + a] -= df_j[a]   # K = −∂f/∂u
            end
        end
        for a in 1:D
            ke[column + a] -= df_i[a]
        end
    end
    return nothing
end

end
