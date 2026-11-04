# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Bondbased_Elastic

using ArraysOfArrays: VectorOfVectors, flatview, internal_element_ptr
using ..Element_Interface: AbstractElement
import ..Element_Interface: element_name, element_dofs, compute_element!, n_elements

export BondbasedElastic, create_bondbased_data

"""
    BondbasedElastic{D}

Linearised bond-based elastic element in `D` dimensions (README A3.5). The element of
point `i` writes only into the rows of `i`:

    k_ij = c_i ω_ij V_j / |ξ_ij| (e_ij ⊗ e_ij),   K[i,i] += k_ij,   K[i,j] −= k_ij

With different volumes `V_j` or bond damage the matrix is not symmetric.
"""
struct BondbasedElastic{D} <: AbstractElement end
BondbasedElastic(dim::Int64) = BondbasedElastic{dim}()

element_name(::Type{<:BondbasedElastic}) = "Bond-based Elastic"

"""
    create_bondbased_data(coordinates, volume, nlist, bond_damage, bond_constant)

Collects the field data of the element as flat arrays (README A0.3, A8.1). The caller
delivers the data of the own points and their ghost neighbours.

# Arguments
- `coordinates::AbstractMatrix{Float64}`: `D × n_points`
- `volume::AbstractVector{Float64}`: Volume per point
- `nlist::VectorOfVectors{Int64}`: Neighbours per point
- `bond_damage::VectorOfVectors{Float64}`: Bond damage, same layout as `nlist`
- `bond_constant::AbstractVector{Float64}`: Bond constant `c` per point
# Returns
- `NamedTuple` of flat arrays; move it to a device with `Adapt.adapt`
"""
function create_bondbased_data(coordinates::AbstractMatrix{Float64},
                               volume::AbstractVector{Float64},
                               nlist::VectorOfVectors{Int64},
                               bond_damage::VectorOfVectors{Float64},
                               bond_constant::AbstractVector{Float64})
    nlist_ptr = internal_element_ptr(nlist)
    if internal_element_ptr(bond_damage) != nlist_ptr
        throw(ArgumentError("bond_damage must have the layout of nlist."))
    end
    n_points = size(coordinates, 2)
    if length(volume) != n_points || length(bond_constant) != n_points
        throw(ArgumentError("volume and bond_constant need one entry per point."))
    end
    check_bond_lengths(coordinates, nlist)
    return (coordinates = coordinates, volume = volume,
            nlist_ptr = collect(Int64, nlist_ptr), nlist = collect(flatview(nlist)),
            bond_damage = collect(flatview(bond_damage)), bond_constant = bond_constant)
end

function check_bond_lengths(coordinates::AbstractMatrix{Float64},
                            nlist::VectorOfVectors{Int64})
    for (i, neighbors) in enumerate(nlist), j in neighbors
        same = true
        for a in axes(coordinates, 1)
            same &= coordinates[a, i] == coordinates[a, j]
        end
        same && throw(ArgumentError("Bond $(i)-$(j) has zero length."))
    end
end

n_elements(::BondbasedElastic, data) = length(data.nlist_ptr) - 1

point_dofs(i::Int64, D::Int64) = ((i - 1) * D + 1):(i * D)

function element_dofs(::BondbasedElastic{D}, data) where {D}
    nlist_ptr = Array(data.nlist_ptr)
    nlist = Array(data.nlist)
    n_points = length(nlist_ptr) - 1
    rows = VectorOfVectors{Int64}()
    cols = VectorOfVectors{Int64}()
    for i in 1:n_points
        push!(rows, collect(point_dofs(i, D)))
        col = collect(point_dofs(i, D))
        for b in nlist_ptr[i]:(nlist_ptr[i + 1] - 1)
            append!(col, point_dofs(nlist[b], D))
        end
        push!(cols, col)
    end
    return rows, cols
end

@inline function compute_element!(::BondbasedElastic{D}, ke, start, data, i) where {D}
    x = data.coordinates
    first_bond = data.nlist_ptr[i]
    n_bonds = data.nlist_ptr[i + 1] - first_bond
    n_values = D * D * (n_bonds + 1)
    for q in 0:(n_values - 1)
        ke[start + q] = zero(eltype(ke))
    end
    for jj in 1:n_bonds
        b = first_bond + jj - 1
        j = data.nlist[b]
        length_sq = zero(eltype(ke))
        for a in 1:D
            length_sq += (x[a, j] - x[a, i])^2
        end
        bond_length = sqrt(length_sq)
        k = data.bond_constant[i] * data.bond_damage[b] * data.volume[j] /
            (bond_length * length_sq)
        for cb in 1:D, ra in 1:D
            kab = k * (x[ra, j] - x[ra, i]) * (x[cb, j] - x[cb, i])
            ke[start + (cb - 1) * D + ra - 1] += kab
            ke[start + (jj * D + cb - 1) * D + ra - 1] -= kab
        end
    end
    return nothing
end

end
