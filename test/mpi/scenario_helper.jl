# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Emulates what PeriLab delivers to a rank (README A0.3): own points, ghost points with
# their owner ranks and the field data of both. The prototype itself never sees the global
# problem; here every rank builds it only to cut out its part and to compute a serial
# reference for comparison.

using ArraysOfArrays: VectorOfVectors
using LinearAlgebra: norm

"""
    distribute_points(coordinates, volume, horizon, rank, n_ranks)

Block distribution of the points (rank-sorted global numbering, README A0.4).
"""
function distribute_points(coordinates::Matrix{Float64}, volume::Vector{Float64},
                           horizon::Float64, rank::Int64, n_ranks::Int64;
                           bond_constant::Float64 = 1.0)
    n = size(coordinates, 2)
    bounds = [div(r * n, n_ranks) for r in 0:n_ranks]
    owner(g) = searchsortedlast(bounds, g - 1) - 1
    own = collect((bounds[rank + 1] + 1):bounds[rank + 2])
    global_nlist = neighbor_list(coordinates, horizon)

    ghost_set = Set{Int64}()
    for g in own, j in global_nlist[g]
        owner(j) == rank || push!(ghost_set, j)
    end
    ghosts = sort(collect(ghost_set))            # sorted ids = grouped by owner rank
    ghost_owner = [owner(g) for g in ghosts]
    local_points = vcat(own, ghosts)
    local_of = Dict(g => i for (i, g) in enumerate(local_points))

    nlist = VectorOfVectors([[local_of[j] for j in global_nlist[g]] for g in own])
    damage = VectorOfVectors([ones(length(l)) for l in nlist])
    data = create_bondbased_data(coordinates[:, local_points], volume[local_points], nlist,
                                 damage, fill(bond_constant, length(local_points)))
    return (data = data, own = own, ghosts = ghosts, ghost_owner = ghost_owner,
            local_points = local_points, n_global = n, global_nlist = global_nlist)
end

"Local degree of freedom → global degree of freedom (point-wise numbering)."
local_to_global_dofs(part, D) = [(g - 1) * D + d for g in part.local_points for d in 1:D]

"""
    solve_distributed(part, D, prescribed, values, f; element, array_type, comm, rtol)

Full distributed pipeline of one rank (README A4.0b, one-horizon ghosts):

- local symbolic phase over the own elements; the local matrix has the own rows and the
  ghost rows (partial sums, non-empty only for elements that write into neighbour rows),
- numerical phase on the backend, no communication,
- solver: ghost values of x forward, local SpMV, partial ghost rows back to the owners
  (only if some rank has ghost rows),
- right-hand side, Jacobi diagonal and reactions are accumulated the same way.

Global input (`prescribed`, `values`, `f` in global dof numbering) is cut down to the
rank's local degrees of freedom here, as the caller (PeriLab) would do.
"""
function solve_distributed(part, D::Int64, prescribed::Vector{Int64},
                           values::Vector{Float64}, f_global::Vector{Float64};
                           element = BondbasedElastic(D), array_type = Array,
                           comm = MPI.COMM_WORLD, rtol = 1e-12, itmax = 50_000)
    l2g = local_to_global_dofs(part, D)
    n_own = D * length(part.own)
    n_local = length(l2g)

    rows, cols = element_dofs(element, part.data)
    host_pattern = create_assembly_pattern(rows, cols, n_local, n_local)
    accumulate = needs_accumulate(comm, host_pattern.rowptr, n_own)
    pattern = adapt(array_type, host_pattern)
    data = adapt(array_type, part.data)
    K = create_matrix(pattern)
    ke = create_element_buffer(pattern)
    compute_elements!(ke, pattern, element, data)
    assemble!(K, pattern, ke)

    value_of = Dict(zip(prescribed, values))
    local_prescribed = [i for (i, g) in enumerate(l2g) if haskey(value_of, g)]
    u_host = zeros(n_local)
    for i in local_prescribed
        u_host[i] = value_of[l2g[i]]
    end
    f_host = zeros(n_local)                     # loads only on own rows
    f_host[1:n_own] = f_global[l2g[1:n_own]]

    bc = adapt(array_type, create_dof_set(local_prescribed, n_local))
    u = adapt(array_type, u_host)
    f = adapt(array_type, f_host)
    halo = create_halo_exchange(comm, part.own, part.ghosts, part.ghost_owner, D;
                                array_type = array_type)
    A = DistributedOperator(ConstrainedOperator(K, bc; n_own = n_own), halo;
                            accumulate = accumulate)
    space = MPISpace(comm, n_own)

    b = fill!(similar(f), 0.0)
    constrained_rhs!(b, K, bc, u, f)
    accumulate && accumulate_ghosts!(halo, b)
    ws = BiCGStabWorkspace(b)
    jacobi!(ws, K, bc; halo = halo, accumulate = accumulate, n_own = n_own)
    x, stats = solve_bicgstab!(ws, A, b, space; rtol = rtol, itmax = itmax)
    apply_prescribed!(x, bc, u)
    exchange!(halo, x)
    r = fill!(similar(f), 0.0)
    reactions!(r, K, bc, x, f)
    accumulate && accumulate_ghosts!(halo, r)

    return (u = Array(x)[1:n_own], r = Array(r)[1:n_own], K = K, l2g = l2g,
            n_own = n_own, n_local = n_local, stats = stats, halo = halo,
            accumulate = accumulate,
            memory = ["K local" => K,
                "gather map" => (pattern.ke_ptr, pattern.src_ptr, pattern.src),
                "ke buffer" => ke,
                "element data" => data, "BC mask" => bc,
                "u, f, b, r" => (u, f, b, r), "solver workspace" => ws,
                "halo buffers" => halo],
            pattern = pattern, ke = ke, bc = bc, u_vec = u, f_vec = f, ws = ws, A = A,
            space = space)
end

"""
    distributed_matvec(result, x_global)

`K x` with the distributed (unconstrained) matrix: forward exchange, local SpMV, partial
ghost rows back. Returns the own part. Used to compare against the serial matrix when the
own rows are not complete locally.
"""
function distributed_matvec(result, x_global::Vector{Float64}, array_type)
    x = adapt(array_type, x_global[result.l2g])
    x_host = Array(x)
    x_host[(result.n_own + 1):end] .= 0.0           # ghosts must come from the owners
    x = adapt(array_type, x_host)
    y = fill!(similar(x), 0.0)
    mul!(y, DistributedOperator(result.K, result.halo; accumulate = result.accumulate), x)
    return Array(y)[1:(result.n_own)]
end
