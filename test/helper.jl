# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Test data is created here, without PeriLab (README A0.3b).

using ArraysOfArrays: VectorOfVectors
using LinearAlgebra: norm

"""
    neighbor_list(coordinates, horizon)

Brute-force neighbour search (all points within `horizon`, excluding the point itself).
"""
function neighbor_list(coordinates::AbstractMatrix{Float64}, horizon::Float64)
    n = size(coordinates, 2)
    nlist = VectorOfVectors{Int64}()
    for i in 1:n
        push!(nlist,
              [j
               for j in 1:n
               if j != i && norm(coordinates[:, j] - coordinates[:, i]) <= horizon + 1e-12])
    end
    return nlist
end

function unit_damage(nlist::VectorOfVectors{Int64})
    VectorOfVectors([ones(length(n)) for n in nlist])
end

"""
    bar_1d(n; dx, horizon_factor, volume)

1D bar with `n` points (README A7.1a). Values are arbitrary.
"""
function bar_1d(n::Int64; dx::Float64 = 1.0, horizon_factor::Float64 = 3.0,
                volume::Vector{Float64} = fill(1.0, n), constant::Float64 = 1.0)
    coordinates = reshape(collect(0:(n - 1)) .* dx, 1, n)
    nlist = neighbor_list(coordinates, horizon_factor * dx)
    return create_bondbased_data(coordinates, volume, nlist, unit_damage(nlist),
                                 fill(constant, n))
end

"""
    plate_2d(nx, ny; dx, horizon_factor)

2D grid of points for checking `D = 2`.
"""
function plate_2d(nx::Int64, ny::Int64; dx::Float64 = 1.0, horizon_factor::Float64 = 2.0,
                  seed_volume::Bool = false)
    coordinates = zeros(2, nx * ny)
    for j in 1:ny, i in 1:nx
        coordinates[:, (j - 1) * nx + i] = [(i - 1) * dx, (j - 1) * dx]
    end
    n = nx * ny
    volume = seed_volume ? [0.5 + 0.1 * mod(7i, 11) for i in 1:n] : fill(1.0, n)
    nlist = neighbor_list(coordinates, horizon_factor * dx)
    return create_bondbased_data(coordinates, volume, nlist, unit_damage(nlist),
                                 fill(1.0, n))
end

"""
    reference_bondbased(data, D)

Independent dense assembly of the bond-based matrix, straight from the formula.
"""
function reference_bondbased(data, D::Int64)
    x = data.coordinates
    n = size(x, 2)
    K = zeros(D * n, D * n)
    for i in 1:n, b in data.nlist_ptr[i]:(data.nlist_ptr[i + 1] - 1)
        j = data.nlist[b]
        xi = x[:, j] - x[:, i]
        L = norm(xi)
        k = data.bond_constant[i] * data.bond_damage[b] * data.volume[j] / L *
            (xi * xi') / L^2
        di = ((i - 1) * D + 1):(i * D)
        dj = ((j - 1) * D + 1):(j * D)
        K[di, di] .+= k
        K[di, dj] .-= k
    end
    return K
end

"""
    reference_solve(K, prescribed, u_p, f)

Direct solve with explicit sub-matrices; only as reference.
"""
function reference_solve(K::AbstractMatrix, prescribed::Vector{Int64},
                         u_p::Vector{Float64}, f::Vector{Float64})
    n = size(K, 1)
    free = setdiff(1:n, prescribed)
    u = zeros(n)
    u[prescribed] = u_p
    u[free] = K[free, free] \ (f[free] - K[free, prescribed] * u_p)
    return u
end

"""
    solve_bar(data, D, prescribed, u_p, f; array_type)

Full pipeline of the prototype: symbolic phase on the host, numerical phase and solve on
the backend of `array_type` (`Array` or `CuArray`).
"""
function solve_bar(data, D::Int64, prescribed::Vector{Int64}, u_p::Vector{Float64},
                   f::Vector{Float64}; array_type = Array)
    el = BondbasedElastic(D)
    n_dofs = D * (length(data.nlist_ptr) - 1)
    rows, cols = element_dofs(el, data)
    host_pattern = create_assembly_pattern(rows, cols, n_dofs)

    pattern = adapt(array_type, host_pattern)
    device_data = adapt(array_type, data)
    K = create_matrix(pattern)
    ke = create_element_buffer(pattern)
    compute_elements!(ke, pattern, el, device_data)
    assemble!(K, pattern, ke)

    bc = adapt(array_type, create_dof_set(prescribed, n_dofs))
    u_full = zeros(n_dofs)
    u_full[prescribed] = u_p
    u_dev = adapt(array_type, u_full)
    f_dev = adapt(array_type, f)
    b = similar(f_dev)
    constrained_rhs!(b, K, bc, u_dev, f_dev)

    workspace = LinearSolverWorkspace(n_dofs, typeof(b); memory = n_dofs)
    u, stats = solve_linear!(workspace, ConstrainedOperator(K, bc), b; rtol = 1e-13)
    apply_prescribed!(u, bc, u_dev)
    r = similar(b)
    reactions!(r, K, bc, u, f_dev)
    return (u = Array(u), r = Array(r), K = K, stats = stats)
end

"""
    correspondence_forces(u, data, D, lambda, mu, beta)

Independent reference: force density of the linear correspondence model with bond
stabilisation, straight from the equations (plain loops, dense matrices).
"""
function correspondence_forces(u::Vector{Float64}, data, D::Int64, lambda::Float64,
                               mu::Float64, beta::Float64)
    x = data.coordinates
    n = length(data.nlist_ptr) - 1
    n_points = size(x, 2)
    U = reshape(u, D, n_points)
    f = zeros(D, n_points)
    for i in 1:n
        bonds = data.nlist_ptr[i]:(data.nlist_ptr[i + 1] - 1)
        shape = zeros(D, D)
        for b in bonds
            xi = x[:, data.nlist[b]] - x[:, i]
            shape += data.bond_damage[b] * data.volume[data.nlist[b]] * xi * xi'
        end
        Kinv = inv(shape)
        H = zeros(D, D)
        for b in bonds
            j = data.nlist[b]
            xi = x[:, j] - x[:, i]
            H += (U[:, j] - U[:, i]) * (data.bond_damage[b] * data.volume[j] * Kinv * xi)'
        end
        eps = 0.5 * (H + H')
        sigma = lambda * sum(eps[k, k] for k in 1:D) * Matrix(1.0I, D, D) + 2mu * eps
        for b in bonds
            j = data.nlist[b]
            omega = data.bond_damage[b]
            xi = x[:, j] - x[:, i]
            t = omega * sigma * Kinv * xi
            z = U[:, j] - U[:, i] - H * xi
            p = beta * omega * z
            f[:, i] += (t + p) * data.volume[j]
            f[:, j] -= (t + p) * data.volume[i]
        end
    end
    return vec(f)
end

"Reference matrix of the linear correspondence model, column by column from the forces."
function reference_correspondence(data, D::Int64, lambda::Float64, mu::Float64,
                                  beta::Float64)
    n_dofs = D * size(data.coordinates, 2)
    K = zeros(n_dofs, n_dofs)
    for k in 1:n_dofs
        e = zeros(n_dofs)
        e[k] = 1.0
        K[:, k] = -correspondence_forces(e, data, D, lambda, mu, beta)   # K u = −f_int
    end
    return K
end
