# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

@testset "ut_BiCGStab" begin
    n = 60
    data = bar_1d(n; volume = [0.5 + 0.01i for i in 1:n])   # not symmetric
    el = BondbasedElastic(1)
    pattern, K, ke = build_system(el, data, n)
    compute_elements!(ke, pattern, el, data)
    assemble!(K, pattern, ke)
    p = [1, n]
    bc = create_dof_set(p, n)
    u = zeros(n)
    u[n] = 100.0
    f = zeros(n)
    f[30] = 10.0
    b = zeros(n)
    constrained_rhs!(b, K, bc, u, f)

    ws = BiCGStabWorkspace(b)
    jacobi!(ws, K, bc)
    A = DistributedOperator(ConstrainedOperator(K, bc), nothing)   # serial: no halo
    x, stats = solve_bicgstab!(ws, A, b, SerialSpace(n); rtol = 1e-12)
    apply_prescribed!(x, bc, u)
    reference = reference_solve(Matrix(to_sparse(K)), p, [0.0, 100.0], f)
    @test stats.solved
    @test x ≈ reference rtol=1e-9
    @test x[p] == [0.0, 100.0]

    # second solve with the same workspace allocates nothing (measured inside a function,
    # globals of the test set would allocate themselves)
    function solve_allocations(ws, A, b, n)
        solve_bicgstab!(ws, A, b, SerialSpace(n); rtol = 1e-12)
        return @allocated solve_bicgstab!(ws, A, b, SerialSpace(n); rtol = 1e-12)
    end
    @test solve_allocations(ws, A, b, n) == 0
end
