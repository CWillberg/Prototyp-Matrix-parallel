# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

@testset "ut_Operators" begin
    function bar_matrix(n)
        data = bar_1d(n; volume = [0.5 + 0.01i for i in 1:n])
        el = BondbasedElastic(1)
        pattern, K, ke = build_system(el, data, n)
        compute_elements!(ke, pattern, el, data)
        assemble!(K, pattern, ke)
        return K
    end

    n = 20
    K = bar_matrix(n)
    Kd = Matrix(to_sparse(K))
    p = [1, 7, 20]
    f_set = setdiff(1:n, p)
    bc = create_dof_set(p, n)
    x = collect(range(-1.0, 2.0; length = n))
    y = zeros(n)

    @testset "ConstrainedOperator = K_ff on free, identity on prescribed" begin
        mul!(y, ConstrainedOperator(K, bc), x)
        expected = copy(x)
        expected[f_set] = Kd[f_set, f_set] * x[f_set]
        @test y ≈ expected rtol=1e-14
    end

    @testset "constrained_rhs! and reactions!" begin
        u = zeros(n)
        u[p] = [0.0, 3.0, 5.0]
        f = collect(1.0:n)
        b = zeros(n)
        constrained_rhs!(b, K, bc, u, f)
        @test all(iszero, b[p])
        @test b[f_set] ≈ f[f_set] - Kd[f_set, p] * u[p] rtol=1e-14
        r = zeros(n)
        reactions!(r, K, bc, x, f)
        @test r[p] ≈ Kd[p, :] * x - f[p] rtol=1e-14
        @test all(iszero, r[f_set])
    end

    @testset "SubOperator = blocks of K without copying" begin
        m = create_dof_set(p, n)
        mul!(y, SubOperator(K, m, m), x)
        @test y[p] ≈ Kd[p, p] * x[p]
        @test all(iszero, y[f_set])
        mul!(y, SubOperator(K, m, m; complement_rows = true), x)
        @test y[f_set] ≈ Kd[f_set, p] * x[p]
        mul!(y, SubOperator(K, m, m; complement_cols = true), x)
        @test y[p] ≈ Kd[p, f_set] * x[f_set]
    end

    @testset "no sub-matrix memory: allocation independent of problem size (README A6.1)" begin
        function allocation(n)
            K = bar_matrix(n)
            bc = create_dof_set([1, n], n)
            A = ConstrainedOperator(K, bc)
            x = ones(n)
            y = zeros(n)
            mul!(y, A, x)
            return @allocated mul!(y, A, x)
        end
        small = allocation(100)
        large = allocation(20_000)
        @test large == small
        @test small < 4096
    end
end
