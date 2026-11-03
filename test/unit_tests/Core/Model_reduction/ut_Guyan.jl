# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

@testset "ut_Guyan" begin
    n = 40
    data = bar_1d(n; volume = [0.5 + 0.02i for i in 1:n])   # not symmetric
    el = BondbasedElastic(1)
    pattern, K, ke = build_system(el, data, n)
    compute_elements!(ke, pattern, el, data)
    assemble!(K, pattern, ke)
    Kd = Matrix(to_sparse(K))

    m = sort(unique([1:3; 18:22; (n - 2):n]))
    s = setdiff(1:n, m)
    masters = create_dof_set(m, n)

    @testset "extract" begin
        @test Matrix(extract(K, m, s)) == Kd[m, s]
        @test Matrix(extract(K, s, s)) == Kd[s, s]
    end

    @testset "reduce_matrices equals the Schur complement" begin
        K_r, M_r = reduce_matrices(K, ones(n), masters)
        T = -Kd[s, s] \ Kd[s, m]
        @test K_r ≈ Kd[m, m] + Kd[m, s] * T rtol=1e-12
        @test M_r ≈ I + T' * T rtol=1e-12
    end

    @testset "matrix-free SchurOperator equals reduce_matrices (README A4a.3)" begin
        K_r, _ = reduce_matrices(K, ones(n), masters)
        S = SchurOperator(K, masters; memory = n)
        x = zeros(n)
        x[m] = collect(range(-1.0, 1.0; length = length(m)))
        x[s] .= 7.0
        y = zeros(n)
        mul!(y, S, x)
        @test y[m] ≈ K_r * x[m] rtol=1e-9
        @test y[s] == x[s]
    end

    @testset "static condensation is exact for loads on masters" begin
        p = [1, n]
        u_p = [0.0, 100.0]
        f = zeros(n)
        f[20] = 10.0
        u_full = reference_solve(Kd, p, u_p, f)

        K_r, _ = reduce_matrices(K, ones(n), masters)
        local_p = [findfirst(==(i), m) for i in p]
        u_m = reference_solve(K_r, local_p, u_p, f[m])
        @test u_m ≈ u_full[m] rtol=1e-10
    end
end
