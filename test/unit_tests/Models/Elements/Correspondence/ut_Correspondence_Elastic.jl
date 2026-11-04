# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

@testset "ut_Correspondence_Elastic" begin
    function assembled(el, data, D)
        n = size(data.coordinates, 2)
        pattern, K, ke = build_system(el, data, D * n)
        compute_elements!(ke, pattern, el, data)
        assemble!(K, pattern, ke)
        return Matrix(to_sparse(K))
    end

    @testset "1D chain, horizon 1: row couples over two horizons" begin
        data = bar_1d(7; horizon_factor = 1.0)
        el = CorrespondenceElastic(1; lambda = 0.0, mu = 0.5)       # σ = E ε, E = 1
        K = assembled(el, data, 1)
        @test K[4, :] ≈ [0.0, -0.25, 0.0, 0.5, 0.0, -0.25, 0.0] atol=1e-14
        @test K ≈ reference_correspondence(data, 1, 0.0, 0.5, 0.0) rtol=1e-12
    end

    @testset "1D bar with stabilisation and different volumes" begin
        data = bar_1d(15; volume = [0.5 + 0.05i for i in 1:15])
        el = CorrespondenceElastic(1; lambda = 0.0, mu = 0.5, stabilization = 0.1)
        K = assembled(el, data, 1)
        @test K ≈ reference_correspondence(data, 1, 0.0, 0.5, 0.1) rtol=1e-12
        @test maximum(abs, K * ones(15)) < 1e-12                   # rigid body motion
        @test !(K ≈ K')
    end

    @testset "2D plate against reference" begin
        data = plate_2d(5, 4; seed_volume = true)
        el = CorrespondenceElastic(2; lambda = 0.6, mu = 0.4, stabilization = 0.05)
        K = assembled(el, data, 2)
        @test K ≈ reference_correspondence(data, 2, 0.6, 0.4, 0.05) rtol=1e-11
        n = 20
        translation = repeat([1.0, -2.0], n)
        @test maximum(abs, K * translation) < 1e-11
        # the matrix couples over two horizons: some entries without a bond
        nlist = data.nlist
        bond(i, j) = j in nlist[data.nlist_ptr[i]:(data.nlist_ptr[i + 1] - 1)]
        @test any(abs(K[2i - 1, 2j - 1]) > 1e-12 && i != j && !bond(i, j)
                  for i in 1:n, j in 1:n)
    end
end
