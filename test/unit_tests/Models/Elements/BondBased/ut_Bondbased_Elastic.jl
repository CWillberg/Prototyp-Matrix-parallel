# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

@testset "ut_Bondbased_Elastic" begin
    function assembled(data, D)
        el = BondbasedElastic(D)
        pattern, K, ke = build_system(el, data, D * (length(data.nlist_ptr) - 1))
        compute_elements!(ke, pattern, el, data)
        assemble!(K, pattern, ke)
        return Matrix(to_sparse(K))
    end

    @testset "1D bar against dense formula" begin
        data = bar_1d(12)
        K = assembled(data, 1)
        @test K ≈ reference_bondbased(data, 1) rtol=1e-14
        @test maximum(abs, sum(K; dims = 2)) < 1e-12   # rigid body motion
        @test K ≈ K'                                    # uniform volumes
    end

    @testset "1D bar with different volumes is not symmetric (README A2.2)" begin
        data = bar_1d(12; volume = [0.5 + 0.1i for i in 1:12])
        K = assembled(data, 1)
        @test K ≈ reference_bondbased(data, 1) rtol=1e-14
        @test !(K ≈ K')
    end

    @testset "2D plate against dense formula" begin
        for seed_volume in (false, true)
            data = plate_2d(4, 3; seed_volume = seed_volume)
            K = assembled(data, 2)
            @test K ≈ reference_bondbased(data, 2) rtol=1e-14
            @test maximum(abs, sum(K[:, 1:2:end]; dims = 2)) < 1e-12   # translation x
        end
    end

    @testset "zero bond length is rejected at initialisation" begin
        coordinates = [0.0 0.0]
        nlist = VectorOfVectors([[2], [1]])
        @test_throws ArgumentError create_bondbased_data(coordinates, [1.0, 1.0], nlist,
                                                         VectorOfVectors([[1.0], [1.0]]),
                                                         [1.0, 1.0])
    end
end
