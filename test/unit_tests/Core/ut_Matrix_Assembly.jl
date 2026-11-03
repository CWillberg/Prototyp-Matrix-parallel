# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

@testset "ut_Matrix_Assembly" begin
    @testset "pattern of two small elements" begin
        rows = VectorOfVectors([[1], [1, 2]])
        cols = VectorOfVectors([[1, 3], [2, 3]])
        pattern = create_assembly_pattern(rows, cols, 3)
        @test pattern.ke_ptr == [1, 3, 7]
        @test pattern.rowptr == [1, 4, 6, 6]
        @test pattern.colval == [1, 2, 3, 2, 3]
        K = create_matrix(pattern)
        ke = Float64[1, 2, 10, 20, 30, 40]
        assemble!(K, pattern, ke)
        # K[1,1]=1, K[1,3]=2+30, K[1,2]=10, K[2,2]=20, K[2,3]=40
        @test Matrix(to_sparse(K)) == [1 10 32; 0 20 40; 0 0 0]
    end

    @testset "elements writing into shared rows (correspondence / IGA-PD like)" begin
        rng = Random.MersenneTwister(1)
        n_dofs = 30
        rows = VectorOfVectors{Int64}()
        cols = VectorOfVectors{Int64}()
        for _ in 1:40
            push!(rows, unique(rand(rng, 1:n_dofs, rand(rng, 1:4))))
            push!(cols, unique(rand(rng, 1:n_dofs, rand(rng, 1:8))))
        end
        pattern = create_assembly_pattern(rows, cols, n_dofs)
        ke = rand(rng, pattern.ke_ptr[end] - 1)
        K = create_matrix(pattern)
        assemble!(K, pattern, ke)

        reference = zeros(n_dofs, n_dofs)
        for e in eachindex(rows)
            n_r = length(rows[e])
            for (jc, c) in enumerate(cols[e]), (ir, r) in enumerate(rows[e])
                reference[r, c] += ke[pattern.ke_ptr[e] + (jc - 1) * n_r + ir - 1]
            end
        end
        @test Matrix(to_sparse(K)) ≈ reference rtol=1e-14
        # pattern stays, values are overwritten on reassembly (README A2.5)
        rowptr = copy(K.rowptr)
        assemble!(K, pattern, 2 .* ke)
        @test K.rowptr == rowptr
        @test Matrix(to_sparse(K)) ≈ 2 .* reference rtol=1e-14
    end

    @testset "invalid input" begin
        @test_throws ArgumentError create_assembly_pattern(VectorOfVectors([[4]]),
                                                           VectorOfVectors([[1]]), 3)
        @test_throws ArgumentError create_assembly_pattern(VectorOfVectors([[1]]),
                                                           VectorOfVectors{Int64}(), 3)
    end
end
