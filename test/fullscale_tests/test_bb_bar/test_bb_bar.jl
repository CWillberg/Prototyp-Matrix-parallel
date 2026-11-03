# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference example README A7.1a: 1D bond-based bar, 100 points,
# u = 0 left, u = 100 mm right, point force F in the middle.

@testset "test_bb_bar" begin
    n = 100
    F = 10.0
    prescribed = [1, n]
    u_p = [0.0, 100.0]
    f = zeros(n)
    f[50] = F

    @testset "uniform volumes (symmetric)" begin
        data = bar_1d(n)
        result = solve_bar(data, 1, prescribed, u_p, f)
        reference = reference_solve(reference_bondbased(data, 1), prescribed, u_p, f)
        @test result.stats.solved
        @test result.u ≈ reference rtol=1e-9
        @test result.u[prescribed] == u_p
        # equilibrium: reactions balance F (column sums of K vanish for symmetric K)
        @test sum(result.r) ≈ -F rtol=1e-8
    end

    @testset "different volumes (not symmetric)" begin
        data = bar_1d(n; volume = [0.5 + 0.01i for i in 1:n])
        result = solve_bar(data, 1, prescribed, u_p, f)
        K = reference_bondbased(data, 1)
        @test !(K ≈ K')
        @test result.stats.solved
        @test result.u ≈ reference_solve(K, prescribed, u_p, f) rtol=1e-9
    end

    @testset "boundary conditions change without reassembly (README A6.3)" begin
        data = bar_1d(n)
        r1 = solve_bar(data, 1, prescribed, u_p, f)
        r2 = solve_bar(data, 1, [1, 50, n], [0.0, 20.0, 100.0], zeros(n))
        @test r1.K.nzval == r2.K.nzval
        @test r2.u[50] == 20.0
    end

    if GPU_AVAILABLE
        @testset "GPU (CUDA) gives the CPU result (README A7.2)" begin
            for volume in (fill(1.0, n), [0.5 + 0.01i for i in 1:n])
                data = bar_1d(n; volume = volume)
                cpu = solve_bar(data, 1, prescribed, u_p, f)
                gpu = solve_bar(data, 1, prescribed, u_p, f; array_type = CuArray)
                @test gpu.stats.solved
                @test Array(gpu.K.nzval) == cpu.K.nzval      # deterministic gather
                @test gpu.u ≈ cpu.u rtol=1e-9
                @test gpu.r≈cpu.r rtol=1e-7 atol=1e-9
            end
        end
    else
        @info "CUDA not functional, GPU tests skipped."
    end
end
