# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

using Test
using Random
using LinearAlgebra: mul!, I
using Adapt: adapt
using ArraysOfArrays: VectorOfVectors
using PD_Matrix

const GPU_AVAILABLE = let
    use_gpu = get(ENV, "PD_MATRIX_TEST_GPU", "true") == "true"
    ok = false
    if use_gpu
        try
            @eval using CUDA
            ok = CUDA.functional()
        catch err
            @info "CUDA could not be loaded: $err"
        end
    end
    ok
end

include("helper.jl")

@testset "PD_Matrix" begin
    @testset "unit_tests" begin
        include("unit_tests/Core/ut_Matrix_Assembly.jl")
        include("unit_tests/Core/ut_Operators.jl")
        include("unit_tests/Core/Model_reduction/ut_Guyan.jl")
        include("unit_tests/Models/Elements/BondBased/ut_Bondbased_Elastic.jl")
    end
    @testset "fullscale_tests" begin
        include("fullscale_tests/test_bb_bar/test_bb_bar.jl")
    end
end
