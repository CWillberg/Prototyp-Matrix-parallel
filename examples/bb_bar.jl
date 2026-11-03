# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference example README A7.1a: 1D bond-based bar, 100 points, u = 0 left,
# u = 100 mm right, point force F at point 50. Run from the repository root:
#
#   julia --project=. examples/bb_bar.jl          # CPU
#   julia --project=. examples/bb_bar.jl gpu      # CUDA (needs CUDA in the environment)

using PD_Matrix
using Adapt: adapt
using LinearAlgebra: norm
include(joinpath(@__DIR__, "..", "test", "helper.jl"))

array_type = Array
if "gpu" in ARGS
    using CUDA
    array_type = CuArray
end

n = 100
f = zeros(n)
f[50] = 10.0
data = bar_1d(n)
result = solve_bar(data, 1, [1, n], [0.0, 100.0], f; array_type = array_type)

println("backend:         ", array_type)
println("GMRES solved:    ", result.stats.solved, " in ", result.stats.niter,
        " iterations")
println("u[1], u[50], u[n]: ", result.u[1], ", ", result.u[50], ", ", result.u[n])
println("reactions:       ", result.r[1], ", ", result.r[n], "  (sum ", sum(result.r),
        ")")
