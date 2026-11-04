# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Runs all distributed scenarios for several rank counts and backends and prints a summary.
#
#   julia --project=test/mpi test/mpi/run_scenarios.jl              # CPU, ranks 1 2 4
#   julia --project=test/mpi test/mpi/run_scenarios.jl cpu gpu      # CPU and GPU ranks
#   julia --project=test/mpi test/mpi/run_scenarios.jl gpu ranks=1,2
#
# Further arguments (scenario names, `timer`) are passed on to scenarios.jl.

using MPI: mpiexec

backends = filter(in(("cpu", "gpu")), ARGS)
isempty(backends) && (backends = ["cpu"])
rank_arg = findfirst(startswith("ranks="), ARGS)
ranks = isnothing(rank_arg) ? [1, 2, 4] :
        parse.(Int, split(split(ARGS[rank_arg], "=")[2], ","))
passthrough = filter(a -> !(a in ("cpu", "gpu")) && !startswith(a, "ranks="), ARGS)

script = joinpath(@__DIR__, "scenarios.jl")
project = @__DIR__
results = Tuple{String,Int,Bool}[]
for backend in backends, n in ranks
    println("\n=== backend = $backend, ranks = $n ===")
    cmd = `$(mpiexec()) -n $n $(Base.julia_cmd()) --project=$project $script $backend $passthrough`
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    push!(results, (backend, n, ok))
end
println("\nSummary")
for (backend, n, ok) in results
    println(ok ? "  PASS " : "  FAIL ", backend, ", ranks = ", n)
end
exit(all(last, results) ? 0 : 1)
