# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# The one script to start: runs all combinations (backend × ranks × DOFs), writes one CSV
# and plots solve time and time per iteration over the number of ranks.
#
#   julia --project=test/scaling test/scaling/run_scaling.jl
#   julia --project=test/scaling test/scaling/run_scaling.jl dims=2,3 backends=cpu,gpu,mixed
#   julia --project=test/scaling test/scaling/run_scaling.jl ranks=1,2,4,8,16 \
#         dofs=100000,1000000 launcher=srun                      # cluster (inside a job)
#
# Keys (defaults in brackets):
#   ranks=1,2,4          numbers of MPI ranks                       [1,2,4]
#   dofs=1000,3000,10000 target global DOFs                         [1000,3000,10000]
#   backends=cpu,gpu,mixed                                          [cpu,gpu,mixed]
#   dims=2               1, 2 and/or 3                              [2]
#   element=bb           bb | corr                                  [bb]
#   gpu_weight=2         points of a GPU rank relative to a CPU rank (mixed)  [2]
#   launcher=mpiexec     mpiexec (from MPI.jl / system MPI) | srun  [mpiexec]
#   threads=1            Julia threads per rank                     [1]
#   out=results/scaling  output folder                              [results/scaling]
#   horizon, rtol, itmax are passed on to scaling_case.jl.
# GPU backends are skipped automatically if CUDA is not functional.

using MPI: mpiexec
using Dates: now, format

options = Dict(split(a, "="; limit = 2)[1] => split(a, "="; limit = 2)[2]
               for a in ARGS if occursin("=", a))
list(key, default) = haskey(options, key) ? String.(split(options[key], ",")) : default
ranks = parse.(Int, list("ranks", ["1", "2", "4"]))
dofs = parse.(Int, list("dofs", ["1000", "3000", "10000"]))
backends = list("backends", ["cpu", "gpu", "mixed"])
dims = parse.(Int, list("dims", ["2"]))
element = get(options, "element", "bb")
gpu_weight = get(options, "gpu_weight", "2")
launcher = get(options, "launcher", "mpiexec")
threads = get(options, "threads", "1")
out = get(options, "out", joinpath("results", "scaling"))
passthrough = ["$k=$(options[k])"
               for k in ("horizon", "rtol", "itmax") if haskey(options, k)]

project = @__DIR__
julia = Base.julia_cmd()
if any(b -> b != "cpu", backends)
    gpu_ok = success(`$julia --project=$project -e "using CUDA; exit(CUDA.functional() ? 0 : 1)"`)
    if !gpu_ok
        println("CUDA not functional: backends gpu and mixed are skipped.")
        filter!(==("cpu"), backends)
    end
end

mkpath(out)
stamp = format(now(), "yyyy-mm-dd_HHMMSS")
csv = joinpath(out, "scaling_$stamp.csv")
script = joinpath(@__DIR__, "scaling_case.jl")
failed = String[]
total = length(dims) * length(backends) * length(ranks) * length(dofs)
println("$total runs, results: $csv\n")
for dim in dims, backend in backends, n in ranks, d in dofs
    case = `$script dim=$dim dofs=$d backend=$backend element=$element gpu_weight=$gpu_weight csv=$csv $passthrough`
    run_cmd = launcher == "srun" ?
              `srun -n $n $julia --project=$project -t $threads $case` :
              `$(mpiexec()) -bind-to core -n $n $julia --project=$project -t $threads $case`
    ok = success(pipeline(run_cmd; stdout = stdout, stderr = stderr))
    ok || push!(failed, "dim=$dim backend=$backend ranks=$n dofs=$d")
end

println()
if isempty(failed)
    println("All $total runs passed their checks.")
else
    println("$(length(failed)) run(s) failed or did not converge:")
    foreach(f -> println("  ", f), failed)
end

if isfile(csv)
    plot_project = joinpath(@__DIR__, "plot")
    run(`$julia --project=$plot_project -e "using Pkg; Pkg.instantiate()"`)
    run(`$julia --project=$plot_project $(joinpath(plot_project, "plot_scaling.jl")) $csv`)
end
exit(isempty(failed) ? 0 : 1)
