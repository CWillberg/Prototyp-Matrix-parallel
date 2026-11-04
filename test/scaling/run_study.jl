# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Study: for every problem size
#   1. reference   1 CPU rank
#   2. MPI only    CPU ranks          (cpu_ranks)
#   3. CUDA only   GPU ranks          (gpu_ranks, one GPU per rank)
#   4. mixed       GPU + CPU ranks    (mixed_ranks; one rank per GPU on the GPU)
# then plots solve time and speedup relative to the 1-CPU reference of the same size.
#
#   julia --project=test/scaling test/scaling/run_study.jl
#   julia --project=test/scaling test/scaling/run_study.jl dofs=10000 dims=3
#   julia --project=test/scaling test/scaling/run_study.jl launcher=srun \
#         dofs=100000,1000000 cpu_ranks=2,4,8,16,32 gpu_ranks=1,2,4,8 mixed_ranks=8,16,32
#
# Keys (defaults): dofs=1000,3000,10000  cpu_ranks=2,4  gpu_ranks=1  mixed_ranks=2,4
#   dims=2  element=bb|corr  gpu_weight=2  launcher=mpiexec|srun  threads=1
#   out=results/study  horizon, rtol, itmax (passed on)

using MPI: mpiexec
using Dates: now, format

options = Dict(split(a, "="; limit = 2)[1] => split(a, "="; limit = 2)[2]
               for a in ARGS if occursin("=", a))
ints(key, default) = haskey(options, key) ?
                     [parse(Int, v) for v in split(options[key], ",") if !isempty(v)] :
                     default
dofs = ints("dofs", [1000, 3000, 10000])
cpu_ranks = ints("cpu_ranks", [2, 4])
gpu_ranks = ints("gpu_ranks", [1])
mixed_ranks = ints("mixed_ranks", [2, 4])
dims = ints("dims", [2])
element = get(options, "element", "bb")
gpu_weight = get(options, "gpu_weight", "2")
launcher = get(options, "launcher", "mpiexec")
threads = get(options, "threads", "1")
out = get(options, "out", joinpath("results", "study"))
passthrough = ["$k=$(options[k])" for k in ("horizon", "rtol", "itmax")
               if haskey(options, k)]

project = @__DIR__
julia = Base.julia_cmd()
gpu_ok = success(`$julia --project=$project -e "using CUDA; exit(CUDA.functional() ? 0 : 1)"`)
if !gpu_ok
    println("CUDA not functional: CUDA-only and mixed runs are skipped.")
    empty!(gpu_ranks)
    empty!(mixed_ranks)
end

mkpath(out)
csv = joinpath(out, "study_$(format(now(), "yyyy-mm-dd_HHMMSS")).csv")
script = joinpath(@__DIR__, "scaling_case.jl")

# run list in the order of the study
runs = Tuple{Int,Int,String,Int}[]
for dim in dims, d in dofs
    push!(runs, (dim, d, "cpu", 1))                                 # reference
    foreach(n -> push!(runs, (dim, d, "cpu", n)), filter(!=(1), cpu_ranks))
    foreach(n -> push!(runs, (dim, d, "gpu", n)), gpu_ranks)
    foreach(n -> push!(runs, (dim, d, "mixed", n)), mixed_ranks)
end
println("$(length(runs)) runs, results: $csv\n")

failed = String[]
for (dim, d, backend, n) in runs
    case = `$script dim=$dim dofs=$d backend=$backend element=$element gpu_weight=$gpu_weight csv=$csv $passthrough`
    cmd = launcher == "srun" ? `srun -n $n $julia --project=$project -t $threads $case` :
          `$(mpiexec()) -n $n $julia --project=$project -t $threads $case`
    success(pipeline(cmd; stdout = stdout, stderr = stderr)) ||
        push!(failed, "dim=$dim dofs=$d backend=$backend ranks=$n")
end

println()
isempty(failed) ? println("All $(length(runs)) runs passed their checks.") :
(println("$(length(failed)) run(s) failed:"); foreach(f -> println("  ", f), failed))

if isfile(csv)
    plot_project = joinpath(@__DIR__, "plot")
    run(`$julia --project=$plot_project -e "using Pkg; Pkg.instantiate()"`)
    run(`$julia --project=$plot_project $(joinpath(plot_project, "plot_study.jl")) $csv`)
end
exit(isempty(failed) ? 0 : 1)
