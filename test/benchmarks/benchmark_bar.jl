# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Laufzeitvergleich am 1D-bond-based-Stab: SparseArrays (Referenz, seriell),
# PD_Matrix auf der CPU (Threads aus `julia -t`) und auf der GPU.
#
#   julia --project=test/benchmarks -t 1 (bzw. -t 4, Maximum in WSL) test/benchmarks/benchmark_bar.jl

using PD_Matrix
using Adapt: adapt
using ArraysOfArrays: VectorOfVectors
using LinearAlgebra: mul!, norm
using SparseArrays: sparse
using Printf: @printf
using CUDA

function bar_data(n; m = 3)
    coordinates = reshape(collect(0.0:(n - 1)), 1, n)
    nlist = VectorOfVectors{Int64}()
    for i in 1:n
        push!(nlist, [j for j in max(1, i - m):min(n, i + m) if j != i])
    end
    damage = VectorOfVectors([ones(length(l)) for l in nlist])
    return create_bondbased_data(coordinates, ones(n), nlist, damage, ones(n))
end

# Referenz: COO-Tripel + sparse() wie im seriellen PeriLab-Pfad
function reference_assembly(data, n)
    I = Int64[]; J = Int64[]; V = Float64[]
    x = data.coordinates
    for i in 1:n, b in data.nlist_ptr[i]:(data.nlist_ptr[i + 1] - 1)
        j = data.nlist[b]
        k = data.bond_constant[i] * data.volume[j] / abs(x[1, j] - x[1, i])
        push!(I, i); push!(J, i); push!(V, k)
        push!(I, i); push!(J, j); push!(V, -k)
    end
    return sparse(I, J, V, n, n)
end

function best_time(f; repeats = 5)
    f()
    t = Inf
    for _ in 1:repeats
        t0 = time_ns()
        f()
        t = min(t, (time_ns() - t0) / 1e6)
    end
    return t
end

function run(n)
    data = bar_data(n)
    el = BondbasedElastic(1)
    rows, cols = element_dofs(el, data)
    t_sym = @elapsed (host_pattern = create_assembly_pattern(rows, cols, n))

    t_ref_asm = best_time(() -> reference_assembly(data, n))
    Kref = reference_assembly(data, n)
    x = rand(n); y = zeros(n)
    t_ref_mv = best_time(() -> mul!(y, Kref, x); repeats = 20)

    K = create_matrix(host_pattern); ke = create_element_buffer(host_pattern)
    t_cpu_asm = best_time(() -> (compute_elements!(ke, host_pattern, el, data);
                                 assemble!(K, host_pattern, ke)))
    t_cpu_mv = best_time(() -> mul!(y, K, x); repeats = 20)
    @assert norm(PD_Matrix.to_sparse(K) - Kref) < 1e-10

    t_gpu_asm = t_gpu_mv = NaN
    if CUDA.functional()
        p = adapt(CuArray, host_pattern); d = adapt(CuArray, data)
        Kg = create_matrix(p); keg = create_element_buffer(p)
        xg = CuArray(x); yg = similar(xg)
        t_gpu_asm = best_time(() -> (compute_elements!(keg, p, el, d); assemble!(Kg, p, keg)))
        t_gpu_mv = best_time(() -> mul!(yg, Kg, xg); repeats = 20)
    end
    @printf("%9d | %8.2f | %9.3f %9.3f %9.3f | %8.4f %8.4f %8.4f\n", n, t_sym * 1e3,
            t_ref_asm, t_cpu_asm, t_gpu_asm, t_ref_mv, t_cpu_mv, t_gpu_mv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Threads: ", Threads.nthreads(), ", GPU: ",
            CUDA.functional() ? CUDA.name(CUDA.device()) : "keine")
    println("Zeiten in ms (bestes von 5 bzw. 20 Läufen)")
    println("        n | symbol.  | Assemblierung: Ref.  CPU       GPU       | SpMV: Ref.  CPU      GPU")
end
if abspath(PROGRAM_FILE) == @__FILE__
    for n in (100, 1_000, 10_000)
        run(n)
    end
end
