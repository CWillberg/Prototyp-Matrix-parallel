# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Memory_Report

using MPI
using Printf: @sprintf

export data_bytes, memory_report

"""
    data_bytes(x)

Bytes of all arrays reachable from `x` (host or device arrays, structs, tuples). Only
array payload is counted, no Julia object headers.
"""
data_bytes(x::AbstractArray{<:Number}) = sizeof(x)
data_bytes(x::AbstractArray) = sum(data_bytes, x; init = 0)
data_bytes(::Union{Number,Symbol,AbstractString,Nothing,Function,Type}) = 0
data_bytes(x::Union{Tuple,NamedTuple}) = sum(data_bytes, values(x); init = 0)
data_bytes(::MPI.Comm) = 0
data_bytes(::MPI.AbstractRequest) = 0
function data_bytes(x)
    isstructtype(typeof(x)) || return 0
    return sum((data_bytes(getfield(x, f)) for f in fieldnames(typeof(x))); init = 0)
end

"""
    memory_report(comm, sizes, items; io = stdout)

Prints a table on rank 0 with one line per rank: integer sizes (`sizes`, e.g. own rows,
ghost columns, nnz) and the memory of every item in MB, plus the sum over all ranks.
Returns the per-rank values on rank 0 (`nothing` elsewhere).

# Arguments
- `sizes::Vector{Pair{String,Int64}}`: Counts per rank
- `items::Vector{<:Pair{String}}`: Named objects whose `data_bytes` are reported
"""
function memory_report(comm::MPI.Comm, sizes::Vector{Pair{String,Int64}},
                       items::Vector{<:Pair{String}}; io::IO = stdout)
    local_values = vcat([last(s) for s in sizes], [data_bytes(last(i)) for i in items])
    n_values = length(local_values)
    gathered = MPI.Gather(local_values, comm; root = 0)
    MPI.Comm_rank(comm) == 0 || return nothing
    table = reshape(gathered, n_values, :)'
    names = vcat(first.(sizes), first.(items) .* " MB")
    widths = max.(length.(names), 10)
    header = "rank | " * join([lpad(n, w) for (n, w) in zip(names, widths)], " | ")
    println(io, header)
    println(io, "-"^length(header))
    n_sizes = length(sizes)
    format(j, value) = j <= n_sizes ? string(value) : @sprintf("%.4f", value / 2^20)
    for r in axes(table, 1)
        cells = [lpad(format(j, table[r, j]), widths[j]) for j in 1:n_values]
        println(io, lpad(r - 1, 4), " | ", join(cells, " | "))
    end
    totals = [lpad(format(j, sum(table[:, j])), widths[j]) for j in 1:n_values]
    println(io, " sum | ", join(totals, " | "))
    return table
end

end
