# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Plots a scaling CSV (from run_scaling.jl): one figure per dimension and element,
# columns = backend (cpu | gpu | mixed), rows = solve time and time per iteration over the
# number of ranks, one line per problem size (DOFs). Writes <csv name>_<dim>D_<element>.png.
#
#   julia --project=test/scaling/plot test/scaling/plot/plot_scaling.jl results/scaling/x.csv

using CairoMakie

const TEXT = colorant"#0b0b0b"
const TEXT_2 = colorant"#52514e"
const SURFACE = colorant"#fcfcfb"
const GRID = RGBAf(0, 0, 0, 0.08)
const REFERENCE = colorant"#9a9893"
# categorical slots 1–4 of the validated reference palette, fixed order (smallest DOFs first)
const SERIES = [colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100"]
const MARKERS = [:circle, :rect, :utriangle, :diamond]       # secondary encoding
const BACKEND_TITLE = Dict("cpu" => "CPU", "gpu" => "GPU", "mixed" => "CPU + GPU gekoppelt")

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = [Dict(zip(header, split(l, ","))) for l in lines[2:end] if !isempty(l)]
    return rows
end
num(r, k) = parse(Float64, r[k])
function tick_format(values)
    [v >= 100 ? string(round(Int, v)) :
     string(round(v; sigdigits = 2)) for v in values]
end

"Spreads end labels vertically (log scale) so that they do not overlap."
function spread_labels!(labels, gap)
    sort!(labels; by = l -> l[2])
    for k in 2:length(labels)
        x, y, t = labels[k]
        y_prev = labels[k - 1][2]
        log10(y) - log10(y_prev) < gap && (labels[k] = (x, y_prev * 10^gap, t))
    end
    return labels
end

dof_label(n) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => " ") * " DOF"

function plot_group(rows, dim, element, out_base)
    backends = filter(b -> any(r -> r["backend"] == b, rows), ["cpu", "gpu", "mixed"])
    # problem sizes by target: group by the realised DOFs of the smallest rank count
    sizes = sort(unique(round.(Int, num.(rows, "dofs"))))
    if length(sizes) > 4
        @warn "More than 4 problem sizes; plotting the 4 largest."
        sizes = sizes[(end - 3):end]
    end
    ranks_all = sort(unique(round.(Int, num.(rows, "ranks"))))

    fig = Figure(; size = (420 * length(backends) + 80, 760), backgroundcolor = SURFACE,
                 fontsize = 14)
    element_name = element == "corr" ? "Korrespondenz" : "bond-based"
    Label(fig[0, 1:length(backends)], "Skalierung über MPI-Ranks – $(dim)D, $element_name";
          fontsize = 18, font = :bold, color = TEXT, halign = :left)
    quantities = [("t_solve", "Lösungszeit [s]", 1.0),
        ("t_iteration", "Zeit pro Iteration [ms]", 1e3)]
    for (col, backend) in enumerate(backends),
        (row, (key, ylabel, scale)) in enumerate(quantities)
        ax = Axis(fig[row, col]; xscale = log2, yscale = log10,
                  xticks = (ranks_all, string.(ranks_all)), ytickformat = tick_format,
                  xlabel = row == 2 ? "Anzahl Ranks" : "",
                  ylabel = col == 1 ? ylabel : "",
                  title = row == 1 ? BACKEND_TITLE[backend] : "",
                  titlecolor = TEXT, xlabelcolor = TEXT_2, ylabelcolor = TEXT_2,
                  xticklabelcolor = TEXT_2, yticklabelcolor = TEXT_2,
                  xgridcolor = GRID, ygridcolor = GRID, xminorgridvisible = false,
                  topspinevisible = false, rightspinevisible = false,
                  leftspinecolor = TEXT_2, bottomspinecolor = TEXT_2,
                  backgroundcolor = SURFACE)
        end_labels = Tuple{Float64,Float64,String}[]
        y_all = Float64[]
        for (k, size) in enumerate(sizes)
            series = sort(filter(r -> r["backend"] == backend &&
                                      round(Int, num(r, "dofs")) == size, rows);
                          by = r -> num(r, "ranks"))
            isempty(series) && continue
            x = num.(series, "ranks")
            y = num.(series, key) .* scale
            lines!(ax, x, y; color = SERIES[k], linewidth = 2)
            scatter!(ax, x, y; color = SERIES[k], marker = MARKERS[k], markersize = 10,
                     strokecolor = SURFACE, strokewidth = 2)
            if key == "t_solve" && k == length(sizes) && length(x) > 1
                ideal = y[1] .* x[1] ./ x                           # perfect strong scaling
                lines!(ax, x, ideal; color = REFERENCE, linestyle = :dash, linewidth = 1.5)
            end
            push!(end_labels, (x[end], y[end], dof_label(size)))
            append!(y_all, y)
        end
        if !isempty(y_all)
            gap = 0.07 * max(log10(maximum(y_all)) - log10(minimum(y_all)), 0.3)
            for (x, y, t) in spread_labels!(end_labels, gap)
                text!(ax, x, y; text = "  " * t, color = TEXT_2, align = (:left, :center),
                      fontsize = 12)
            end
        end
        xlims!(ax, minimum(ranks_all) / 1.3, maximum(ranks_all) * 2.6)
    end
    entries = [[LineElement(; color = SERIES[k], linewidth = 2),
                   MarkerElement(; color = SERIES[k], marker = MARKERS[k], markersize = 10)
               ]
               for k in eachindex(sizes)]
    labels = dof_label.(sizes)
    push!(entries, [LineElement(; color = REFERENCE, linestyle = :dash, linewidth = 1.5)])
    push!(labels, "ideale Skalierung (größtes Problem)")
    Legend(fig[3, 1:length(backends)], entries, labels; orientation = :horizontal,
           framevisible = false, labelcolor = TEXT, backgroundcolor = SURFACE)
    path = "$(out_base)_$(dim)D_$(element).png"
    save(path, fig; px_per_unit = 2)
    println("Plot: ", path)
end

csv = ARGS[1]
rows = filter(r -> r["solved"] == "true", read_csv(csv))
isempty(rows) && error("No converged runs in $csv.")
base = splitext(csv)[1]
for dim in sort(unique(getindex.(rows, "dim"))),
    element in unique(getindex.(rows, "element"))
    group = filter(r -> r["dim"] == dim && r["element"] == element, rows)
    isempty(group) || plot_group(group, dim, element, base)
end
