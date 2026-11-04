# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Plots a study CSV (run_study.jl). One figure per dimension and element; one column per
# problem size; rows: solve time and speedup relative to the 1-CPU reference of the same
# size. Series: MPI (CPU ranks), CUDA (GPU ranks), CPU + GPU coupled.
# Also prints the table of the study. Writes <csv name>_<dim>D_<element>.png.

using CairoMakie
using Printf: @printf

const TEXT = colorant"#0b0b0b"
const TEXT_2 = colorant"#52514e"
const SURFACE = colorant"#fcfcfb"
const GRID = RGBAf(0, 0, 0, 0.08)
const REFERENCE = colorant"#9a9893"
# categorical slots 1–3 of the validated reference palette, fixed order
const SERIES = [("cpu", "MPI (nur CPU)", colorant"#2a78d6", :circle),
                ("gpu", "CUDA (nur GPU)", colorant"#eb6834", :rect),
                ("mixed", "CPU + GPU gekoppelt", colorant"#1baf7a", :utriangle)]

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    return [Dict(zip(header, split(l, ","))) for l in lines[2:end] if !isempty(l)]
end
num(r, k) = parse(Float64, r[k])
int(r, k) = round(Int, num(r, k))
dof_label(n) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => " ") * " DOF"
tick_format(values) = [v >= 100 ? string(round(Int, v)) : string(round(v; sigdigits = 2))
                       for v in values]

function plot_group(rows, dim, element, out_base)
    sizes = sort(unique(int.(rows, "dofs")))
    reference = Dict(s => num(only(filter(r -> int(r, "dofs") == s &&
                                              r["backend"] == "cpu" &&
                                              int(r, "ranks") == 1, rows)), "t_solve")
                     for s in sizes
                     if any(r -> int(r, "dofs") == s && r["backend"] == "cpu" &&
                                 int(r, "ranks") == 1, rows))
    ranks_all = sort(unique(int.(rows, "ranks")))
    fig = Figure(; size = (400 * length(sizes) + 90, 760), backgroundcolor = SURFACE,
                 fontsize = 14)
    element_name = element == "corr" ? "Korrespondenz" : "bond-based"
    Label(fig[0, 1:length(sizes)],
          "Lösung gegenüber 1 CPU-Rank – $(dim)D, $element_name"; fontsize = 18,
          font = :bold, color = TEXT, halign = :left)
    println("\n$(dim)D $element_name: solve time [s] and speedup vs. 1 CPU rank")
    for (col, size) in enumerate(sizes)
        haskey(reference, size) || continue
        t_ref = reference[size]
        axes_ = [Axis(fig[row, col]; xscale = log2, yscale = log10,
                      xticks = (ranks_all, string.(ranks_all)), ytickformat = tick_format,
                      xlabel = row == 2 ? "Anzahl Ranks" : "",
                      ylabel = col == 1 ? label : "",
                      title = row == 1 ? dof_label(size) : "", titlecolor = TEXT,
                      xlabelcolor = TEXT_2, ylabelcolor = TEXT_2,
                      xticklabelcolor = TEXT_2, yticklabelcolor = TEXT_2,
                      xgridcolor = GRID, ygridcolor = GRID, topspinevisible = false,
                      rightspinevisible = false, leftspinecolor = TEXT_2,
                      bottomspinecolor = TEXT_2, backgroundcolor = SURFACE)
                 for (row, label) in enumerate(["Lösungszeit [s]",
                                                "Speedup gegenüber 1 CPU"])]
        hlines!(axes_[1], [t_ref]; color = REFERENCE, linestyle = :dash, linewidth = 1.5)
        hlines!(axes_[2], [1.0]; color = REFERENCE, linestyle = :dash, linewidth = 1.5)
        @printf("  %-12s %-22s %6s %10s %9s\n", dof_label(size), "", "ranks", "t_solve",
                "speedup")
        @printf("  %-12s %-22s %6d %10.4f %9.2f\n", "", "Referenz 1 CPU", 1, t_ref, 1.0)
        for (backend, name, color, marker) in SERIES
            series = sort(filter(r -> int(r, "dofs") == size && r["backend"] == backend &&
                                     !(backend == "cpu" && int(r, "ranks") == 1), rows);
                          by = r -> num(r, "ranks"))
            backend == "cpu" && (series = vcat(filter(r -> int(r, "dofs") == size &&
                                                         r["backend"] == "cpu" &&
                                                         int(r, "ranks") == 1, rows),
                                               series))
            isempty(series) && continue
            x = num.(series, "ranks")
            t = num.(series, "t_solve")
            for (ax, y) in zip(axes_, (t, t_ref ./ t))
                lines!(ax, x, y; color = color, linewidth = 2)
                scatter!(ax, x, y; color = color, marker = marker, markersize = 10,
                         strokecolor = SURFACE, strokewidth = 2)
            end
            for r in series
                backend == "cpu" && int(r, "ranks") == 1 && continue
                @printf("  %-12s %-22s %6d %10.4f %9.2f\n", "", name, int(r, "ranks"),
                        num(r, "t_solve"), t_ref / num(r, "t_solve"))
            end
        end
    end
    entries = [[LineElement(; color = c, linewidth = 2),
                MarkerElement(; color = c, marker = m, markersize = 10)]
               for (_, _, c, m) in SERIES]
    labels = [n for (_, n, _, _) in SERIES]
    push!(entries, [LineElement(; color = REFERENCE, linestyle = :dash, linewidth = 1.5)])
    push!(labels, "Referenz: 1 CPU-Rank")
    Legend(fig[3, 1:length(sizes)], entries, labels; orientation = :horizontal,
           framevisible = false, labelcolor = TEXT, backgroundcolor = SURFACE)
    path = "$(out_base)_$(dim)D_$(element).png"
    save(path, fig; px_per_unit = 2)
    println("Plot: ", path)
end

csv = ARGS[1]
rows = filter(r -> r["solved"] == "true", read_csv(csv))
isempty(rows) && error("No converged runs in $csv.")
base = splitext(csv)[1]
for dim in sort(unique(getindex.(rows, "dim"))), element in unique(getindex.(rows, "element"))
    group = filter(r -> r["dim"] == dim && r["element"] == element, rows)
    isempty(group) || plot_group(group, dim, element, base)
end
