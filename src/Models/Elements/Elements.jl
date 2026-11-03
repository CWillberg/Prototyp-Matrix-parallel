# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

module Elements

include("Element_Interface.jl")
using .Element_Interface

export AbstractElement, element_name, element_dofs, compute_element!, compute_elements!,
       n_elements

# Plugin discovery as in PeriLab: every file in a sub-folder defines one element module.
# A new element is a new file; nothing else has to be edited (README A0.2).
for folder in sort(readdir(@__DIR__; join = true))
    isdir(folder) || continue
    for file in sort(readdir(folder; join = true))
        endswith(file, ".jl") && include(file)
    end
end

using .Bondbased_Elastic: BondbasedElastic, create_bondbased_data
export BondbasedElastic, create_bondbased_data

end
