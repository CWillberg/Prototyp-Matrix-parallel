# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

"""
    PD_Matrix

Prototype for building and solving sparse, non-symmetric peridynamic system matrices on
CPU and GPU (KernelAbstractions). Stand-alone; meant to be included into PeriLab later as a
module (README A0.0). All data is delivered through function arguments (README A0.3).
"""
module PD_Matrix

include("Core/Sparse_CSR.jl")
include("Core/Matrix_Assembly.jl")
include("Core/Dof_Sets.jl")
include("Core/Operators.jl")
include("Core/Solver/Linear_Solver.jl")
include("MPI_communication/Halo_Exchange.jl")
include("Core/Distributed_Operators.jl")
include("Core/Solver/BiCGStab.jl")
include("Support/Memory_Report.jl")
include("Models/Elements/Elements.jl")
include("Core/Model_reduction/Guyan.jl")

using .Sparse_CSR
using .Matrix_Assembly
using .Dof_Sets
using .Operators
using .Linear_Solver
using .Halo_Exchange
using .Distributed_Operators
using .BiCGStab
using .Memory_Report
using .Elements
using .Guyan

export CSRMatrix, to_sparse
export AssemblyPattern, create_assembly_pattern, create_matrix, create_element_buffer,
       assemble!
export DofSet, create_dof_set, dof_indices
export ConstrainedOperator, SubOperator, constrained_rhs!, apply_prescribed!, reactions!
export LinearSolverWorkspace, solve_linear!
export HaloExchange, create_halo_exchange, exchange!, accumulate_ghosts!
export AbstractSpace, SerialSpace, MPISpace, inner, DistributedOperator, needs_accumulate
export BiCGStabWorkspace, jacobi!, solve_bicgstab!
export data_bytes, memory_report
export AbstractElement, element_dofs, compute_elements!, BondbasedElastic,
       CorrespondenceElastic,
       create_bondbased_data
export extract, reduce_matrices, SchurOperator

"""
    build_system(el, data)

Convenience for the symbolic phase on the host: pattern, matrix and element buffer for
element `el` with host data `data`.
"""
function build_system(el::AbstractElement, data, n_rows::Int64, n_cols::Int64 = n_rows)
    rows, cols = element_dofs(el, data)
    pattern = create_assembly_pattern(rows, cols, n_rows, n_cols)
    return pattern, create_matrix(pattern), create_element_buffer(pattern)
end
export build_system

end
