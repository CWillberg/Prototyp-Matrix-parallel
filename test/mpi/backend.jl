# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause

# Backend of a rank: `cpu`, `gpu` (every rank on a GPU) or `mixed` (one rank per GPU of a
# node on the GPU, the other ranks of the node on the CPU). GPUs are assigned by the rank
# within the node: device = local_rank % number_of_devices. Expects `using MPI` and, for
# `gpu`/`mixed`, `using CUDA` before including this file.

"""
    select_backend(mode, comm) -> (array_type, on_gpu, device)

Chooses the array type of this rank and selects its CUDA device.
"""
function select_backend(mode::AbstractString, comm::MPI.Comm)
    mode in ("cpu", "gpu", "mixed") || error("Unknown backend $mode (cpu | gpu | mixed).")
    mode == "cpu" && return (Array, false, -1)
    rank = MPI.Comm_rank(comm)
    node_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    local_rank = MPI.Comm_rank(node_comm)
    n_devices = Main.CUDA.functional() ? length(Main.CUDA.devices()) : 0
    if n_devices == 0
        mode == "gpu" && error("Backend gpu requested, but CUDA is not functional.")
        return (Array, false, -1)
    end
    if mode == "gpu" || local_rank < n_devices
        device = local_rank % n_devices
        Main.CUDA.device!(device)
        return (Main.CUDA.CuArray, true, device)
    end
    return (Array, false, -1)
end
