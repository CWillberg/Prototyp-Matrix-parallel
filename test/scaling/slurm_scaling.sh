#!/bin/bash
# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Example SLURM job for the scaling study. Adapt partition, account, modules and sizes to
# your cluster. Submit from the repository root:
#
#   sbatch test/scaling/slurm_scaling.sh
#
# One-time setup on the cluster (login node):
#   julia --project=test/scaling -e 'using Pkg; Pkg.instantiate()'
#   julia --project=test/scaling/plot -e 'using Pkg; Pkg.instantiate()'
#   # use the cluster MPI instead of the bundled MPICH (needed for the interconnect and
#   # for CUDA-aware MPI):
#   julia --project=test/scaling -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
#
#SBATCH --job-name=pd_matrix_scaling
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=results/scaling/slurm_%j.out
##SBATCH --partition=gpu
##SBATCH --account=my_account

set -euo pipefail
# module load julia openmpi cuda          # cluster specific

mkdir -p results/scaling
export JULIA_NUM_THREADS=1

# all combinations inside this allocation; srun starts the ranks of every run
julia --project=test/scaling test/scaling/run_scaling.jl \
    launcher=srun \
    ranks=1,2,4,8,16,32 \
    dofs=100000,1000000,4000000 \
    dims=3 \
    backends=cpu,gpu,mixed \
    gpu_weight=8 \
    itmax=20000 \
    out=results/scaling
