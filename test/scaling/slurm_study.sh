#!/bin/bash
# SPDX-FileCopyrightText: 2026 Christian Willberg
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Example SLURM job for the study: 1 CPU rank as reference, MPI only, CUDA only, mixed. Adapt partition, account, modules and sizes to
# your cluster. Submit from the repository root:
#
#   sbatch test/scaling/slurm_study.sh
#
# One-time setup on the cluster (login node):
#   julia --project=test/scaling -e 'using Pkg; Pkg.instantiate()'
#   julia --project=test/scaling/plot -e 'using Pkg; Pkg.instantiate()'
#   # use the cluster MPI instead of the bundled MPICH (needed for the interconnect and
#   # for CUDA-aware MPI):
#   julia --project=test/scaling -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
#
#SBATCH --job-name=pd_matrix_study
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=results/study/slurm_%j.out
##SBATCH --partition=gpu
##SBATCH --account=my_account

set -euo pipefail
# module load julia openmpi cuda          # cluster specific

mkdir -p results/study
export JULIA_NUM_THREADS=1

# reference (1 CPU rank), MPI only, CUDA only, mixed - per problem size
julia --project=test/scaling test/scaling/run_study.jl \
    launcher=srun \
    dofs=100000,1000000,4000000 \
    dims=3 \
    cpu_ranks=2,4,8,16,32 \
    gpu_ranks=1,2,4,8 \
    mixed_ranks=8,16,32 \
    gpu_weight=8 \
    itmax=20000 \
    out=results/study
