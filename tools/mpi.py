import os
import sys
import subprocess
import numpy as np
from mpi4py import MPI
from typing import Sequence, Tuple, Union


def proc_id() -> int:
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def mpi_fork(n: int, bind_to_core: bool = False) -> None:
    """
    Re-launches the current script with workers linked by MPI. Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function from
    https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n: Number of process to split into.
        bind_to_core: Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    """Sum a scalar or vector over MPI processes."""

    return mpi_op(x, MPI.SUM)


def mpi_statistics_scalar(
        x: Sequence[float], with_min_and_max: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, float, float]]:
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
