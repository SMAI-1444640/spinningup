import os, subprocess, sys
import numpy as np

# Try to import MPI, but provide fallback for single-process execution
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1:
        return
    if not MPI_AVAILABLE:
        print("Warning: MPI not available. Running with single process.")
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    if MPI_AVAILABLE:
        print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))
    else:
        print(('Message from 0: %s \t '%string)+str(m))

def proc_id():
    """Get rank of calling process."""
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_rank()
    return 0

def allreduce(*args, **kwargs):
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)
    # For single process, just copy input to output
    if len(args) >= 2:
        np.copyto(args[1], args[0])

def num_procs():
    """Count active MPI processes."""
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_size()
    return 1

def broadcast(x, root=0):
    if MPI_AVAILABLE:
        MPI.COMM_WORLD.Bcast(x, root=root)
    # For single process, no-op

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    if MPI_AVAILABLE:
        allreduce(x, buff, op=op)
    else:
        np.copyto(buff, x)
    return buff[0] if scalar else buff

def mpi_sum(x):
    if MPI_AVAILABLE:
        return mpi_op(x, MPI.SUM)
    # For single process, just return the value
    x = np.asarray(x, dtype=np.float32)
    return x[0] if x.ndim == 1 and len(x) == 1 else x

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    # Input x: The input x is a 1D NumPy array of scalar values.
    # In the line mpi_statistics_scalar(self.adv_buf), the input is self.adv_buf.
    # self.adv_buf is a 1D array (a vector) where each element is the calculated
    # advantage for a single timestep. Its length is local_steps_per_epoch.
    x = np.array(x, dtype=np.float32)
    
    if MPI_AVAILABLE:
        global_sum, global_n = mpi_sum([np.sum(x), len(x)])
        mean = global_sum / global_n

        global_sum_sq = mpi_sum(np.sum((x - mean)**2))
        std = np.sqrt(global_sum_sq / global_n)  # compute global std

        if with_min_and_max:
            global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
            global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
            return mean, std, global_min, global_max
        return mean, std
    else:
        # Single process fallback
        mean = np.mean(x)
        std = np.std(x)
        if with_min_and_max:
            return mean, std, np.min(x), np.max(x)
        return mean, std