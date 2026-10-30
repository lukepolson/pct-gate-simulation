"""What the runners share: one BLAS thread per process, below-normal priority, and the free memory they keep."""
import os

import psutil

#: Free RAM and free commit (GB) a runner keeps: it starts a process only if both stay above this afterwards.
FLOOR_GB = float(os.environ.get('PCTSIM_FLOOR_GB', 10.0))
#: Each GATE or convert process uses one BLAS and OpenMP thread: otherwise numpy's and scipy's OpenBLAS each reserve a
#: buffer per core at import (1.6 GB of commit on 24 cores, for a single-threaded process).
ONE_THREAD = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                               'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')}
#: subprocess.Popen keywords that start a process at below-normal priority.
LOW_PRIORITY = {'creationflags': 0x4000} if os.name == 'nt' else {'preexec_fn': lambda: os.nice(10)}


def lower_this_process():
    """This process to below-normal priority (worker pools)."""
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)


def free_gb():
    """(free RAM, free commit) in GB: on Windows the commit charge's headroom, elsewhere free RAM plus free swap."""
    if os.name == 'nt':
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / 1e9, s.ullAvailPageFile / 1e9
    ram = psutil.virtual_memory().available
    return ram / 1e9, (ram + psutil.swap_memory().free) / 1e9
