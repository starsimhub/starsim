"""
Large blocks of memory that a sim frees and asks for again, kept for reuse rather than given back to the system

Each step of a large sim makes and frees arrays of megabytes (new edges, their copies without the dead, and temporaries).
By default, glibc's malloc gives such blocks back to the system when they are freed (as mappings of their own above 128 kB
to 32 MB, or by trimming the top of the heap), so each new one starts with a page fault for every 4 kB, which the kernel
fills with zeros: 15-25% of the run of a sim with a random network (see v4_design/oct25/backlog/perf_jax.md). Keeping them in the
heap, as jemalloc and tcmalloc do, avoids that: memory freed is still reused, but the process keeps its peak size until it
ends (or `release()` is called). Linux with glibc only; elsewhere nothing changes.

Used with: agents; declared and coded modules
"""
import sys
import ctypes

M_TRIM_THRESHOLD = -1 # glibc's mallopt() parameters
M_MMAP_THRESHOLD = -3


def libc():
    """ glibc's malloc functions, or None """
    if not sys.platform.startswith('linux'):
        return None
    try:
        lib = ctypes.CDLL(None) # The process's own symbols, which include malloc's (finding libc by name would start a subprocess)
        lib.mallopt, lib.malloc_trim
    except (OSError, AttributeError): # E.g. musl, which has neither
        return None
    return lib


def keep():
    """
    Keep freed blocks (up to 1 GB each, or 32 MB on older glibc, which refuses more) in the heap for reuse, rather than giving them back to
    the system; returns whether it was set. This overrides `MALLOC_TRIM_THRESHOLD_` and `MALLOC_MMAP_THRESHOLD_`, which Dask's nanny sets for its workers.
    """
    lib = libc()
    return bool(lib and (lib.mallopt(M_MMAP_THRESHOLD, 2**30) or lib.mallopt(M_MMAP_THRESHOLD, 2**25)) and lib.mallopt(M_TRIM_THRESHOLD, 2**31 - 1))


def release():
    """ Give the memory that is free in the heap back to the system """
    lib = libc()
    return bool(lib and lib.malloc_trim(0))
