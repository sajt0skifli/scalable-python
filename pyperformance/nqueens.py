"""Simple, brute-force N-Queens solver."""

import pyperf
import numba
import numpy as np


# Pure-Python implementation of itertools.permutations().
def permutations(iterable, r=None):
    """permutations(range(3), 2) --> (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)"""
    pool = tuple(iterable)
    n = len(pool)
    if r is None:
        r = n
    indices = list(range(n))
    cycles = list(range(n - r + 1, n + 1))[::-1]
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1 :] + indices[i : i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


@numba.njit
def permutations_numba(iterable, r=None):
    # Convert the iterable to a NumPy array
    pool = np.array(iterable)
    n = pool.shape[0]
    if r is None:
        r = n

    result = []
    indices = list(range(n))
    cycles = [n - i for i in range(r)]

    # Build the first permutation as a NumPy array
    temp = np.empty(r, dtype=pool.dtype)
    for i in range(r):
        temp[i] = pool[indices[i]]
    result.append(temp.copy())

    while n:
        broke = False
        for i in range(r - 1, -1, -1):
            cycles[i] -= 1
            if cycles[i] == 0:
                new_indices = indices[i + 1 :] + indices[i : i + 1]
                for j in range(len(new_indices)):
                    indices[i + j] = new_indices[j]
                cycles[i] = n - i
            else:
                j = cycles[i]
                # Swap indices[i] and indices[-j]
                indices[i], indices[-j] = indices[-j], indices[i]
                for k in range(r):
                    temp[k] = pool[indices[k]]
                result.append(temp.copy())
                broke = True
                break
        if not broke:
            break
    return result


# From http://code.activestate.com/recipes/576647/
def n_queens(queen_count):
    """N-Queens solver.

    Args:
        queen_count: the number of queens to solve for. This is also the
            board size.

    Yields:
        Solutions to the problem. Each yielded value is looks like
        (3, 8, 2, 1, 4, ..., 6) where each number is the column position for the
        queen, and the index into the tuple indicates the row.
    """
    cols = range(queen_count)
    for vec in permutations(cols):
        if (
            queen_count
            == len(set(vec[i] + i for i in cols))
            == len(set(vec[i] - i for i in cols))
        ):
            yield vec


@numba.njit
def n_queens_numba(queen_count):
    cols = list(range(queen_count))
    result = []
    perms = permutations_numba(cols)
    for vec in perms:
        valid = True
        diag1 = np.empty(queen_count, dtype=np.int64)
        diag2 = np.empty(queen_count, dtype=np.int64)
        for i in range(queen_count):
            diag1[i] = vec[i] + i
            diag2[i] = vec[i] - i
        for i in range(queen_count):
            for j in range(i + 1, queen_count):
                if diag1[i] == diag1[j] or diag2[i] == diag2[j]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            result.append(vec)
    return result


def bench_nqueens(loops, queen_count=8):
    """Benchmark for N-Queens solver"""
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        list(n_queens(queen_count))

    return pyperf.perf_counter() - t0


def bench_nqueens_numba(loops, queen_count=8):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        n_queens_numba(queen_count)

    return pyperf.perf_counter() - t0


# Benchmark definitions
BENCHMARKS = {
    "nqueens": (bench_nqueens, 8),
    "nqueens_numba": (bench_nqueens_numba, 8),
}


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(quiet=False)
    for bench in sorted(BENCHMARKS):
        name = "nqueens_%s" % bench
        args = BENCHMARKS[bench]
        runner.bench_time_func(name, *args)
