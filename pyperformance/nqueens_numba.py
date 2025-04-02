"""Simple, brute-force N-Queens solver."""

import pyperf
import numba
import numpy as np


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


def bench_nqueens_numba(loops, queen_count=8):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        list(n_queens_numba(queen_count))

    return pyperf.perf_counter() - t0


def print_results():
    res = list(n_queens_numba(8))
    for i, r in enumerate(res):
        print(f"Solution {i}: {r}")


# Benchmark definitions
BENCHMARKS = {
    "nqueens_numba": (bench_nqueens_numba, 8),
}


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(quiet=False)
    for bench in sorted(BENCHMARKS):
        name = "nqueens_%s" % bench
        args = BENCHMARKS[bench]
        runner.bench_time_func(name, *args)
