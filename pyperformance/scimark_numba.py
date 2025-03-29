import math
import pyperf
import numba
import numpy as np


@numba.njit
def random_matrix(n, seed=0):
    """Generate a random matrix using NumPy"""
    np.random.seed(seed)
    return np.random.random((n, n))


@numba.njit
def random_vector(n, seed=0):
    """Generate a random vector using NumPy"""
    np.random.seed(seed)
    return np.random.random(n)


@numba.njit
def SOR_execute(omega, G, cycles):
    """Implementation of SOR (Successive Over-Relaxation) algorithm"""
    n = G.shape[0]
    for _ in range(cycles):
        for y in range(1, n - 1):
            for x in range(1, n - 1):
                G[y, x] = (
                    omega
                    * 0.25
                    * (G[y - 1, x] + G[y + 1, x] + G[y, x - 1] + G[y, x + 1])
                    + (1.0 - omega) * G[y, x]
                )
    return G


def bench_SOR(loops, n, cycles):
    """Benchmark for SOR algorithm"""
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        G = np.zeros((n, n), dtype=np.float64)
        SOR_execute(1.25, G, cycles)

    return pyperf.perf_counter() - t0


@numba.njit
def SparseCompRow_matmult(y, val, row, col, x):
    """Sparse matrix multiplication implementation"""
    M = len(y)
    for r in range(M):
        sa = 0.0
        for i in range(row[r], row[r + 1]):
            sa += x[col[i]] * val[i]
        y[r] = sa

    return y


def bench_SparseMatMult(cycles, N, nz):
    """Benchmark for sparse matrix multiplication"""
    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)

    nr = nz // N
    anz = nr * N
    val = np.zeros(anz, dtype=np.float64)
    col = np.zeros(nz, dtype=np.int32)
    row = np.zeros(N + 1, dtype=np.int32)

    row[0] = 0
    for r in range(N):
        rowr = row[r]
        step = max(r // nr, 1)
        row[r + 1] = rowr + nr
        for i in range(nr):
            col[rowr + i] = i * step

    range_it = range(cycles)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        SparseCompRow_matmult(y, val, row, col, x)

    return pyperf.perf_counter() - t0


class MCState:
    """State for the Monte Carlo simulation"""

    def __init__(self, seed):
        self.seed = seed
        np.random.seed(seed)


@numba.njit
def MonteCarlo(Num_samples, seed=113):
    """Monte Carlo pi calculation implementation"""
    np.random.seed(seed)
    under_curve = 0
    for _ in range(Num_samples):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            under_curve += 1
    return float(under_curve) / Num_samples * 4.0


def bench_MonteCarlo(loops, Num_samples):
    """Benchmark for Monte Carlo pi calculation"""
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        MonteCarlo(Num_samples)

    return pyperf.perf_counter() - t0


@numba.njit
def LU_factor(A):
    """LU factorization implementation for NumPy arrays"""
    M, N = A.shape
    minMN = min(M, N)
    pivot = np.zeros(minMN, dtype=np.int32)

    for j in range(minMN):
        # Find pivot
        jp = j
        t = abs(A[j, j])
        for i in range(j + 1, M):
            ab = abs(A[i, j])
            if ab > t:
                jp = i
                t = ab
        pivot[j] = jp

        if abs(A[jp, j]) < 1e-12:
            raise Exception("factorization failed because of zero pivot")

        # Swap rows if necessary
        if jp != j:
            A[j, :], A[jp, :] = A[jp, :].copy(), A[j, :].copy()

        # Scale column
        if j < M - 1:
            recp = 1.0 / A[j, j]
            for k in range(j + 1, M):
                A[k, j] *= recp

        # Update trailing submatrix
        if j < minMN - 1:
            for ii in range(j + 1, M):
                for jj in range(j + 1, N):
                    A[ii, jj] -= A[ii, j] * A[j, jj]

    return A, pivot


def bench_LU(cycles, N):
    """Benchmark for LU decomposition"""
    np.random.seed(7)
    A = np.random.random((N, N))
    range_it = range(cycles)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        LU_factor(A.copy())

    return pyperf.perf_counter() - t0


@numba.njit
def int_log2(n):
    """Integer log base 2"""
    k = 1
    log = 0
    while k < n:
        k *= 2
        log += 1
    if n != 1 << log:
        raise ValueError(f"FFT: Data length is not a power of 2: {n}")
    return log


@numba.njit
def FFT_bitreverse(data):
    """Bit reverse routine for FFT"""
    N = len(data)
    n = N // 2
    nm1 = n - 1
    j = 0
    for i in range(nm1):
        ii = i << 1
        jj = j << 1
        k = n >> 1
        if i < j:
            # Swap elements
            tmp_real = data[ii]
            tmp_imag = data[ii + 1]
            data[ii] = data[jj]
            data[ii + 1] = data[jj + 1]
            data[jj] = tmp_real
            data[jj + 1] = tmp_imag
        while k <= j:
            j -= k
            k >>= 1
        j += k

    return data


@numba.njit
def FFT_transform_internal(data, direction):
    """Internal FFT transform implementation"""
    N = len(data)
    n = N // 2
    if n <= 1:
        return data

    logn = int_log2(n)
    data = FFT_bitreverse(data)

    # Apply FFT recursion
    dual = 1
    for bit in range(logn):
        w_real = 1.0
        w_imag = 0.0
        theta = 2.0 * direction * math.pi / (2.0 * float(dual))
        s = math.sin(theta)
        t = math.sin(theta / 2.0)
        s2 = 2.0 * t * t

        # Process dual elements
        for b in range(0, n, 2 * dual):
            i = 2 * b
            j = 2 * (b + dual)
            wd_real = data[j]
            wd_imag = data[j + 1]
            data[j] = data[i] - wd_real
            data[j + 1] = data[i + 1] - wd_imag
            data[i] += wd_real
            data[i + 1] += wd_imag

        # Process remaining elements
        for a in range(1, dual):
            tmp_real = w_real - s * w_imag - s2 * w_real
            tmp_imag = w_imag + s * w_real - s2 * w_imag
            w_real = tmp_real
            w_imag = tmp_imag

            for b in range(0, n, 2 * dual):
                i = 2 * (b + a)
                j = 2 * (b + a + dual)
                z1_real = data[j]
                z1_imag = data[j + 1]
                wd_real = w_real * z1_real - w_imag * z1_imag
                wd_imag = w_real * z1_imag + w_imag * z1_real
                data[j] = data[i] - wd_real
                data[j + 1] = data[i + 1] - wd_imag
                data[i] += wd_real
                data[i + 1] += wd_imag

        dual *= 2

    return data


@numba.njit
def FFT_transform(data):
    """Forward FFT transform"""
    return FFT_transform_internal(data, -1)


@numba.njit
def FFT_inverse(data):
    """Inverse FFT transform"""
    n = len(data) // 2
    data = FFT_transform_internal(data, +1)
    # Normalize
    norm = 1 / float(n)
    for i in range(len(data)):
        data[i] *= norm
    return data


def bench_FFT(loops, N, cycles):
    """Benchmark for FFT"""
    twoN = 2 * N
    np.random.seed(7)
    init_vec = np.random.random(twoN)
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        x = init_vec.copy()
        for i in range(cycles):
            x = FFT_transform(x)
            x = FFT_inverse(x)

    return pyperf.perf_counter() - t0


# Benchmark definitions
BENCHMARKS = {
    "sor": (bench_SOR, 100, 10),
    "sparse_mat_mult": (bench_SparseMatMult, 1000, 50 * 1000),
    "monte_carlo": (bench_MonteCarlo, 100 * 1000),
    "lu": (bench_LU, 100),
    "fft": (bench_FFT, 1024, 50),
}

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(quiet=False)
    for bench in sorted(BENCHMARKS):
        name = "scimark_numba_%s" % bench
        args = BENCHMARKS[bench]
        runner.bench_time_func(name, *args)
