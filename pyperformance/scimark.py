from array import array
import math
import timeit
import numba

from typing import Callable, Tuple, Union

from pyperformance.utils import run_benchmark


class Array2D:
    """Two-dimensional array implementation"""

    def __init__(self, width: int, height: int, data=None):
        self.width = width
        self.height = height
        self.data = array("d", [0]) * (width * height)
        if data is not None:
            self.setup(data)

    def _idx(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        raise IndexError(f"Coordinates out of bounds: ({x}, {y})")

    def __getitem__(self, x_y: Tuple[int, int]) -> float:
        x, y = x_y
        return self.data[self._idx(x, y)]

    def __setitem__(self, x_y: Tuple[int, int], val: float):
        x, y = x_y
        self.data[self._idx(x, y)] = val

    def setup(self, data) -> "Array2D":
        for y in range(self.height):
            for x in range(self.width):
                self[x, y] = data[y][x]
        return self

    def indexes(self):
        for y in range(self.height):
            for x in range(self.width):
                yield x, y

    def copy_data_from(self, other: "Array2D"):
        self.data[:] = other.data[:]


class ArrayList(Array2D):
    """Array2D variant with list-of-arrays implementation"""

    def __init__(self, width: int, height: int, data=None):
        self.width = width
        self.height = height
        self.data = [array("d", [0]) * width for _ in range(height)]
        if data is not None:
            self.setup(data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.data[idx[1]][idx[0]]
        else:
            return self.data[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            self.data[idx[1]][idx[0]] = val
        else:
            self.data[idx] = val

    def copy_data_from(self, other: "ArrayList"):
        for l1, l2 in zip(self.data, other.data):
            l1[:] = l2


class Random:
    """Random number generator"""

    MDIG = 32
    ONE = 1
    m1 = (ONE << (MDIG - 2)) + ((ONE << (MDIG - 2)) - ONE)
    m2 = ONE << MDIG // 2
    dm1 = 1.0 / float(m1)

    def __init__(self, seed: int):
        self.initialize(seed)
        self.left = 0.0
        self.right = 1.0
        self.width = 1.0
        self.haveRange = False

    def initialize(self, seed: int):
        self.seed = seed
        seed = abs(seed)
        jseed = min(seed, self.m1)
        if jseed % 2 == 0:
            jseed -= 1
        k0 = 9069 % self.m2
        k1 = 9069 // self.m2
        j0 = jseed % self.m2
        j1 = jseed // self.m2
        self.m = array("d", [0]) * 17
        for iloop in range(17):
            jseed = j0 * k0
            j1 = (jseed // self.m2 + j0 * k1 + j1 * k0) % (self.m2 // 2)
            j0 = jseed % self.m2
            self.m[iloop] = j0 + self.m2 * j1
        self.i = 4
        self.j = 16

    def nextDouble(self) -> float:
        I, J, m = self.i, self.j, self.m
        k = m[I] - m[J]
        if k < 0:
            k += self.m1
        self.m[J] = k

        if I == 0:
            I = 16
        else:
            I -= 1
        self.i = I

        if J == 0:
            J = 16
        else:
            J -= 1
        self.j = J

        return (
            self.left + self.dm1 * float(k) * self.width
            if self.haveRange
            else self.dm1 * float(k)
        )

    def RandomMatrix(self, a: Union[Array2D, ArrayList]):
        for x, y in a.indexes():
            a[x, y] = self.nextDouble()
        return a

    def RandomVector(self, n: int):
        return array("d", [self.nextDouble() for _ in range(n)])


def copy_vector(vec):
    """Create a copy of a vector"""
    vec2 = array("d")
    vec2[:] = vec[:]
    return vec2


# SOR (Successive Over-Relaxation) Benchmark
def SOR_execute(omega: float, G: Union[Array2D, ArrayList], cycles: int, _):
    """Implementation of the SOR algorithm"""
    for _ in range(cycles):
        for y in range(1, G.height - 1):
            for x in range(1, G.width - 1):
                G[x, y] = (
                    omega
                    * 0.25
                    * (G[x, y - 1] + G[x, y + 1] + G[x - 1, y] + G[x + 1, y])
                    + (1.0 - omega) * G[x, y]
                )


def bench_SOR(loops: int, n: int, cycles: int, Array: Callable):
    """Benchmark for SOR algorithm"""

    def run_sor():
        G = Array(n, n)
        SOR_execute(1.25, G, cycles, Array)

    return timeit.timeit(run_sor, number=loops)


# Sparse Matrix Multiplication Benchmark
def SparseCompRow_matmult(
    M: int, y: array, val: array, row: array, col: array, x: array, num_iterations: int
):
    """Sparse matrix multiplication implementation"""
    for _ in range(num_iterations):
        for r in range(M):
            sa = 0.0
            for i in range(row[r], row[r + 1]):
                sa += x[col[i]] * val[i]
            y[r] = sa


def bench_SparseMatMult(cycles: int, N: int, nz: int):
    """Benchmark for sparse matrix multiplication"""
    x = array("d", [0]) * N
    y = array("d", [0]) * N

    nr = nz // N
    anz = nr * N
    val = array("d", [0]) * anz
    col = array("i", [0]) * nz
    row = array("i", [0]) * (N + 1)

    row[0] = 0
    for r in range(N):
        rowr = row[r]
        step = r // nr
        row[r + 1] = rowr + nr
        if step < 1:
            step = 1
        for i in range(nr):
            col[rowr + i] = i * step

    def run_sparse():
        SparseCompRow_matmult(N, y, val, row, col, x, cycles)

    return timeit.timeit(run_sparse, number=1)


# Monte Carlo Benchmark
def MonteCarlo(Num_samples: int) -> float:
    """Monte Carlo pi calculation implementation"""
    rnd = Random(113)
    under_curve = 0
    for _ in range(Num_samples):
        x = rnd.nextDouble()
        y = rnd.nextDouble()
        if x * x + y * y <= 1.0:
            under_curve += 1
    return float(under_curve) / Num_samples * 4.0


def bench_MonteCarlo(loops: int, Num_samples: int):
    """Benchmark for Monte Carlo pi calculation"""

    def run_monte_carlo():
        for _ in range(loops):
            MonteCarlo(Num_samples)

    return timeit.timeit(run_monte_carlo, number=1)


# LU Decomposition Benchmark
def LU_factor(A: ArrayList, pivot: array):
    """LU factorization implementation"""
    M, N = A.height, A.width
    minMN = min(M, N)
    for j in range(minMN):
        # Find pivot
        jp = j
        t = abs(A[j][j])
        for i in range(j + 1, M):
            ab = abs(A[i][j])
            if ab > t:
                jp = i
                t = ab
        pivot[j] = jp

        if A[jp][j] == 0:
            raise Exception("factorization failed because of zero pivot")

        # Swap rows if necessary
        if jp != j:
            A[j], A[jp] = A[jp], A[j]

        # Scale column
        if j < M - 1:
            recp = 1.0 / A[j][j]
            for k in range(j + 1, M):
                A[k][j] *= recp

        # Update trailing submatrix
        if j < minMN - 1:
            for ii in range(j + 1, M):
                for jj in range(j + 1, N):
                    A[ii][jj] -= A[ii][j] * A[j][jj]


def LU(lu: ArrayList, A: ArrayList, pivot: array):
    """LU decomposition wrapper"""
    lu.copy_data_from(A)
    LU_factor(lu, pivot)


def bench_LU(cycles: int, N: int):
    """Benchmark for LU decomposition"""
    rnd = Random(7)
    A = rnd.RandomMatrix(ArrayList(N, N))
    lu = ArrayList(N, N)
    pivot = array("i", [0]) * N

    def run_lu():
        for _ in range(cycles):
            LU(lu, A, pivot)

    return timeit.timeit(run_lu, number=1)


# FFT Benchmark
def int_log2(n: int) -> int:
    """Integer log base 2"""
    k = 1
    log = 0
    while k < n:
        k *= 2
        log += 1
    if n != 1 << log:
        raise Exception(f"FFT: Data length is not a power of 2: {n}")
    return log


def FFT_bitreverse(N: int, data: array):
    """Bit reverse routine for FFT"""
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


def FFT_transform_internal(N: int, data: array, direction: int):
    """Internal FFT transform implementation"""
    n = N // 2
    if n <= 1:
        return

    logn = int_log2(n)
    FFT_bitreverse(N, data)

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


def FFT_transform(N: int, data: array):
    """Forward FFT transform"""
    FFT_transform_internal(N, data, -1)


def FFT_inverse(N: int, data: array):
    """Inverse FFT transform"""
    n = N // 2
    FFT_transform_internal(N, data, +1)
    # Normalize
    norm = 1 / float(n)
    for i in range(N):
        data[i] *= norm


def bench_FFT(loops: int, N: int, cycles: int):
    """Benchmark for FFT"""
    twoN = 2 * N
    init_vec = Random(7).RandomVector(twoN)

    def run_fft():
        x = copy_vector(init_vec)
        for _ in range(cycles):
            FFT_transform(twoN, x)
            FFT_inverse(twoN, x)

    return timeit.timeit(lambda: [run_fft() for _ in range(loops)], number=1)


# Benchmark definitions
BENCHMARKS = {
    "sor": (bench_SOR, 100, 10, Array2D),
    "sparse_mat_mult": (bench_SparseMatMult, 1000, 50 * 1000),
    "monte_carlo": (bench_MonteCarlo, 100 * 1000),
    "lu": (bench_LU, 100),
    "fft": (bench_FFT, 1024, 50),
}

if __name__ == "__main__":
    for bench_name in sorted(BENCHMARKS):
        run_benchmark(bench_name, BENCHMARKS, 20)
