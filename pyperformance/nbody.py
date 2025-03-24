"""
N-body benchmark from the Computer Language Benchmarks Game.

This is intended to support Unladen Swallow's pyperf.py. Accordingly, it has been
modified from the Shootout version:
- Accept standard Unladen Swallow benchmark options.
- Run report_energy()/advance() in a loop.
- Reimplement itertools.combinations() to work with older Python versions.

Pulled from:
http://benchmarksgame.alioth.debian.org/u64q/program.php?test=nbody&lang=python3&id=1

Contributed by Kevin Carson.
Modified by Tupteq, Fredrik Johansson, and Daniel Nanz.
"""

import timeit
from pyperformance.utils import run_benchmark

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24
DEFAULT_ITERATIONS = 20000


def combinations(l):
    """Pure-Python implementation of itertools.combinations(l, 2)."""
    result = []
    for x in range(len(l) - 1):
        ls = l[x + 1 :]
        for y in ls:
            result.append((l[x], y))
    return result


BODIES = {
    "sun": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),
    "jupiter": (
        [4.84143144246472090e00, -1.16032004402742839e00, -1.03622044471123109e-01],
        [
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
        ],
        9.54791938424326609e-04 * SOLAR_MASS,
    ),
    "saturn": (
        [8.34336671824457987e00, 4.12479856412430479e00, -4.03523417114321381e-01],
        [
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
        ],
        2.85885980666130812e-04 * SOLAR_MASS,
    ),
    "uranus": (
        [1.28943695621391310e01, -1.51111514016986312e01, -2.23307578892655734e-01],
        [
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
        ],
        4.36624404335156298e-05 * SOLAR_MASS,
    ),
    "neptune": (
        [1.53796971148509165e01, -2.59193146099879641e01, 1.79258772950371181e-01],
        [
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
        ],
        5.15138902046611451e-05 * SOLAR_MASS,
    ),
}


def advance(dt, n, bodies, pairs):
    for i in range(n):
        for ([x1, y1, z1], v1, m1), ([x2, y2, z2], v2, m2) in pairs:
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            b1m = m1 * mag
            b2m = m2 * mag
            v1[0] -= dx * b2m
            v1[1] -= dy * b2m
            v1[2] -= dz * b2m
            v2[0] += dx * b1m
            v2[1] += dy * b1m
            v2[2] += dz * b1m
        for r, [vx, vy, vz], m in bodies:
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


def report_energy(bodies, pairs, e=0.0):
    for ((x1, y1, z1), v1, m1), ((x2, y2, z2), v2, m2) in pairs:
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    for r, [vx, vy, vz], m in bodies:
        e += m * (vx * vx + vy * vy + vz * vz) / 2.0
    return e


def offset_momentum(ref, bodies, px=0.0, py=0.0, pz=0.0):
    for r, [vx, vy, vz], m in bodies:
        px -= vx * m
        py -= vy * m
        pz -= vz * m
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


def bench_nbody(loops, iterations=DEFAULT_ITERATIONS, ref_index=0):
    """Benchmark for N-body simulation"""

    def run_nbody():
        # Create a deep copy of the system to avoid modifying global state
        system = []
        for body in BODIES.values():
            pos, vel, mass = body
            system.append([pos[:], vel[:], mass])

        pairs = combinations(system)

        # Set up initial state
        offset_momentum(system[ref_index], system)  # Assuming sun is first

        # Run simulation
        report_energy(system, pairs)
        advance(0.01, iterations, system, pairs)
        report_energy(system, pairs)

    return timeit.timeit(run_nbody, number=loops)


# Benchmark definitions
BENCHMARKS = {
    "nbody": (bench_nbody, DEFAULT_ITERATIONS, 0),
}


if __name__ == "__main__":
    for bench_name in sorted(BENCHMARKS):
        run_benchmark(bench_name, BENCHMARKS, 20)
