import numpy as np
import numba
import pyperf

DEFAULT_ITERATIONS = 20000
DEFAULT_REFERENCE = 0  # sun index
PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

# Pre-computed body data as NumPy arrays
POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],  # sun
        [
            4.84143144246472090e00,
            -1.16032004402742839e00,
            -1.03622044471123109e-01,
        ],  # jupiter
        [
            8.34336671824457987e00,
            4.12479856412430479e00,
            -4.03523417114321381e-01,
        ],  # saturn
        [
            1.28943695621391310e01,
            -1.51111514016986312e01,
            -2.23307578892655734e-01,
        ],  # uranus
        [
            1.53796971148509165e01,
            -2.59193146099879641e01,
            1.79258772950371181e-01,
        ],  # neptune
    ],
    dtype=np.float64,
)

VELOCITIES = np.array(
    [
        [0.0, 0.0, 0.0],  # sun
        [
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
        ],  # jupiter
        [
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
        ],  # saturn
        [
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
        ],  # uranus
        [
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
        ],  # neptune
    ],
    dtype=np.float64,
)

MASSES = np.array(
    [
        SOLAR_MASS,  # sun
        9.54791938424326609e-04 * SOLAR_MASS,  # jupiter
        2.85885980666130812e-04 * SOLAR_MASS,  # saturn
        4.36624404335156298e-05 * SOLAR_MASS,  # uranus
        5.15138902046611451e-05 * SOLAR_MASS,  # neptune
    ],
    dtype=np.float64,
)


@numba.njit
def advance(dt, iterations, positions, velocities, masses):
    """Advance the system by updating positions and velocities."""
    n = positions.shape[0]

    for _ in range(iterations):
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]

                mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))

                mi_mag = masses[i] * mag
                mj_mag = masses[j] * mag

                velocities[i, 0] -= dx * mj_mag
                velocities[i, 1] -= dy * mj_mag
                velocities[i, 2] -= dz * mj_mag

                velocities[j, 0] += dx * mi_mag
                velocities[j, 1] += dy * mi_mag
                velocities[j, 2] += dz * mi_mag

        for i in range(n):
            positions[i, 0] += dt * velocities[i, 0]
            positions[i, 1] += dt * velocities[i, 1]
            positions[i, 2] += dt * velocities[i, 2]


@numba.njit
def report_energy(positions, velocities, masses):
    """Calculate the energy of the system."""
    n = positions.shape[0]
    e = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]

            e -= (masses[i] * masses[j]) / ((dx * dx + dy * dy + dz * dz) ** 0.5)

    for i in range(n):
        vx, vy, vz = velocities[i]
        e += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz)

    return e


@numba.njit
def offset_momentum(velocities, masses, ref_idx=0):
    """Adjust the momentum of the system relative to the reference body."""
    px = 0.0
    py = 0.0
    pz = 0.0

    for i in range(len(masses)):
        px -= velocities[i, 0] * masses[i]
        py -= velocities[i, 1] * masses[i]
        pz -= velocities[i, 2] * masses[i]

    velocities[ref_idx, 0] = px / masses[ref_idx]
    velocities[ref_idx, 1] = py / masses[ref_idx]
    velocities[ref_idx, 2] = pz / masses[ref_idx]


def setup(reference):
    """Initialize the system state."""
    # Create deep copies to avoid modifying global state
    positions = POSITIONS.copy()
    velocities = VELOCITIES.copy()
    masses = MASSES.copy()

    # Set up initial state (offset momentum)
    offset_momentum(velocities, masses, reference)

    return positions, velocities, masses


def bench_nbody(loops, reference, iterations):
    """Run the nbody benchmark."""
    positions, velocities, masses = setup(reference)

    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        report_energy(positions, velocities, masses)
        advance(0.01, iterations, positions, velocities, masses)
        report_energy(positions, velocities, masses)

    return pyperf.perf_counter() - t0


def print_results():
    positions, velocities, masses = setup(DEFAULT_REFERENCE)
    initial_energy = report_energy(positions, velocities, masses)
    advance(0.01, DEFAULT_ITERATIONS, positions, velocities, masses)
    final_energy = report_energy(positions, velocities, masses)

    print(f"Initial Energy: {initial_energy}")
    print(f"Final Energy: {final_energy}")
    print(f"Energy Delta: {abs(final_energy - initial_energy)}")


BENCHMARKS = {
    "nbody": (bench_nbody, DEFAULT_REFERENCE, DEFAULT_ITERATIONS),
}

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(quiet=False)
    for bench in sorted(BENCHMARKS):
        name = f"nbody_numba_{bench}"
        args = BENCHMARKS[bench]
        runner.bench_time_func(name, *args)
    # print_results()
