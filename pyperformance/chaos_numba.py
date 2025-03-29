import math
import random
import pyperf
import numba
import numpy as np

DEFAULT_THICKNESS = 0.25
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_ITERATIONS = 5000
DEFAULT_RNG_SEED = 1234


@numba.njit
def gvector_mag(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)


@numba.njit
def gvector_dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


@numba.njit
def gvector_add(x1, y1, z1, x2, y2, z2):
    return x1 + x2, y1 + y2, z1 + z2


@numba.njit
def gvector_sub(x1, y1, z1, x2, y2, z2):
    return x1 - x2, y1 - y2, z1 - z2


@numba.njit
def gvector_mul(x, y, z, scalar):
    return x * scalar, y * scalar, z * scalar


@numba.njit
def gvector_linear_combination(x1, y1, z1, x2, y2, z2, l1, l2=None):
    if l2 is None:
        l2 = 1 - l1
    return x1 * l1 + x2 * l2, y1 * l1 + y2 * l2, z1 * l1 + z2 * l2


@numba.njit
def spline_get_domain(knots, degree):
    """Returns the domain of the B-Spline"""
    return knots[degree - 1], knots[len(knots) - degree]


@numba.njit
def spline_get_index(u, knots, degree):
    for ii in range(degree - 1, len(knots) - degree):
        if knots[ii] <= u < knots[ii + 1]:
            return ii
    return len(knots) - degree - 1


@numba.njit
def spline_call(u, points_x, points_y, points_z, knots, degree):
    """Calculates a point of the B-Spline using de Boors Algorithm"""
    dom_start, dom_end = spline_get_domain(knots, degree)

    if u < dom_start or u > dom_end:
        return 0.0, 0.0, 0.0  # Error case, caller should handle this

    if u == dom_start:
        return points_x[0], points_y[0], points_z[0]

    if u == dom_end:
        return points_x[-1], points_y[-1], points_z[-1]

    I = spline_get_index(u, knots, degree)

    # Create d arrays for the de Boor algorithm
    d_x = np.zeros(degree + 1, dtype=np.float64)
    d_y = np.zeros(degree + 1, dtype=np.float64)
    d_z = np.zeros(degree + 1, dtype=np.float64)

    # Initialize d with the relevant control points
    for ii in range(degree + 1):
        idx = I - degree + 1 + ii
        d_x[ii] = points_x[idx]
        d_y[ii] = points_y[idx]
        d_z[ii] = points_z[idx]

    # Apply de Boor algorithm
    for ik in range(1, degree + 1):
        for ii in range(I - degree + ik + 1, I + 2):
            ua = knots[ii + degree - ik]
            ub = knots[ii - 1]
            co1 = (ua - u) / (ua - ub)
            co2 = (u - ub) / (ua - ub)
            index = ii - I + degree - ik - 1

            # Linear combination
            d_x[index] = d_x[index] * co1 + d_x[index + 1] * co2
            d_y[index] = d_y[index] * co1 + d_y[index + 1] * co2
            d_z[index] = d_z[index] * co1 + d_z[index + 1] * co2

    return d_x[0], d_y[0], d_z[0]


@numba.njit
def spline_length(points_x, points_y, points_z, knots, degree):
    """Calculate approximate length of spline"""
    length = 0.0
    curr_x, curr_y, curr_z = spline_call(0, points_x, points_y, points_z, knots, degree)

    for i in range(1, 1000):
        last_x, last_y, last_z = curr_x, curr_y, curr_z
        t = 1 / 999 * i
        curr_x, curr_y, curr_z = spline_call(
            t, points_x, points_y, points_z, knots, degree
        )
        length += gvector_dist(curr_x, curr_y, curr_z, last_x, last_y, last_z)

    return length


@numba.njit
def chaos_init(
    splines_points_x,
    splines_points_y,
    splines_points_z,
    splines_knots,
    splines_degrees,
    thickness,
):
    """Initialize chaos game parameters"""
    # Calculate bounding box
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf

    for i in range(len(splines_points_x)):
        for j in range(len(splines_points_x[i])):
            minx = min(minx, splines_points_x[i][j])
            miny = min(miny, splines_points_y[i][j])
            maxx = max(maxx, splines_points_x[i][j])
            maxy = max(maxy, splines_points_y[i][j])

    height = maxy - miny
    width = maxx - minx

    # Calculate transformations
    num_trafos = np.zeros(len(splines_points_x), dtype=np.int32)
    maxlength = thickness * width / height

    for i in range(len(splines_points_x)):
        length = spline_length(
            splines_points_x[i],
            splines_points_y[i],
            splines_points_z[i],
            splines_knots[i],
            splines_degrees[i],
        )
        num_trafos[i] = max(1, int(length / maxlength * 1.5))

    num_total = np.sum(num_trafos)

    return minx, miny, maxx, maxy, width, height, num_trafos, num_total


@numba.njit
def get_random_trafo(num_trafos, num_total):
    """Get a random transformation"""
    r = random.randrange(int(num_total) + 1)
    l = 0

    for i in range(len(num_trafos)):
        if l <= r < l + num_trafos[i]:
            return i, random.randrange(num_trafos[i])
        l += num_trafos[i]

    return len(num_trafos) - 1, random.randrange(num_trafos[-1])


@numba.njit
def truncate_point(x, y, minx, miny, maxx, maxy):
    """Truncate point to stay within bounds"""
    if x >= maxx:
        x = maxx
    if y >= maxy:
        y = maxy
    if x < minx:
        x = minx
    if y < miny:
        y = miny
    return x, y


@numba.njit
def transform_point(
    point_x,
    point_y,
    point_z,
    splines_points_x,
    splines_points_y,
    splines_points_z,
    splines_knots,
    splines_degrees,
    minx,
    miny,
    maxx,
    maxy,
    width,
    height,
    num_trafos,
    num_total,
    thickness,
    trafo_idx=None,
    trafo_val=None,
):
    """Transform a point using the chaos game rules"""
    x = (point_x - minx) / width
    y = (point_y - miny) / height

    if trafo_idx is None or trafo_val is None:
        trafo_idx, trafo_val = get_random_trafo(num_trafos, num_total)

    # Get domain and calculate t
    start, end = spline_get_domain(splines_knots[trafo_idx], splines_degrees[trafo_idx])
    length = end - start
    seg_length = length / num_trafos[trafo_idx]
    t = start + seg_length * trafo_val + seg_length * x

    # Get base point
    basepoint_x, basepoint_y, basepoint_z = spline_call(
        t,
        splines_points_x[trafo_idx],
        splines_points_y[trafo_idx],
        splines_points_z[trafo_idx],
        splines_knots[trafo_idx],
        splines_degrees[trafo_idx],
    )

    # Get derivative
    if t + 1 / 50000 > end:
        neighbour_x, neighbour_y, neighbour_z = spline_call(
            t - 1 / 50000,
            splines_points_x[trafo_idx],
            splines_points_y[trafo_idx],
            splines_points_z[trafo_idx],
            splines_knots[trafo_idx],
            splines_degrees[trafo_idx],
        )
        derivative_x = neighbour_x - basepoint_x
        derivative_y = neighbour_y - basepoint_y
        derivative_z = neighbour_z - basepoint_z
    else:
        neighbour_x, neighbour_y, neighbour_z = spline_call(
            t + 1 / 50000,
            splines_points_x[trafo_idx],
            splines_points_y[trafo_idx],
            splines_points_z[trafo_idx],
            splines_knots[trafo_idx],
            splines_degrees[trafo_idx],
        )
        derivative_x = basepoint_x - neighbour_x
        derivative_y = basepoint_y - neighbour_y
        derivative_z = basepoint_z - neighbour_z

    # Calculate new point
    mag = gvector_mag(derivative_x, derivative_y, derivative_z)
    if mag != 0:
        basepoint_x += derivative_y / mag * (y - 0.5) * thickness
        basepoint_y += -derivative_x / mag * (y - 0.5) * thickness

    # Truncate
    basepoint_x, basepoint_y = truncate_point(
        basepoint_x, basepoint_y, minx, miny, maxx, maxy
    )

    return basepoint_x, basepoint_y, basepoint_z


@numba.njit
def create_image_chaos(
    w,
    h,
    iterations,
    splines_points_x,
    splines_points_y,
    splines_points_z,
    splines_knots,
    splines_degrees,
    minx,
    miny,
    maxx,
    maxy,
    width,
    height,
    num_trafos,
    num_total,
    thickness,
    rng_seed=DEFAULT_RNG_SEED,
):
    """Create a chaos game fractal image"""
    # Set random seed for reproducibility
    random.seed(rng_seed)

    # Create image array (initialized to 1)
    im = np.ones((w, h), dtype=np.uint8)

    # Start at middle point
    point_x = (maxx + minx) / 2
    point_y = (maxy + miny) / 2
    point_z = 0

    # Run iterations
    for _ in range(iterations):
        point_x, point_y, point_z = transform_point(
            point_x,
            point_y,
            point_z,
            splines_points_x,
            splines_points_y,
            splines_points_z,
            splines_knots,
            splines_degrees,
            minx,
            miny,
            maxx,
            maxy,
            width,
            height,
            num_trafos,
            num_total,
            thickness,
        )

        # Map point to image coordinates
        x = int((point_x - minx) / width * w)
        y = int((point_y - miny) / height * h)

        # Boundary check
        if x == w:
            x -= 1
        if y == h:
            y -= 1

        # Set pixel
        im[x, h - y - 1] = 0

    return im


def write_ppm(im, filename):
    """Write image to PPM file"""
    w, h = im.shape

    with open(filename, "w", encoding="latin1", newline="") as fp:
        fp.write("P6\n")
        fp.write("%i %i\n255\n" % (w, h))

        for j in range(h):
            for i in range(w):
                val = im[i, j]
                c = val * 255
                fp.write("%c%c%c" % (c, c, c))


def get_default_splines():
    """Return the default set of splines for the chaos game in Numba-compatible format"""
    # Define points for spline 1
    points_x = [
        np.array(
            [1.597350, 1.575810, 1.313210, 1.618900, 2.889940, 2.373060, 1.662000],
            dtype=np.float64,
        ),
        np.array([2.804500, 2.550500, 1.979010, 1.979010], dtype=np.float64),
        np.array([2.001670, 2.335040, 2.366800, 2.366800], dtype=np.float64),
    ]

    points_y = [
        np.array(
            [3.304460, 4.123260, 5.288350, 5.329910, 5.502700, 4.381830, 4.360280],
            dtype=np.float64,
        ),
        np.array([4.017350, 3.525230, 2.620360, 2.620360], dtype=np.float64),
        np.array([4.011320, 3.312830, 3.233460, 3.233460], dtype=np.float64),
    ]

    points_z = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ]

    knots = [
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.float64),
        np.array([0, 0, 0, 1, 1, 1], dtype=np.float64),
        np.array([0, 0, 0, 1, 1, 1], dtype=np.float64),
    ]

    degrees = np.array([3, 3, 3], dtype=np.int32)

    return points_x, points_y, points_z, knots, degrees


def bench_chaos(
    loops,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    iterations=DEFAULT_ITERATIONS,
    thickness=DEFAULT_THICKNESS,
    rng_seed=DEFAULT_RNG_SEED,
    filename=None,
):
    """Benchmark for chaos fractal generation using Numba"""
    # Get splines directly in Numba-compatible format
    (
        splines_points_x,
        splines_points_y,
        splines_points_z,
        splines_knots,
        splines_degrees,
    ) = get_default_splines()

    # Initialize chaos parameters
    minx, miny, maxx, maxy, width_chaos, height_chaos, num_trafos, num_total = (
        chaos_init(
            splines_points_x,
            splines_points_y,
            splines_points_z,
            splines_knots,
            splines_degrees,
            thickness,
        )
    )

    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        im = create_image_chaos(
            width,
            height,
            iterations,
            splines_points_x,
            splines_points_y,
            splines_points_z,
            splines_knots,
            splines_degrees,
            minx,
            miny,
            maxx,
            maxy,
            width_chaos,
            height_chaos,
            num_trafos,
            num_total,
            thickness,
            rng_seed,
        )

        if filename:
            write_ppm(im, filename)

    return pyperf.perf_counter() - t0


# Benchmark definitions
BENCHMARKS = {
    "chaos": (
        bench_chaos,
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
        DEFAULT_ITERATIONS,
        DEFAULT_THICKNESS,
        DEFAULT_RNG_SEED,
        "fractal.ppm",
    ),
}


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(quiet=False)
    for bench in sorted(BENCHMARKS):
        name = "chaos_numba_%s" % bench
        args = BENCHMARKS[bench]
        runner.bench_time_func(name, *args)
