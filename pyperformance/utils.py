import statistics
import timeit
from typing import Dict, Tuple


def run_benchmark(bench_name: str, bench_dict: Dict[str, Tuple], repetitions: int = 20):
    """Run a benchmark multiple times and report statistics"""
    print(f"Running benchmark: {bench_name}", end="")
    func_args = bench_dict[bench_name]
    func = func_args[0]
    args = func_args[1:]

    times = []
    for _ in range(repetitions):
        print(".", end="", flush=True)
        elapsed = timeit.timeit(lambda: func(1, *args), number=1)
        times.append(elapsed)

    # Convert seconds to milliseconds
    times_ms = [t * 1000 for t in times]
    avg_time_ms = sum(times_ms) / len(times_ms)

    print()  # New line after dots

    if len(times) > 1:
        std_dev_ms = statistics.stdev(times_ms)
        print(f"Average: {avg_time_ms:.2f} ms, stdev: {std_dev_ms:.2f} ms\n")
    else:
        print(f"Time: {avg_time_ms:.2f} ms\n")
