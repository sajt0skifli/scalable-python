import importlib
import argparse


BENCHMARK_SUITES = [
    "scimark",
]


def main():
    parser = argparse.ArgumentParser(description="Run various benchmark suites")
    parser.add_argument(
        "suite",
        nargs="?",
        choices=BENCHMARK_SUITES + ["all"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=20,
        help="Number of times to repeat each benchmark (default: 20)",
    )

    args = parser.parse_args()

    if args.suite == "all":
        suites_to_run = BENCHMARK_SUITES
    else:
        suites_to_run = [args.suite]

    for module_name in suites_to_run:
        print(f"\n=== Running {module_name} benchmark suite ===\n")

        # Dynamically import the benchmark module
        module = importlib.import_module(f"pyperformance.{module_name}")

        # Run all benchmarks from the module
        from pyperformance.utils import run_benchmark

        for bench in sorted(module.BENCHMARKS):
            run_benchmark(bench, module.BENCHMARKS, args.repetitions)


if __name__ == "__main__":
    main()
