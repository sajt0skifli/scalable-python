import importlib
import pyperf


BENCHMARK_SUITES = [
    "chaos",
    "crypto",
    "nbody",
    "nqueens",
    "raytrace",
    "scimark",
]


def main():
    runner = pyperf.Runner()

    runner.argparser.add_argument(
        "suite",
        nargs="?",
        choices=BENCHMARK_SUITES + ["all"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    args = runner.parse_args()

    if args.suite == "all":
        suites_to_run = BENCHMARK_SUITES
    else:
        suites_to_run = [args.suite]

    for suite in suites_to_run:
        # Dynamically import the benchmark suite module
        module = importlib.import_module(f"{suite}")

        # Run each benchmark in the suite
        for bench in sorted(module.BENCHMARKS):
            name = f"{suite}_{bench}"
            args = module.BENCHMARKS[bench]
            runner.bench_time_func(name, *args)


if __name__ == "__main__":
    main()
