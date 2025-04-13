import pyperf
import numpy as np
import cudf

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    export_df,
)

Q_NUM = 1


def get_ds():
    lineitem = get_line_item_ds("cudf")
    return lineitem


def query():
    lineitem = get_ds()

    # Create target date directly as a GPU timestamp to minimize CPU->GPU transfer
    target_date = np.datetime64(date(1998, 9, 2))

    # Perform filtering and calculations in a single GPU operation chain
    filtered = lineitem[lineitem["l_shipdate"] <= target_date]

    # Pre-compute derived columns in a single operation to maximize GPU utilization
    filtered = filtered.assign(
        disc_price=filtered["l_extendedprice"] * (1 - filtered["l_discount"]),
        charge=filtered["l_extendedprice"]
        * (1 - filtered["l_discount"])
        * (1 + filtered["l_tax"]),
    )

    # Group and aggregate in a single operation
    q_final = filtered.groupby(["l_returnflag", "l_linestatus"], as_index=False).agg(
        {
            "l_quantity": ["sum", "mean"],
            "l_extendedprice": ["sum", "mean"],
            "l_discount": ["mean"],
            "disc_price": ["sum"],
            "charge": ["sum"],
            "l_returnflag": ["count"],
        }
    )

    # Flatten multi-index columns
    q_final.columns = [
        "l_returnflag",
        "l_linestatus",
        "sum_qty",
        "avg_qty",
        "sum_base_price",
        "avg_price",
        "avg_disc",
        "sum_disc_price",
        "sum_charge",
        "count_order",
    ]

    # Sort results
    q_final = q_final.sort_values(["l_returnflag", "l_linestatus"])

    return q_final


def bench_q1():
    t0 = pyperf.perf_counter()
    result = query()  # Actually materialize the result for fair benchmarking
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q1", bench_q1)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
