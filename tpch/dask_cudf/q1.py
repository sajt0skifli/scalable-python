import pyperf
import dask_cudf
import numpy as np

from datetime import date
from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    export_df,
)

Q_NUM = 1


def get_ds():
    lineitem = get_line_item_ds("cudask")
    return lineitem


def query():
    lineitem = get_ds()

    # Convert target date to string format that dask_cudf can compare with
    target_date = np.datetime64(date(1998, 9, 2))

    # Filter and create new columns
    filtered = lineitem[lineitem["l_shipdate"] <= target_date]
    filtered = filtered.assign(
        disc_price=lambda df: df["l_extendedprice"] * (1 - df["l_discount"])
    )
    filtered = filtered.assign(charge=lambda df: df["disc_price"] * (1 + df["l_tax"]))

    # Group by and aggregate - using a dictionary for aggregation
    agg_dict = {
        "l_quantity": ["sum", "mean"],
        "l_extendedprice": ["sum", "mean"],
        "disc_price": ["sum"],
        "charge": ["sum"],
        "l_discount": ["mean"],
        "l_returnflag": ["count"],
    }

    result = filtered.groupby(["l_returnflag", "l_linestatus"]).agg(agg_dict)

    # Rename columns to match expected output
    result.columns = [
        "sum_qty",
        "avg_qty",
        "sum_base_price",
        "avg_price",
        "sum_disc_price",
        "sum_charge",
        "avg_disc",
        "count_order",
    ]

    # Reset index to get groupby columns back as regular columns
    result = result.reset_index()

    # Sort the results - compute is needed to materialize the results
    q_final = result.reset_index().sort_values(["l_returnflag", "l_linestatus"])

    return q_final.compute()


def bench_q1():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    client = Client()
    print(client.scheduler_info)
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q1", bench_q1)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
