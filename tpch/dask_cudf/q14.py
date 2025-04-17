import pyperf
import dask_cudf
import pandas as pd
import numpy as np

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 14


def get_ds():
    lineitem = get_line_item_ds(mode="cudask")
    part = get_part_ds(mode="cudask")

    return lineitem, part


def query():
    lineitem, part = get_ds()

    var1 = np.datetime64(date(1995, 9, 1))
    var2 = np.datetime64(date(1995, 10, 1))

    # Filter lineitem first to reduce data size before merge
    filtered_lineitem = lineitem[
        (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
    ]

    # Merge filtered lineitem with part
    filtered_df = filtered_lineitem.merge(
        part, left_on="l_partkey", right_on="p_partkey"
    )

    # Calculate both values in a single pass
    filtered_df = filtered_df.assign(
        discounted_price=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]),
    )

    # Compute the result to materialize it
    result_df = filtered_df.compute()

    # Use pandas for the final calculation as it's simpler for this small result
    result_df["promo_price"] = result_df["discounted_price"].where(
        result_df["p_type"].str.startswith("PROMO"), 0
    )

    sums = result_df[["discounted_price", "promo_price"]].sum()
    promo_revenue = 100.00 * sums["promo_price"] / sums["discounted_price"]

    q_final = pd.DataFrame({"promo_revenue": [round(promo_revenue, 2)]})
    return q_final


def bench_q14():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q14", bench_q14)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=False)
