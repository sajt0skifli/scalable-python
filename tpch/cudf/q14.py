import pyperf
import cudf
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
    lineitem = get_line_item_ds(mode="cudf")
    part = get_part_ds(mode="cudf")

    return lineitem, part


def query():
    lineitem, part = get_ds()

    var1 = np.datetime64(date(1995, 9, 1))
    var2 = np.datetime64(date(1995, 10, 1))

    filtered_df = lineitem.query("l_shipdate >= @var1 and l_shipdate < @var2").merge(
        part, left_on="l_partkey", right_on="p_partkey"
    )

    # Calculate both values in a single pass using cuDF's GPU acceleration
    filtered_df["discounted_price"] = filtered_df["l_extendedprice"] * (
        1 - filtered_df["l_discount"]
    )
    filtered_df["promo_price"] = filtered_df["discounted_price"].where(
        filtered_df["p_type"].str.startswith("PROMO"), 0
    )

    sums = filtered_df[["discounted_price", "promo_price"]].sum()
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
    runner.bench_func("cudf-q14", bench_q14)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=False)
