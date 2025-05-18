import pyperf
import dask_cudf
import pandas as pd
import numpy as np

from datetime import date
from dask.distributed import Client
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

    filtered_lineitem = lineitem[
        (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
    ]

    filtered_df = filtered_lineitem.merge(
        part, left_on="l_partkey", right_on="p_partkey"
    )
    filtered_df = filtered_df.assign(
        discounted_price=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]),
    )

    result_df = filtered_df.compute()

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
    client = Client()
    print(client.scheduler_info)
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q14", bench_q14)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=False)
