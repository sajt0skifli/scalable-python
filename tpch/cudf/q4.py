import pyperf
import cudf
import numpy as np

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 4


def get_ds():
    lineitem = get_line_item_ds("cudf")
    orders = get_orders_ds("cudf")
    return lineitem, orders


def query():
    lineitem, orders = get_ds()

    orders = orders.drop(columns=["o_comment"])

    # Convert dates to numpy.datetime64 for cuDF compatibility
    var1 = np.datetime64(date(1993, 7, 1))
    var2 = np.datetime64(date(1993, 10, 1))

    merged = orders.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")

    date_filtered = merged[
        (merged["o_orderdate"] >= var1) & (merged["o_orderdate"] < var2)
    ]
    filtered = date_filtered[
        date_filtered["l_commitdate"] < date_filtered["l_receiptdate"]
    ]

    q_final = (
        filtered.drop_duplicates(["o_orderpriority", "o_orderkey"])
        .groupby("o_orderpriority")
        .agg({"o_orderkey": "count"})
        .reset_index()
        .rename(columns={"o_orderkey": "order_count"})
        .sort_values("o_orderpriority")
    )

    return q_final


def bench_q4():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q4", bench_q4)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
