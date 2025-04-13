import pyperf
import numpy as np
import cudf

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

    # Convert dates to numpy.datetime64 for cuDF compatibility
    var1 = np.datetime64(date(1993, 7, 1))
    var2 = np.datetime64(date(1993, 10, 1))

    # Select only necessary columns right after loading to reduce memory footprint
    orders = orders[["o_orderkey", "o_orderpriority", "o_orderdate"]]
    lineitem = lineitem[["l_orderkey", "l_commitdate", "l_receiptdate"]]

    # Pre-filter both dataframes before merge to reduce join size
    filtered_orders = orders[
        (orders["o_orderdate"] >= var1) & (orders["o_orderdate"] < var2)
    ]
    filtered_lineitem = lineitem[lineitem["l_commitdate"] < lineitem["l_receiptdate"]]

    # Join pre-filtered dataframes with SortMergeJoin hint for large tables
    merged = filtered_orders.merge(
        filtered_lineitem, left_on="o_orderkey", right_on="l_orderkey", how="inner"
    )

    # Use drop_duplicates before groupby to reduce aggregation work
    deduped = merged.drop_duplicates(["o_orderpriority", "o_orderkey"])

    # Use more efficient size() method for counting
    q_final = (
        deduped.groupby("o_orderpriority")
        .size()
        .reset_index()
        .rename(columns={0: "order_count"})
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
