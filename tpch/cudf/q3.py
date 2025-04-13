import pyperf
import numpy as np
import cudf

from datetime import date
from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    get_line_item_ds,
    export_df,
)

Q_NUM = 3


def get_ds():
    customer = get_customer_ds("cudf")
    orders = get_orders_ds("cudf")
    lineitem = get_line_item_ds("cudf")
    return customer, orders, lineitem


def query():
    customer, orders, lineitem = get_ds()

    var1 = "BUILDING"
    var2 = np.datetime64(date(1995, 3, 15))

    # Pre-filter with more efficient cuDF operations
    filtered_customer = customer[customer["c_mktsegment"] == var1]
    filtered_orders = orders[orders["o_orderdate"] < var2]
    filtered_lineitem = lineitem[lineitem["l_shipdate"] > var2]

    # Use a more efficient multi-join approach (reduce intermediate results)
    merged = filtered_customer.merge(
        filtered_orders, left_on="c_custkey", right_on="o_custkey"
    ).merge(filtered_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Calculate revenue as part of the final operation to maximize GPU parallelism
    result = (
        merged.assign(revenue=merged["l_extendedprice"] * (1 - merged["l_discount"]))
        .groupby(["l_orderkey", "o_orderdate", "o_shippriority"])
        .agg({"revenue": "sum"})
        .reset_index()
        .sort_values(["revenue", "o_orderdate"], ascending=[False, True])
        .head(10)
    )[["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]]

    return result


def bench_q3():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q3", bench_q3)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
