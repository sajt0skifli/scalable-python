import pyperf
import cudf
import numpy as np

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

    # Filter the datasets early to reduce merge sizes
    filtered_customer = customer[customer["c_mktsegment"] == var1]
    filtered_orders = orders[orders["o_orderdate"] < var2]
    filtered_lineitem = lineitem[lineitem["l_shipdate"] > var2]

    # Chain the merge operations
    merged_co = filtered_customer.merge(
        filtered_orders, left_on="c_custkey", right_on="o_custkey"
    )
    merged_col = merged_co.merge(
        filtered_lineitem, left_on="o_orderkey", right_on="l_orderkey"
    )

    # Calculate revenue and perform final aggregation
    merged_col = merged_col.assign(
        revenue=merged_col["l_extendedprice"] * (1 - merged_col["l_discount"])
    )

    q_final = (
        merged_col.groupby(
            ["l_orderkey", "o_orderdate", "o_shippriority"], as_index=False
        )
        .agg({"revenue": "sum"})[
            ["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]
        ]
        .sort_values(["revenue", "o_orderdate"], ascending=[False, True])
        .head(10)
    )

    return q_final


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
