import pyperf
import dask_cudf
import numpy as np

from datetime import date
from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    get_line_item_ds,
    get_nation_ds,
    export_df,
)

Q_NUM = 10


def get_ds():
    customer = get_customer_ds("cudask")
    orders = get_orders_ds("cudask")
    lineitem = get_line_item_ds("cudask")
    nation = get_nation_ds("cudask")

    return customer, orders, lineitem, nation


def query():
    customer, orders, lineitem, nation = get_ds()

    var1 = np.datetime64(date(1993, 10, 1))
    var2 = np.datetime64(date(1994, 1, 1))

    # Apply early filters to reduce data size before joins
    filtered_orders = orders[
        (orders["o_orderdate"] >= var1) & (orders["o_orderdate"] < var2)
    ]
    filtered_lineitem = lineitem[lineitem["l_returnflag"] == "R"]

    # Select only needed columns
    slim_customer = customer[
        [
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "c_nationkey",
            "c_address",
            "c_comment",
        ]
    ]
    slim_orders = filtered_orders[["o_orderkey", "o_custkey"]]
    slim_lineitem = filtered_lineitem[["l_orderkey", "l_extendedprice", "l_discount"]]
    slim_nation = nation[["n_nationkey", "n_name"]]

    # Join chain with filtered data
    result = (
        slim_customer.merge(slim_orders, left_on="c_custkey", right_on="o_custkey")
        .merge(slim_lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .merge(slim_nation, left_on="c_nationkey", right_on="n_nationkey")
        .assign(revenue=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]))
        .groupby(
            [
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            ]
        )
        .agg({"revenue": "sum"})
        .reset_index()
        .sort_values(by="revenue", ascending=False)
    )

    # Compute the result and take top 20 rows
    result = result.head(20)

    # Order columns as specified in the output
    result = result[
        [
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        ]
    ]

    return result


def bench_q10():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q10", bench_q10)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
