import pyperf

from datetime import date
from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    get_line_item_ds,
    export_df,
)

Q_NUM = 3


def get_ds():
    customer = get_customer_ds()
    orders = get_orders_ds()
    lineitem = get_line_item_ds()

    return customer, orders, lineitem


def query():
    customer, orders, lineitem = get_ds()

    var1 = "BUILDING"
    var2 = date(1995, 3, 15)

    q_final = (
        customer[customer["c_mktsegment"] == var1]
        .merge(orders, left_on="c_custkey", right_on="o_custkey")
        .merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .pipe(lambda df: df[df["o_orderdate"] < var2])
        .pipe(lambda df: df[df["l_shipdate"] > var2])
        .assign(revenue=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]))
        .groupby(["l_orderkey", "o_orderdate", "o_shippriority"], as_index=False)
        .agg({"revenue": "sum"})[
            [
                "l_orderkey",
                "revenue",
                "o_orderdate",
                "o_shippriority",
            ]
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
    runner.bench_func("pandas-q3", bench_q3)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
