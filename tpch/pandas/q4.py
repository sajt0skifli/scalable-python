import pyperf

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 4


def get_ds():
    lineitem = get_line_item_ds()
    orders = get_orders_ds()

    return lineitem, orders


def query():
    lineitem, orders = get_ds()
    orders = orders.drop(columns=["o_comment"])

    var1 = date(1993, 7, 1)
    var2 = date(1993, 10, 1)

    q_final = (
        orders.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .pipe(lambda df: df[(df["o_orderdate"] < var2) & (df["o_orderdate"] >= var1)])
        .pipe(lambda df: df[df["l_commitdate"] < df["l_receiptdate"]])
        .drop_duplicates(["o_orderpriority", "o_orderkey"])
        .groupby("o_orderpriority", as_index=False)["o_orderkey"]
        .count()
        .sort_values(["o_orderpriority"])
        .rename(columns={"o_orderkey": "order_count"})
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
    runner.bench_func("pandas-q4", bench_q4)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
