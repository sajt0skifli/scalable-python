import pyperf

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 12


def get_ds():
    lineitem = get_line_item_ds()
    orders = get_orders_ds()

    return lineitem, orders


def query():
    lineitem, orders = get_ds()

    var1 = "MAIL"
    var2 = "SHIP"
    var3 = date(1994, 1, 1)
    var4 = date(1995, 1, 1)
    high_priorities = ["1-URGENT", "2-HIGH"]

    q_final = (
        orders.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .pipe(lambda df: df[df["l_shipmode"].isin([var1, var2])])
        .pipe(lambda df: df[df["l_commitdate"] < df["l_receiptdate"]])
        .pipe(lambda df: df[df["l_shipdate"] < df["l_commitdate"]])
        .pipe(
            lambda df: df[(df["l_receiptdate"] >= var3) & (df["l_receiptdate"] < var4)]
        )
        .assign(
            high_line_count=lambda df: df["o_orderpriority"].isin(high_priorities),
            low_line_count=lambda df: ~df["o_orderpriority"].isin(high_priorities),
        )
        .groupby("l_shipmode", as_index=False, sort=True)
        .agg({"high_line_count": "sum", "low_line_count": "sum"})
        .astype({"high_line_count": int, "low_line_count": int})
    )
    return q_final


def bench_q12():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q12", bench_q12)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
