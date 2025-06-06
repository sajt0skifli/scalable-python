import pyperf
import cudf
import numpy as np

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 12


def get_ds():
    lineitem = get_line_item_ds("cudf")
    orders = get_orders_ds("cudf")

    return lineitem, orders


def query():
    lineitem, orders = get_ds()

    var1 = "MAIL"
    var2 = "SHIP"
    var3 = np.datetime64(date(1994, 1, 1))
    var4 = np.datetime64(date(1995, 1, 1))
    high_priorities = ["1-URGENT", "2-HIGH"]

    filtered_lineitem = lineitem[
        (lineitem["l_shipmode"].isin([var1, var2]))
        & (lineitem["l_commitdate"] < lineitem["l_receiptdate"])
        & (lineitem["l_shipdate"] < lineitem["l_commitdate"])
        & (lineitem["l_receiptdate"] >= var3)
        & (lineitem["l_receiptdate"] < var4)
    ][["l_orderkey", "l_shipmode"]]

    slim_orders = orders[["o_orderkey", "o_orderpriority"]]

    slim_orders["is_high_priority"] = slim_orders["o_orderpriority"].isin(
        high_priorities
    )

    q_final = (
        filtered_lineitem.merge(
            slim_orders, left_on="l_orderkey", right_on="o_orderkey"
        )
        .assign(
            high_line_count=lambda df: df["is_high_priority"],
            low_line_count=lambda df: ~df["is_high_priority"],
        )
        .groupby("l_shipmode", as_index=False)
        .agg({"high_line_count": "sum", "low_line_count": "sum"})
        .sort_values("l_shipmode")
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
    runner.bench_func("cudf-q12", bench_q12)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
