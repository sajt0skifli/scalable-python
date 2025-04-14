import pyperf
import cudf
import numpy as np

from datetime import date
from tpch.utils import (
    get_supplier_ds,
    get_line_item_ds,
    export_df,
)

Q_NUM = 15


def get_ds():
    supplier = get_supplier_ds(mode="cudf")
    lineitem = get_line_item_ds(mode="cudf")

    return supplier, lineitem


def query():
    supplier, lineitem = get_ds()

    var1 = np.datetime64(date(1996, 1, 1))
    var2 = np.datetime64(date(1996, 4, 1))

    # Calculate revenue in one chained operation
    revenue = (
        lineitem[(lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)][
            ["l_suppkey", "l_extendedprice", "l_discount"]
        ]
        .assign(total_revenue=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]))
        .groupby("l_suppkey", as_index=False, sort=False)
        .agg({"total_revenue": "sum"})
    )

    # Get max revenue suppliers
    max_revenue = revenue["total_revenue"].max()
    max_revenue_suppliers = revenue[revenue["total_revenue"] == max_revenue]

    # Join and format final result
    q_final = (
        supplier[["s_suppkey", "s_name", "s_address", "s_phone"]]
        .merge(max_revenue_suppliers, left_on="s_suppkey", right_on="l_suppkey")
        .assign(total_revenue=lambda df: df["total_revenue"].round(2))[
            ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
        ]
        .sort_values("s_suppkey")
        .reset_index(drop=True)
    )

    return q_final


def bench_q15():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q15", bench_q15)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
