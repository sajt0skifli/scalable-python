import pyperf

from datetime import date
from dask import dataframe as dd
from dask.distributed import Client
from tpch.utils import (
    get_supplier_ds,
    get_line_item_ds,
    export_df,
)

Q_NUM = 15


def get_ds():
    supplier = get_supplier_ds("dask")
    lineitem = get_line_item_ds("dask")

    return supplier, lineitem


def query() -> dd.DataFrame:
    supplier, lineitem = get_ds()

    var1 = date(1996, 1, 1)
    var2 = date(1996, 4, 1)

    # Filter lineitem by date range
    filtered_lineitem = lineitem[
        (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
    ]

    # Calculate revenue
    filtered_lineitem["total_revenue"] = filtered_lineitem["l_extendedprice"] * (
        1 - filtered_lineitem["l_discount"]
    )

    # Group by supplier key and sum revenue
    revenue = (
        filtered_lineitem.groupby("l_suppkey")
        .agg({"total_revenue": "sum"})
        .reset_index()
    )

    computed_revenue = revenue.compute()
    max_revenue = computed_revenue["total_revenue"].max()

    # Filter suppliers with maximum revenue
    max_revenue_suppliers = computed_revenue[
        computed_revenue["total_revenue"] == max_revenue
    ]

    # Merge with supplier information
    result = (
        supplier.merge(max_revenue_suppliers, left_on="s_suppkey", right_on="l_suppkey")
        .round({"total_revenue": 2})[
            ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
        ]
        .sort_values("s_suppkey")
        .reset_index(drop=True)
    )

    return result.compute()


def bench_q15():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    client = Client()
    print(client.scheduler_info)
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask-q15", bench_q15)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
