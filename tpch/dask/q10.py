import pyperf

from datetime import date
from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    get_line_item_ds,
    get_nation_ds,
    export_df,
)
from dask import dataframe as dd
from dask.distributed import Client

Q_NUM = 10


def get_ds():
    customer = get_customer_ds("dask")
    orders = get_orders_ds("dask")
    lineitem = get_line_item_ds("dask")
    nation = get_nation_ds("dask")

    return customer, orders, lineitem, nation


def query() -> dd.DataFrame:
    customer, orders, lineitem, nation = get_ds()

    var1 = date(1993, 10, 1)
    var2 = date(1994, 1, 1)

    merged_data = customer.merge(orders, left_on="c_custkey", right_on="o_custkey")
    with_lineitem = merged_data.merge(
        lineitem, left_on="o_orderkey", right_on="l_orderkey"
    )
    with_nation = with_lineitem.merge(
        nation, left_on="c_nationkey", right_on="n_nationkey"
    )

    # Apply filters
    date_filtered = with_nation[
        (with_nation["o_orderdate"] < var2) & (with_nation["o_orderdate"] >= var1)
    ]
    return_filtered = date_filtered[date_filtered["l_returnflag"] == "R"]

    # Calculate volume
    return_filtered["volume"] = return_filtered["l_extendedprice"] * (
        1 - return_filtered["l_discount"]
    )

    # Group and aggregate
    grouped = return_filtered.groupby(
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

    agg_result = grouped.agg({"volume": "sum"}).reset_index()
    agg_result = agg_result.rename(columns={"volume": "revenue"})

    # Select and order columns
    final_result = agg_result[
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

    # Sort and limit results
    sorted_result = final_result.sort_values(by="revenue", ascending=False).reset_index(
        drop=True
    )

    return sorted_result.head(20, compute=True)


def bench_q10():
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
    runner.bench_func("dask-q10", bench_q10)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
