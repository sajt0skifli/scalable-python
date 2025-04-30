import pyperf
import pandas as pd

from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    get_nation_ds,
    get_supplier_ds,
    export_df,
)

Q_NUM = 21


def get_ds():
    lineitem = get_line_item_ds("dask")
    orders = get_orders_ds("dask")
    nation = get_nation_ds("dask")
    supplier = get_supplier_ds("dask")

    return lineitem, orders, nation, supplier


def query() -> pd.DataFrame:
    lineitem, orders, nation, supplier = get_ds()

    var1 = "SAUDI ARABIA"

    # Filter small datasets first
    filtered_nation = nation[nation["n_name"] == var1]
    filtered_orders = orders[orders["o_orderstatus"] == "F"]

    # Filter line items with late delivery
    late_delivery = lineitem[lineitem["l_receiptdate"] > lineitem["l_commitdate"]]

    # Find orders with multiple suppliers - compute to get a stable intermediate result
    multi_supp_orders = (
        lineitem.groupby("l_orderkey")
        .l_suppkey.nunique()
        .to_frame(name="n_supp_by_order")
        .reset_index()
        .persist()  # Mark for caching without computing immediately
    )

    # Filter for orders with more than one supplier
    multi_supp_filtered = multi_supp_orders[multi_supp_orders["n_supp_by_order"] > 1]

    # Join with late delivery items - this step is key for the query
    q1 = multi_supp_filtered.merge(late_delivery, on="l_orderkey").persist()

    q1_agg = (
        q1.groupby("l_orderkey")
        .l_suppkey.nunique()
        .to_frame(name="n_supp_by_order_left")
        .reset_index()
    )

    result = (
        q1_agg.merge(q1, on="l_orderkey")
        .merge(supplier.compute(), left_on="l_suppkey", right_on="s_suppkey")
        .merge(filtered_nation.compute(), left_on="s_nationkey", right_on="n_nationkey")
        .merge(filtered_orders.compute(), left_on="l_orderkey", right_on="o_orderkey")
    )

    # Apply the filter, group, and sort
    result = (
        result[result["n_supp_by_order_left"] == 1]
        .groupby("s_name")
        .agg(numwait=("l_suppkey", "count"))
        .reset_index()
        .sort_values(["numwait", "s_name"], ascending=[False, True])
        .head(100)
    )

    return result


def bench_q21():
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
    runner.bench_func("dask-q21", bench_q21)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
