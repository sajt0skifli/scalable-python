import pyperf
import dask_cudf

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
    lineitem = get_line_item_ds(mode="cudask")
    orders = get_orders_ds(mode="cudask")
    nation = get_nation_ds(mode="cudask")
    supplier = get_supplier_ds(mode="cudask")

    return lineitem, orders, nation, supplier


def query():
    lineitem, orders, nation, supplier = get_ds()

    var1 = "SAUDI ARABIA"

    # Project columns early and filter late deliveries in one chain
    late_lineitem = lineitem[
        ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
    ].loc[lambda df: df["l_receiptdate"] > df["l_commitdate"]][
        ["l_orderkey", "l_suppkey"]
    ]

    # Prefilter nation and orders tables
    nation_filtered = nation[nation["n_name"] == var1][["n_nationkey"]]
    orders_filtered = orders[orders["o_orderstatus"] == "F"][["o_orderkey"]]
    supplier_minimal = supplier[["s_suppkey", "s_name", "s_nationkey"]]

    # Count distinct suppliers per order and find multi-supplier orders
    all_suppliers_by_order = (
        lineitem[["l_orderkey", "l_suppkey"]]
        .groupby("l_orderkey")
        .l_suppkey.nunique()
        .to_frame(name="n_supp_by_order")
        .reset_index()
        .loc[lambda df: df["n_supp_by_order"] > 1]
    )

    # Chain operations for the final result
    q_final = (
        all_suppliers_by_order.merge(late_lineitem, on="l_orderkey")
        .groupby("l_orderkey")
        .l_suppkey.nunique()
        .to_frame(name="n_supp_by_order_left")
        .reset_index()
        .loc[lambda df: df["n_supp_by_order_left"] == 1]
        .merge(late_lineitem, on="l_orderkey")
        .merge(supplier_minimal, left_on="l_suppkey", right_on="s_suppkey")
        .merge(nation_filtered, left_on="s_nationkey", right_on="n_nationkey")
        .merge(orders_filtered, left_on="l_orderkey", right_on="o_orderkey")
        .groupby("s_name")
        .agg({"l_suppkey": "count"})
        .reset_index()
        .rename(columns={"l_suppkey": "numwait"})
        .sort_values(["numwait", "s_name"], ascending=[False, True])
        .head(100)[["s_name", "numwait"]]
    )

    return q_final


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
    runner.bench_func("dask_cudf-q21", bench_q21)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
