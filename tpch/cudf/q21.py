import pyperf
import cudf

from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    get_nation_ds,
    get_supplier_ds,
    export_df,
)

Q_NUM = 21


def get_ds():
    lineitem = get_line_item_ds(mode="cudf")
    orders = get_orders_ds(mode="cudf")
    nation = get_nation_ds(mode="cudf")
    supplier = get_supplier_ds(mode="cudf")

    return lineitem, orders, nation, supplier


def query():
    lineitem, orders, nation, supplier = get_ds()

    var1 = "SAUDI ARABIA"

    lineitem_minimal = lineitem[
        ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
    ]
    late_lineitem = lineitem_minimal[
        lineitem_minimal["l_receiptdate"] > lineitem_minimal["l_commitdate"]
    ]
    late_lineitem = late_lineitem[["l_orderkey", "l_suppkey"]]

    q1 = (
        lineitem_minimal[["l_orderkey", "l_suppkey"]]
        .groupby("l_orderkey", as_index=False)
        .agg({"l_suppkey": "nunique"})
        .rename(columns={"l_suppkey": "n_supp_by_order"})
        .query("n_supp_by_order > 1")
        .merge(late_lineitem, on="l_orderkey")
    )

    nation_filtered = nation[nation["n_name"] == var1][["n_nationkey"]]
    orders_filtered = orders[orders["o_orderstatus"] == "F"][["o_orderkey"]]
    supplier_minimal = supplier[["s_suppkey", "s_name", "s_nationkey"]]

    q_final = (
        q1.groupby("l_orderkey", as_index=False)
        .agg({"l_suppkey": "nunique"})
        .rename(columns={"l_suppkey": "n_supp_by_order_left"})
        .query("n_supp_by_order_left == 1")
        .merge(q1, on="l_orderkey")
        .merge(supplier_minimal, left_on="l_suppkey", right_on="s_suppkey")
        .merge(nation_filtered, left_on="s_nationkey", right_on="n_nationkey")
        .merge(orders_filtered, left_on="l_orderkey", right_on="o_orderkey")
        .groupby("s_name", as_index=False)
        .agg({"l_suppkey": "count"})
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
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q21", bench_q21)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
