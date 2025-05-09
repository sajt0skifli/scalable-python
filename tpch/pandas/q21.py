import pyperf

from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    get_nation_ds,
    get_supplier_ds,
    export_df,
)

Q_NUM = 21


def get_ds():
    lineitem = get_line_item_ds()
    orders = get_orders_ds()
    nation = get_nation_ds()
    supplier = get_supplier_ds()

    return lineitem, orders, nation, supplier


def query():
    lineitem, orders, nation, supplier = get_ds()

    var1 = "SAUDI ARABIA"

    q1 = (
        lineitem.groupby("l_orderkey", as_index=False)
        .agg(n_supp_by_order=("l_suppkey", "nunique"))
        .pipe(lambda df: df[df["n_supp_by_order"] > 1])
        .merge(
            lineitem[(lineitem["l_receiptdate"] > lineitem["l_commitdate"])],
            on="l_orderkey",
        )
    )

    q_final = (
        q1.groupby("l_orderkey", as_index=False)
        .agg(n_supp_by_order_left=("l_suppkey", "nunique"))
        .merge(q1, on="l_orderkey")
        .merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
        .merge(orders, left_on="l_orderkey", right_on="o_orderkey")
        .pipe(lambda df: df[df["n_supp_by_order_left"] == 1])
        .pipe(lambda df: df[df["n_name"] == var1])
        .pipe(lambda df: df[df["o_orderstatus"] == "F"])
        .groupby("s_name", as_index=False)
        .agg(numwait=("l_suppkey", "count"))
        .sort_values(["numwait", "s_name"], ascending=[False, True])
        .head(n=100)
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
    runner.bench_func("pandas-q21", bench_q21)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
