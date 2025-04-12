import pyperf
import tpch.utils as utils

from datetime import date
from dask import dataframe as dd

Q_NUM = 5


def get_ds():
    region_ds = utils.get_region_ds("dask")
    nation_ds = utils.get_nation_ds("dask")
    customer_ds = utils.get_customer_ds("dask")
    line_item_ds = utils.get_line_item_ds("dask")
    orders_ds = utils.get_orders_ds("dask")
    supplier_ds = utils.get_supplier_ds("dask")

    return region_ds, nation_ds, customer_ds, line_item_ds, orders_ds, supplier_ds


def query() -> dd.DataFrame:
    region_ds, nation_ds, customer_ds, line_item_ds, orders_ds, supplier_ds = get_ds()

    var1 = "ASIA"
    var2 = date(1994, 1, 1)
    var3 = date(1995, 1, 1)

    jn1 = region_ds.merge(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
    jn2 = jn1.merge(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
    jn3 = jn2.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn4 = jn3.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn5 = jn4.merge(
        supplier_ds,
        left_on=["l_suppkey", "n_nationkey"],
        right_on=["s_suppkey", "s_nationkey"],
    )

    jn5 = jn5[jn5["r_name"] == var1]
    jn5 = jn5[(jn5["o_orderdate"] >= var2) & (jn5["o_orderdate"] < var3)]
    jn5["revenue"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)

    gb = jn5.groupby("n_name")["revenue"].sum().reset_index()
    result_df = gb.sort_values("revenue", ascending=False)

    return result_df.compute()  # type: ignore[no-any-return]


def bench_q5():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask-q5", bench_q5)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
