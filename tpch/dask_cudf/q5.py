import pyperf
import dask_cudf
import numpy as np
import tpch.utils as utils

from datetime import date
from dask.distributed import Client

Q_NUM = 5


def get_ds():
    customer = utils.get_customer_ds("cudask")
    orders = utils.get_orders_ds("cudask")
    lineitem = utils.get_line_item_ds("cudask")
    supplier = utils.get_supplier_ds("cudask")
    nation = utils.get_nation_ds("cudask")
    region = utils.get_region_ds("cudask")

    return customer, orders, lineitem, supplier, nation, region


def query():
    customer, orders, lineitem, supplier, nation, region = get_ds()

    var1 = "ASIA"
    var2 = np.datetime64(date(1994, 1, 1))
    var3 = np.datetime64(date(1995, 1, 1))

    region_filtered = region[region["r_name"] == var1]

    orders_filtered = orders[
        (orders["o_orderdate"] >= var2) & (orders["o_orderdate"] < var3)
    ]

    result = (
        region_filtered.merge(nation, left_on="r_regionkey", right_on="n_regionkey")
        .merge(customer, left_on="n_nationkey", right_on="c_nationkey")
        .merge(orders_filtered, left_on="c_custkey", right_on="o_custkey")
        .merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .merge(
            supplier,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
    )

    q_final = (
        result.assign(revenue=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]))
        .groupby("n_name")
        .agg({"revenue": "sum"})
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    return q_final.compute()


def bench_q5():
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
    runner.bench_func("dask_cudf-q5", bench_q5)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # utils.export_df(result, file_name, is_cudf=True)
