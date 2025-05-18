import pyperf
import dask_cudf
import numpy as np
import tpch.utils as utils

from datetime import date
from dask.distributed import Client

Q_NUM = 8


def get_ds():
    customer = utils.get_customer_ds("cudask")
    orders = utils.get_orders_ds("cudask")
    lineitem = utils.get_line_item_ds("cudask")
    part = utils.get_part_ds("cudask")
    supplier = utils.get_supplier_ds("cudask")
    nation = utils.get_nation_ds("cudask")
    region = utils.get_region_ds("cudask")

    return customer, orders, lineitem, part, supplier, nation, region


def query():
    customer, orders, lineitem, part, supplier, nation, region = get_ds()

    var1 = "BRAZIL"
    var2 = "AMERICA"
    var3 = "ECONOMY ANODIZED STEEL"
    var4 = np.datetime64(date(1995, 1, 1))
    var5 = np.datetime64(date(1996, 12, 31))

    filtered_part = part[part["p_type"] == var3][["p_partkey"]]
    filtered_region = region[region["r_name"] == var2][["r_regionkey"]]
    filtered_orders = orders[
        (orders["o_orderdate"] >= var4) & (orders["o_orderdate"] <= var5)
    ][["o_orderkey", "o_custkey", "o_orderdate"]]

    slim_lineitem = lineitem[
        ["l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]
    slim_supplier = supplier[["s_suppkey", "s_nationkey"]]
    slim_customer = customer[["c_custkey", "c_nationkey"]]

    result = (
        filtered_part.merge(slim_lineitem, left_on="p_partkey", right_on="l_partkey")
        .merge(slim_supplier, left_on="l_suppkey", right_on="s_suppkey")
        .merge(filtered_orders, left_on="l_orderkey", right_on="o_orderkey")
        .merge(slim_customer, left_on="o_custkey", right_on="c_custkey")
        .merge(
            nation[["n_nationkey", "n_regionkey"]],
            left_on="c_nationkey",
            right_on="n_nationkey",
        )
        .merge(filtered_region, left_on="n_regionkey", right_on="r_regionkey")
        .merge(
            nation[["n_nationkey", "n_name"]],
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
    )

    result = result.assign(
        volume=result["l_extendedprice"] * (1 - result["l_discount"]),
        o_year=result["o_orderdate"].dt.year,
    )

    result = result.assign(
        brazil_volume=lambda df: df["volume"] * (df["n_name"] == var1)
    )

    q_final = (
        result.groupby("o_year")
        .agg({"brazil_volume": "sum", "volume": "sum"})
        .reset_index()
        .assign(mkt_share=lambda df: df["brazil_volume"] / df["volume"])[
            ["o_year", "mkt_share"]
        ]
        .sort_values("o_year")
    )

    q_final = q_final.compute()
    q_final["mkt_share"] = q_final["mkt_share"].round(2)

    return q_final


def bench_q8():
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
    runner.bench_func("dask_cudf-q8", bench_q8)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # utils.export_df(result, file_name, is_cudf=True)
