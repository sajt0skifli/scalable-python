import pyperf
import tpch.utils as utils

from datetime import date
from dask import dataframe as dd
from dask.distributed import Client

Q_NUM = 8


def get_ds():
    customer = utils.get_customer_ds("dask")
    orders = utils.get_orders_ds("dask")
    lineitem = utils.get_line_item_ds("dask")
    part = utils.get_part_ds("dask")
    supplier = utils.get_supplier_ds("dask")
    nation = utils.get_nation_ds("dask")
    region = utils.get_region_ds("dask")

    return customer, orders, lineitem, part, supplier, nation, region


def query() -> dd.DataFrame:
    customer, orders, lineitem, part, supplier, nation, region = get_ds()

    var1 = "BRAZIL"
    var2 = "AMERICA"
    var3 = "ECONOMY ANODIZED STEEL"
    var4 = date(1995, 1, 1)
    var5 = date(1996, 12, 31)

    n1 = nation[["n_nationkey", "n_regionkey"]]
    n2 = nation[["n_nationkey", "n_name"]]

    # Chain of merges and filters
    part_lineitem = part.merge(lineitem, left_on="p_partkey", right_on="l_partkey")
    with_supplier = part_lineitem.merge(
        supplier, left_on="l_suppkey", right_on="s_suppkey"
    )
    with_orders = with_supplier.merge(
        orders, left_on="l_orderkey", right_on="o_orderkey"
    )
    with_customer = with_orders.merge(
        customer, left_on="o_custkey", right_on="c_custkey"
    )
    with_nation1 = with_customer.merge(
        n1, left_on="c_nationkey", right_on="n_nationkey"
    )
    with_region = with_nation1.merge(
        region, left_on="n_regionkey", right_on="r_regionkey"
    )

    # Apply filters
    filtered_by_region = with_region[with_region["r_name"] == var2]
    with_nation2 = filtered_by_region.merge(
        n2, left_on="s_nationkey", right_on="n_nationkey"
    )
    filtered_by_date = with_nation2[
        (with_nation2["o_orderdate"] <= var5) & (with_nation2["o_orderdate"] >= var4)
    ]
    filtered_by_type = filtered_by_date[filtered_by_date["p_type"] == var3]

    # Create computed columns
    filtered_by_type["volume"] = filtered_by_type["l_extendedprice"] * (
        1 - filtered_by_type["l_discount"]
    )
    filtered_by_type["o_year"] = filtered_by_type["o_orderdate"].dt.year
    filtered_by_type["case_volume"] = filtered_by_type["volume"].where(
        filtered_by_type["n_name"] == var1, 0
    )

    # Groupby and aggregation
    gb = filtered_by_type.groupby("o_year")
    agg = gb.agg({"case_volume": "sum", "volume": "sum"}).reset_index()

    # Calculate final result and round
    agg["mkt_share"] = agg["case_volume"] / agg["volume"]
    final_result = agg[["o_year", "mkt_share"]].round(2)

    # Compute the final Dask dataframe to get a pandas dataframe
    return final_result.compute()


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
    runner.bench_func("dask-q8", bench_q8)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
