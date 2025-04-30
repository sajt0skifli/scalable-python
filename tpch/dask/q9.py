import pyperf
import tpch.utils as utils

from dask import dataframe as dd
from dask.distributed import Client

Q_NUM = 9


def get_ds():
    part = utils.get_part_ds("dask")
    supplier = utils.get_supplier_ds("dask")
    lineitem = utils.get_line_item_ds("dask")
    partsupp = utils.get_part_supp_ds("dask")
    orders = utils.get_orders_ds("dask")
    nation = utils.get_nation_ds("dask")

    return part, supplier, lineitem, partsupp, orders, nation


def query() -> dd.DataFrame:
    part, supplier, lineitem, partsupp, orders, nation = get_ds()

    # Chain of merges
    part_partsupp = part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
    with_supplier = part_partsupp.merge(
        supplier, left_on="ps_suppkey", right_on="s_suppkey"
    )
    with_lineitem = with_supplier.merge(
        lineitem,
        left_on=["p_partkey", "ps_suppkey"],
        right_on=["l_partkey", "l_suppkey"],
    )
    with_orders = with_lineitem.merge(
        orders, left_on="l_orderkey", right_on="o_orderkey"
    )
    with_nation = with_orders.merge(
        nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Filter and compute derived columns
    filtered_data = with_nation[
        with_nation["p_name"].str.contains("green", regex=False)
    ]
    filtered_data["o_year"] = filtered_data["o_orderdate"].dt.year
    filtered_data["amount"] = filtered_data["l_extendedprice"] * (
        1 - filtered_data["l_discount"]
    ) - (filtered_data["ps_supplycost"] * filtered_data["l_quantity"])
    filtered_data = filtered_data.rename(columns={"n_name": "nation"})

    # Group and aggregate
    grouped = filtered_data.groupby(["nation", "o_year"])
    agg_result = grouped.agg({"amount": "sum"}).reset_index()
    agg_result = agg_result.rename(columns={"amount": "sum_profit"})

    # Sort results
    sorted_result = agg_result.sort_values(
        by=["nation", "o_year"], ascending=[True, False]
    ).reset_index(drop=True)

    return sorted_result.compute()


def bench_q9():
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
    runner.bench_func("dask-q9", bench_q9)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
