import pyperf
import pandas as pd

from datetime import date
from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    get_nation_ds,
    get_supplier_ds,
    get_part_supp_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 20


def get_ds():
    lineitem = get_line_item_ds("dask")
    nation = get_nation_ds("dask")
    supplier = get_supplier_ds("dask")
    partsupp = get_part_supp_ds("dask")
    part = get_part_ds("dask")

    return lineitem, nation, supplier, partsupp, part


def query() -> pd.DataFrame:
    lineitem, nation, supplier, partsupp, part = get_ds()

    var1 = date(1994, 1, 1)
    var2 = date(1995, 1, 1)
    var3 = "CANADA"
    var4 = "forest"

    # Filter lineitem by date range and aggregate
    date_filtered_lineitem = lineitem[
        (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
    ]

    # Compute the aggregation - this is relatively small result
    agg_lineitem = (
        date_filtered_lineitem.groupby(["l_partkey", "l_suppkey"])
        .agg(sum_quantity=("l_quantity", "sum"))
        .reset_index()
        .assign(sum_quantity=lambda df: df["sum_quantity"] * 0.5)
        .compute()
    )

    # Filter parts
    filtered_parts = part[part["p_name"].str.startswith(var4)]

    # Filter nations and join with suppliers - this is a small dataset, so computing is fine
    nation_supplier = (
        nation[nation["n_name"] == var3]
        .merge(supplier, left_on="n_nationkey", right_on="s_nationkey")
        .compute()
    )

    # Join parts and partsupp in Dask
    parts_partsupp = filtered_parts.merge(
        partsupp, left_on="p_partkey", right_on="ps_partkey"
    ).compute()

    # Now perform the remaining operations in pandas which is efficient for these steps
    result = (
        parts_partsupp.merge(
            agg_lineitem,
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )[lambda df: df["ps_availqty"] > df["sum_quantity"]]
        .drop_duplicates(subset=["ps_suppkey"])
        .merge(nation_supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .sort_values("s_name")[["s_name", "s_address"]]
    )

    return result


def bench_q20():
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
    runner.bench_func("dask-q20", bench_q20)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
