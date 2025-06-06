import pyperf
import pandas as pd

from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 19


def get_ds():
    lineitem = get_line_item_ds("dask")
    part = get_part_ds("dask")

    return lineitem, part


def query() -> pd.DataFrame:
    lineitem, part = get_ds()

    merged = part.merge(lineitem, left_on="p_partkey", right_on="l_partkey")

    base_cond = (merged["l_shipmode"].isin(["AIR", "AIR REG"])) & (
        merged["l_shipinstruct"] == "DELIVER IN PERSON"
    )

    # Define the three specific condition sets
    cond1 = (
        (merged["p_brand"] == "Brand#12")
        & (merged["p_container"].isin(["SM CASE", "SM BOX", "SM PACK", "SM PKG"]))
        & (merged["l_quantity"] >= 1)
        & (merged["l_quantity"] <= 11)
        & (merged["p_size"] >= 1)
        & (merged["p_size"] <= 5)
    )

    cond2 = (
        (merged["p_brand"] == "Brand#23")
        & (merged["p_container"].isin(["MED BAG", "MED BOX", "MED PKG", "MED PACK"]))
        & (merged["l_quantity"] >= 10)
        & (merged["l_quantity"] <= 20)
        & (merged["p_size"] >= 1)
        & (merged["p_size"] <= 10)
    )

    cond3 = (
        (merged["p_brand"] == "Brand#34")
        & (merged["p_container"].isin(["LG CASE", "LG BOX", "LG PACK", "LG PKG"]))
        & (merged["l_quantity"] >= 20)
        & (merged["l_quantity"] <= 30)
        & (merged["p_size"] >= 1)
        & (merged["p_size"] <= 15)
    )

    filtered = merged[base_cond & (cond1 | cond2 | cond3)]

    # Calculate the revenue
    revenue = filtered["l_extendedprice"] * (1 - filtered["l_discount"])
    total_revenue = revenue.sum().compute()

    return pd.DataFrame({"revenue": [total_revenue]})


def bench_q19():
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
    runner.bench_func("dask-q19", bench_q19)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
