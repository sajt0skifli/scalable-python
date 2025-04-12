import pyperf

import pandas as pd
from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 17


def get_ds():
    lineitem = get_line_item_ds("dask")
    part = get_part_ds("dask")

    return lineitem, part


def query() -> pd.DataFrame:
    lineitem, part = get_ds()

    var1 = "Brand#23"
    var2 = "MED BOX"

    # Filter parts first (typically smaller dataset)
    filtered_parts = part[(part["p_brand"] == var1) & (part["p_container"] == var2)]

    # Merge with lineitem
    merged_data = filtered_parts.merge(
        lineitem, how="left", left_on="p_partkey", right_on="l_partkey"
    )

    # Calculate the average quantity per part
    avg_quantity = (
        merged_data.groupby("p_partkey")
        .agg(avg_quantity=("l_quantity", "mean"))
        .reset_index()
        .assign(avg_quantity=lambda df: df["avg_quantity"] * 0.2)
    )

    # Merge back and filter rows where quantity is less than 0.2 * avg_quantity
    filtered_data = merged_data.merge(avg_quantity, on="p_partkey")

    # We need to compute here because we're doing a complex operation
    # that's not well-supported by Dask's query system
    computed_filtered = filtered_data.compute()
    result_filtered = computed_filtered[
        computed_filtered["l_quantity"] < computed_filtered["avg_quantity"]
    ]

    # Calculate the final average yearly value
    avg_yearly = result_filtered["l_extendedprice"].sum() / 7.0

    # Create a dataframe with the result
    return pd.DataFrame({"avg_yearly": [avg_yearly]})


def bench_q17():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask-q17", bench_q17)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
