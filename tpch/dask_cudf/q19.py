import pyperf
import dask_cudf
import pandas as pd

from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 19


def get_ds():
    lineitem = get_line_item_ds(mode="cudask")
    part = get_part_ds(mode="cudask")

    return lineitem, part


def query():
    lineitem, part = get_ds()

    # Early column projection
    part_cols = part[["p_partkey", "p_brand", "p_container", "p_size"]]
    lineitem_cols = lineitem[
        [
            "l_partkey",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_shipmode",
            "l_shipinstruct",
        ]
    ]

    # Merge with reduced columns
    merged_df = part_cols.merge(
        lineitem_cols, left_on="p_partkey", right_on="l_partkey"
    )

    # Apply initial filters to reduce data volume early
    filtered = merged_df[
        merged_df["l_shipmode"].isin(["AIR", "AIR REG"])
        & (merged_df["l_shipinstruct"] == "DELIVER IN PERSON")
    ]

    # Create masks for the three conditions
    mask1 = (
        (filtered["p_brand"] == "Brand#12")
        & filtered["p_container"].isin(["SM CASE", "SM BOX", "SM PACK", "SM PKG"])
        & (filtered["l_quantity"] >= 1)
        & (filtered["l_quantity"] <= 11)
        & (filtered["p_size"] >= 1)
        & (filtered["p_size"] <= 5)
    )

    mask2 = (
        (filtered["p_brand"] == "Brand#23")
        & filtered["p_container"].isin(["MED BAG", "MED BOX", "MED PKG", "MED PACK"])
        & (filtered["l_quantity"] >= 10)
        & (filtered["l_quantity"] <= 20)
        & (filtered["p_size"] >= 1)
        & (filtered["p_size"] <= 10)
    )

    mask3 = (
        (filtered["p_brand"] == "Brand#34")
        & filtered["p_container"].isin(["LG CASE", "LG BOX", "LG PACK", "LG PKG"])
        & (filtered["l_quantity"] >= 20)
        & (filtered["l_quantity"] <= 30)
        & (filtered["p_size"] >= 1)
        & (filtered["p_size"] <= 15)
    )

    # Select rows that match any condition and calculate revenue
    combined_mask = mask1 | mask2 | mask3
    revenue_calc = filtered[combined_mask]["l_extendedprice"] * (
        1 - filtered[combined_mask]["l_discount"]
    )

    # Sum only at the end (triggers computation)
    total_revenue = revenue_calc.sum().compute()

    # Create final DataFrame
    q_final = pd.DataFrame({"revenue": [total_revenue]})

    return q_final


def bench_q19():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q19", bench_q19)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=False)
