import pyperf
import dask_cudf
import pandas as pd

from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 17


def get_ds():
    lineitem = get_line_item_ds(mode="cudask")
    part = get_part_ds(mode="cudask")

    return lineitem, part


def query():
    lineitem, part = get_ds()

    var1 = "Brand#23"
    var2 = "MED BOX"

    # Select only needed columns during filtering
    part_keys = part[(part["p_brand"] == var1) & (part["p_container"] == var2)][
        ["p_partkey"]
    ]

    # Join with lineitem (selecting only needed columns)
    joined_data = part_keys.merge(
        lineitem[["l_partkey", "l_quantity", "l_extendedprice"]],
        how="left",
        left_on="p_partkey",
        right_on="l_partkey",
    )

    # Compute to materialize the joined data for aggregation
    joined_data = joined_data.compute()

    # Calculate average quantity per part
    avg_qty = joined_data.groupby("p_partkey", as_index=False).agg(
        {"l_quantity": "mean"}
    )
    avg_qty["avg_quantity"] = avg_qty["l_quantity"] * 0.2

    # Apply threshold filter and calculate final result
    final_result = joined_data.merge(
        avg_qty[["p_partkey", "avg_quantity"]], on="p_partkey"
    )
    sum_value = (
        final_result[final_result["l_quantity"] < final_result["avg_quantity"]][
            "l_extendedprice"
        ].sum()
        / 7.0
    )

    return pd.DataFrame({"avg_yearly": [sum_value]})


def bench_q17():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q17", bench_q17)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=False)
