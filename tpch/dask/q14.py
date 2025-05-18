import pyperf
import pandas as pd

from datetime import date
from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 14


def get_ds():
    lineitem = get_line_item_ds("dask")
    part = get_part_ds("dask")

    return lineitem, part


def query() -> pd.DataFrame:
    lineitem, part = get_ds()

    var1 = date(1995, 9, 1)
    var2 = date(1995, 10, 1)

    merged_data = lineitem.merge(part, left_on="l_partkey", right_on="p_partkey")

    # Filter by date range
    filtered_data = merged_data[
        (merged_data["l_shipdate"] >= var1) & (merged_data["l_shipdate"] < var2)
    ]

    # Create column for revenue calculations
    filtered_data["revenue"] = filtered_data["l_extendedprice"] * (
        1 - filtered_data["l_discount"]
    )

    filtered_data["is_promo"] = filtered_data["p_type"].str.startswith("PROMO")
    computed_data = filtered_data.compute()
    promo_revenue = computed_data[computed_data["is_promo"]]["revenue"].sum()
    total_revenue = computed_data["revenue"].sum()

    promo_percentage = 100.00 * promo_revenue / total_revenue

    result = pd.DataFrame({"promo_revenue": [round(promo_percentage, 2)]})

    return result


def bench_q14():
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
    runner.bench_func("dask-q14", bench_q14)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
