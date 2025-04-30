import pyperf
import dask_cudf
import pandas as pd
import numpy as np

from datetime import date
from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    export_df,
)

Q_NUM = 6


def get_ds():
    lineitem = get_line_item_ds("cudask")
    return lineitem


def query():
    lineitem = get_ds()

    var1 = np.datetime64(date(1994, 1, 1))
    var2 = np.datetime64(date(1995, 1, 1))
    var3 = 0.05
    var4 = 0.07
    var5 = 24

    # Use efficient filtering
    filtered = lineitem[
        (lineitem["l_shipdate"] >= var1)
        & (lineitem["l_shipdate"] < var2)
        & (lineitem["l_discount"] >= var3)
        & (lineitem["l_discount"] <= var4)
        & (lineitem["l_quantity"] < var5)
    ]

    # Calculate revenue and compute the result
    result = (filtered["l_extendedprice"] * filtered["l_discount"]).sum().compute()

    # Return result as a DataFrame
    return pd.DataFrame({"revenue": [result]})


def bench_q6():
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
    runner.bench_func("dask_cudf-q6", bench_q6)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=False)
