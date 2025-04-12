import pyperf
import pandas as pd

from datetime import date
from dask import dataframe as dd
from tpch.utils import (
    get_line_item_ds,
    export_df,
)

Q_NUM = 6


def get_ds():
    line_item_ds = get_line_item_ds("dask")

    return line_item_ds


def query() -> dd.DataFrame:
    line_item_ds = get_ds()

    var1 = date(1994, 1, 1)
    var2 = date(1995, 1, 1)
    var3 = 0.05
    var4 = 0.07
    var5 = 24

    filt = line_item_ds[
        (line_item_ds["l_shipdate"] >= var1) & (line_item_ds["l_shipdate"] < var2)
    ]
    filt = filt[(filt["l_discount"] >= var3) & (filt["l_discount"] <= var4)]
    filt = filt[filt["l_quantity"] < var5]
    result_value = (filt["l_extendedprice"] * filt["l_discount"]).sum().compute()
    result_df = pd.DataFrame({"revenue": [result_value]})

    return result_df


def bench_q6():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask-q6", bench_q6)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
