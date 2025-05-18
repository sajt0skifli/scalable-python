import pyperf
import pandas as pd

from datetime import date
from dask import dataframe as dd
from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 4


def get_ds():
    line_item_ds = get_line_item_ds("dask")
    orders_ds = get_orders_ds("dask")

    return line_item_ds, orders_ds


def query() -> dd.DataFrame:
    line_item_ds, orders_ds = get_ds()

    var1 = date(1993, 7, 1)
    var2 = date(1993, 10, 1)

    exists = line_item_ds[line_item_ds["l_commitdate"] < line_item_ds["l_receiptdate"]]

    jn = orders_ds.merge(
        exists, left_on="o_orderkey", right_on="l_orderkey", how="leftsemi"
    )
    jn = jn[(jn["o_orderdate"] >= var1) & (jn["o_orderdate"] < var2)]

    gb = jn.groupby("o_orderpriority")
    agg = gb.agg(
        order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count")
    ).reset_index()

    result_df = agg.sort_values(["o_orderpriority"])

    return result_df.compute()  # type: ignore[no-any-return]


def bench_q4():
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
    runner.bench_func("dask-q4", bench_q4)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
