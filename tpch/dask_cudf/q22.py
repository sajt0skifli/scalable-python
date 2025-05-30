import pyperf
import dask_cudf

from dask.distributed import Client
from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 22


def get_ds():
    customer = get_customer_ds(mode="cudask")
    orders = get_orders_ds(mode="cudask")

    return customer, orders


def query():
    customer, orders = get_ds()

    cntry_codes = ["13", "31", "23", "29", "30", "18", "17"]

    customer_filtered = (
        customer[["c_custkey", "c_phone", "c_acctbal"]]
        .assign(cntrycode=lambda df: df["c_phone"].str.slice(0, 2))
        .loc[lambda df: df["cntrycode"].isin(cntry_codes)]
    )

    avg_acctbal = (
        customer_filtered.loc[lambda df: df["c_acctbal"] > 0.0]["c_acctbal"]
        .mean()
        .compute()
    )

    orders_custkeys = orders[["o_custkey"]].drop_duplicates()

    q_final = (
        customer_filtered.merge(
            orders_custkeys, left_on="c_custkey", right_on="o_custkey", how="left"
        )
        .loc[lambda df: df["o_custkey"].isna() & (df["c_acctbal"] > avg_acctbal)]
        .groupby("cntrycode")
        .agg({"c_acctbal": ["count", "sum"]})
        .reset_index()
    )

    q_final.columns = ["cntrycode", "numcust", "totacctbal"]

    result = q_final.sort_values("cntrycode")

    return result.compute()


def bench_q22():
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
    runner.bench_func("dask_cudf-q22", bench_q22)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
