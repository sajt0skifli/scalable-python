import pyperf
import pandas as pd

from dask.distributed import Client
from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 22


def get_ds():
    customer = get_customer_ds("dask")
    orders = get_orders_ds("dask")

    return customer, orders


def query() -> pd.DataFrame:
    customer, orders = get_ds()

    # Extract country code and filter by specific codes
    filtered_customers = customer.assign(
        cntrycode=customer["c_phone"].str.slice(0, 2)
    ).pipe(
        lambda df: df[df["cntrycode"].isin(["13", "31", "23", "29", "30", "18", "17"])]
    )

    positive_balance = filtered_customers[filtered_customers["c_acctbal"] > 0.0]

    avg_balance = positive_balance["c_acctbal"].mean().compute()

    # Get unique customer keys from orders
    orders_unique_customers = orders[["o_custkey"]].drop_duplicates()

    # Perform left join
    result = (
        filtered_customers.merge(
            orders_unique_customers,
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )
        .pipe(lambda df: df[df["o_custkey"].isna()])
        .pipe(lambda df: df[df["c_acctbal"] > avg_balance])
        .groupby("cntrycode")
        .agg(numcust=("c_acctbal", "count"), totacctbal=("c_acctbal", "sum"))
        .reset_index()
        .sort_values("cntrycode")
    )

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
    runner.bench_func("dask-q22", bench_q22)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
