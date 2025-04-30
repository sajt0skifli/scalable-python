import pyperf
import pandas as pd

from dask.distributed import Client
from tpch.utils import (
    get_line_item_ds,
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 18


def get_ds():
    lineitem = get_line_item_ds("dask")
    customer = get_customer_ds("dask")
    orders = get_orders_ds("dask")

    return lineitem, customer, orders


def query() -> pd.DataFrame:
    lineitem, customer, orders = get_ds()

    var1 = 300

    # Aggregate lineitem quantities by order key
    order_quantities = (
        lineitem.groupby("l_orderkey")
        .agg(sum_quantity=("l_quantity", "sum"))
        .reset_index()
    )

    # Filter orders with quantity > 300
    large_quantity_orders = order_quantities[order_quantities["sum_quantity"] > var1]

    # Join with orders
    result = (
        orders.merge(large_quantity_orders, left_on="o_orderkey", right_on="l_orderkey")
        # Join with lineitem for quantities
        .merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        # Join with customer data
        .merge(customer, left_on="o_custkey", right_on="c_custkey")
    )

    # Group by and sort for final result
    final_result = (
        result.groupby(
            ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"],
        )
        .agg(col6=("l_quantity", "sum"))
        .reset_index()
        .sort_values(["o_totalprice", "o_orderdate"], ascending=[False, True])
        .rename(columns={"o_orderdate": "o_orderdat"})
        .astype({"col6": "float64"})
        .head(100)
    )

    return final_result


def bench_q18():
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
    runner.bench_func("dask-q18", bench_q18)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
