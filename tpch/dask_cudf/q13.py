import pyperf
import dask_cudf

from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 13


def get_ds():
    customer = get_customer_ds(mode="cudask")
    orders = get_orders_ds(mode="cudask")

    return customer, orders


def query():
    customer, orders = get_ds()

    var1 = "special"
    var2 = "requests"

    # Filter using dask_cudf string operations
    filtered_orders = orders[
        ~orders["o_comment"].str.contains(f"{var1}.*{var2}", regex=True)
    ]

    # Perform left join and chain operations
    # Add compute() at the end to materialize the result
    q_final = (
        customer.merge(
            filtered_orders, how="left", left_on="c_custkey", right_on="o_custkey"
        )
        .groupby("c_custkey")
        .agg({"o_orderkey": "count"})
        .rename(columns={"o_orderkey": "c_count"})
        .reset_index()
        .groupby("c_count")
        .agg({"c_custkey": "count"})
        .rename(columns={"c_custkey": "custdist"})
        .reset_index()
        .sort_values(by=["custdist", "c_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return q_final.compute()


def bench_q13():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q13", bench_q13)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
