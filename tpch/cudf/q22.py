import pyperf
import cudf

from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 22


def get_ds():
    customer = get_customer_ds(mode="cudf")
    orders = get_orders_ds(mode="cudf")

    return customer, orders


def query():
    customer, orders = get_ds()

    cntry_codes = ["13", "31", "23", "29", "30", "18", "17"]
    q1 = customer[["c_custkey", "c_phone", "c_acctbal"]]

    q1["cntrycode"] = q1["c_phone"].str.slice(0, 2)
    q1 = q1[q1["cntrycode"].isin(cntry_codes)]

    q2 = q1[q1["c_acctbal"] > 0.0]["c_acctbal"].mean()
    q3 = orders[["o_custkey"]].drop_duplicates()

    merged_df = q1.merge(q3, left_on="c_custkey", right_on="o_custkey", how="left")
    filtered_df = merged_df[
        merged_df["o_custkey"].isna() & (merged_df["c_acctbal"] > q2)
    ]

    q_final = (
        filtered_df.groupby("cntrycode")
        .agg(numcust=("c_acctbal", "count"), totacctbal=("c_acctbal", "sum"))
        .reset_index()
        .sort_values("cntrycode")
    )

    return q_final


def bench_q22():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q22", bench_q22)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
