import pyperf

from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 22


def get_ds():
    customer = get_customer_ds()
    orders = get_orders_ds()

    return customer, orders


def query():
    customer, orders = get_ds()

    q1 = customer.assign(cntrycode=customer["c_phone"].str.slice(0, 2)).pipe(
        lambda df: df[df["cntrycode"].isin(["13", "31", "23", "29", "30", "18", "17"])]
    )

    q2 = q1[q1["c_acctbal"] > 0.0]["c_acctbal"].mean()

    # q3 = orders.groupby("o_custkey", as_index=False).size()
    q3 = orders.drop_duplicates(subset=["o_custkey"])
    # q3 = orders[["o_custkey"]].drop_duplicates(ignore_index=True)

    q_final = (
        q1.merge(q3, left_on="c_custkey", right_on="o_custkey", how="left")
        .pipe(lambda df: df[df["o_custkey"].isnull()])
        .pipe(lambda df: df[df["c_acctbal"] > q2])
        .groupby("cntrycode", as_index=False)
        .agg(numcust=("c_acctbal", "count"), totacctbal=("c_acctbal", "sum"))
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
    runner.bench_func("pandas-q22", bench_q22)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
