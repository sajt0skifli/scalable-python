import pyperf

from tpch.utils import (
    get_line_item_ds,
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 18


def get_ds():
    lineitem = get_line_item_ds()
    customer = get_customer_ds()
    orders = get_orders_ds()

    return lineitem, customer, orders


def query():
    lineitem, customer, orders = get_ds()

    var1 = 300

    q1 = (
        lineitem.groupby("l_orderkey", as_index=False)
        .agg(sum_quantity=("l_quantity", "sum"))
        .pipe(lambda df: df[df["sum_quantity"] > var1])
    )

    q_final = (
        orders.merge(q1, left_on="o_orderkey", right_on="l_orderkey")
        .merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .merge(customer, left_on="o_custkey", right_on="c_custkey")
        .groupby(
            ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"],
            as_index=False,
        )
        .agg(col6=("l_quantity", "sum"))
        .sort_values(["o_totalprice", "o_orderdate"], ascending=[False, True])
        .rename(columns={"o_orderdate": "o_orderdat"})
        .astype({"col6": "float64"})
        .head(100)
    )

    return q_final


def bench_q18():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q18", bench_q18)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
