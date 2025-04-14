import pyperf
import cudf

from tpch.utils import (
    get_line_item_ds,
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 18


def get_ds():
    lineitem = get_line_item_ds(mode="cudf")
    customer = get_customer_ds(mode="cudf")
    orders = get_orders_ds(mode="cudf")

    return lineitem, customer, orders


def query():
    lineitem, customer, orders = get_ds()

    var1 = 300

    # Calculate orders with sum of quantity > 300 (with column selection)
    filtered_orders = (
        lineitem[["l_orderkey", "l_quantity"]]
        .groupby("l_orderkey", as_index=False, sort=False)
        .agg({"l_quantity": "sum"})
        .rename(columns={"l_quantity": "sum_quantity"})
        .query(f"sum_quantity > {var1}")[["l_orderkey", "sum_quantity"]]
    )

    # Select only needed columns from each table
    orders_cols = orders[["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]]
    customer_cols = customer[["c_custkey", "c_name"]]
    lineitem_cols = lineitem[["l_orderkey", "l_quantity"]]

    # Perform joins with optimized column selection
    join1 = orders_cols.merge(
        filtered_orders, left_on="o_orderkey", right_on="l_orderkey"
    )
    join2 = join1.merge(lineitem_cols, left_on="o_orderkey", right_on="l_orderkey")
    join3 = join2.merge(customer_cols, left_on="o_custkey", right_on="c_custkey")

    # Group, sort and format results
    q_final = (
        join3.groupby(
            ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"],
            as_index=False,
            sort=False,
        )
        .agg({"l_quantity": "sum"})
        .rename(columns={"l_quantity": "col6", "o_orderdate": "o_orderdat"})
        .sort_values(["o_totalprice", "o_orderdat"], ascending=[False, True])
        .astype({"col6": "float64"})
        .head(100)
        .reset_index(drop=True)
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
    runner.bench_func("cudf-q18", bench_q18)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
