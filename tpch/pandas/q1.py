import pyperf

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    export_df,
)

Q_NUM = 1


def get_ds():
    lineitem = get_line_item_ds()
    return lineitem


def query(lineitem):
    q_final = (
        lineitem[lineitem["l_shipdate"] <= date(1998, 9, 2)]
        .assign(disc_price=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]))
        .assign(charge=lambda df: df["disc_price"] * (1 + df["l_tax"]))
        .groupby(["l_returnflag", "l_linestatus"], as_index=False)
        .agg(
            sum_qty=("l_quantity", "sum"),
            sum_base_price=("l_extendedprice", "sum"),
            sum_disc_price=("disc_price", "sum"),
            sum_charge=("charge", "sum"),
            avg_qty=("l_quantity", "mean"),
            avg_price=("l_extendedprice", "mean"),
            avg_disc=("l_discount", "mean"),
            count_order=("l_returnflag", "count"),
        )
        .sort_values(["l_returnflag", "l_linestatus"])
    )

    return q_final


def bench_q1():
    lineitem = get_ds()

    t0 = pyperf.perf_counter()
    query(lineitem)
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    # runner = pyperf.Runner()
    # runner.argparser.set_defaults(
    #     quiet=False, loops=1, values=1, processes=1, warmups=0
    # )
    # runner.bench_func("pandas-q1", bench_q1)
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
