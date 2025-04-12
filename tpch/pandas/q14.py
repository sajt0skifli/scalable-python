import pyperf

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 14


def get_ds():
    lineitem = get_line_item_ds()
    part = get_part_ds()

    return lineitem, part


def query():
    lineitem, part = get_ds()

    var1 = date(1995, 9, 1)
    var2 = date(1995, 10, 1)

    q_final = (
        lineitem.merge(part, left_on="l_partkey", right_on="p_partkey")
        .pipe(lambda df: df[(df["l_shipdate"] < var2) & (df["l_shipdate"] >= var1)])
        .pipe(
            lambda df: (
                100.00
                * (df["l_extendedprice"] * (1 - df["l_discount"]))
                .where(df["p_type"].str.startswith("PROMO"))
                .agg(["sum"])
                / (df["l_extendedprice"] * (1 - df["l_discount"])).agg(["sum"])
            )
        )
        .round(2)
        .to_frame(name="promo_revenue")
    )

    return q_final


def bench_q14():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q14", bench_q14)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
