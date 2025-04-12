import pyperf

from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 17


def get_ds():
    lineitem = get_line_item_ds()
    part = get_part_ds()

    return lineitem, part


def query():
    lineitem, part = get_ds()

    var1 = "Brand#23"
    var2 = "MED BOX"

    q1 = (
        part[part["p_brand"] == var1]
        .pipe(lambda df: df[df["p_container"] == var2])
        .merge(lineitem, how="left", left_on="p_partkey", right_on="l_partkey")
    )

    q_final = (
        q1.groupby("p_partkey", as_index=False)
        .agg(avg_quantity=("l_quantity", "mean"))
        .assign(avg_quantity=lambda df: df["avg_quantity"] * 0.2)
        .merge(q1, left_on="p_partkey", right_on="p_partkey")
        .pipe(lambda df: df[df["l_quantity"] < df["avg_quantity"]])
        .pipe(lambda df: df[["l_extendedprice"]].sum() / 7.0)
        .to_frame(name="avg_yearly")
    )

    return q_final


def bench_q17():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q17", bench_q17)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
