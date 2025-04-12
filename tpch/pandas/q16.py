import pyperf

from tpch.utils import (
    get_supplier_ds,
    get_part_supp_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 16


def get_ds():
    supplier = get_supplier_ds()
    partsupp = get_part_supp_ds()
    part = get_part_ds()

    return supplier, partsupp, part


def query():
    supplier, partsupp, part = get_ds()

    var1 = "Brand#45"

    supplier = supplier[supplier["s_comment"].str.contains(".*Customer.*Complaints.*")][
        ["s_suppkey"]
    ]

    q_final = (
        part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .pipe(lambda df: df[df["p_brand"] != var1])
        .pipe(lambda df: df[~(df["p_type"].str.startswith("MEDIUM POLISHED"))])
        .pipe(lambda df: df[df["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])])
        .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey", how="left")
        .pipe(lambda df: df[df["s_suppkey"].isnull()])
        .groupby(["p_brand", "p_type", "p_size"], as_index=False)
        .agg(supplier_cnt=("ps_suppkey", "nunique"))
        .sort_values(
            ["supplier_cnt", "p_brand", "p_type", "p_size"],
            ascending=[False, True, True, True],
        )
    )

    return q_final


def bench_q16():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q16", bench_q16)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
