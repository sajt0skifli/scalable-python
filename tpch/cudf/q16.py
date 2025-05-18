import pyperf
import cudf

from tpch.utils import (
    get_supplier_ds,
    get_part_supp_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 16


def get_ds():
    supplier = get_supplier_ds(mode="cudf")
    partsupp = get_part_supp_ds(mode="cudf")
    part = get_part_ds(mode="cudf")

    return supplier, partsupp, part


def query():
    supplier, partsupp, part = get_ds()

    var1 = "Brand#45"

    complaint_suppliers = supplier[
        supplier["s_comment"].str.contains(".*Customer.*Complaints.*")
    ][["s_suppkey"]]

    result_df = (
        part[
            (part["p_brand"] != var1)
            & (~part["p_type"].str.startswith("MEDIUM POLISHED"))
            & part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])
        ][["p_partkey", "p_brand", "p_type", "p_size"]]
        .merge(
            partsupp[["ps_partkey", "ps_suppkey"]],
            left_on="p_partkey",
            right_on="ps_partkey",
        )
        .merge(
            complaint_suppliers, left_on="ps_suppkey", right_on="s_suppkey", how="left"
        )
    )

    q_final = (
        result_df[result_df["s_suppkey"].isna()]
        .groupby(["p_brand", "p_type", "p_size"], as_index=False, sort=False)
        .agg({"ps_suppkey": "nunique"})
        .rename(columns={"ps_suppkey": "supplier_cnt"})
        .sort_values(
            ["supplier_cnt", "p_brand", "p_type", "p_size"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
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
    runner.bench_func("cudf-q16", bench_q16)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
