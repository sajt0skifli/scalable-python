import pyperf
import cudf
import numpy as np
import tpch.utils as utils

from datetime import date

Q_NUM = 20


def get_ds():
    lineitem = utils.get_line_item_ds(mode="cudf")
    nation = utils.get_nation_ds(mode="cudf")
    supplier = utils.get_supplier_ds(mode="cudf")
    partsupp = utils.get_part_supp_ds(mode="cudf")
    part = utils.get_part_ds(mode="cudf")

    return lineitem, nation, supplier, partsupp, part


def query():
    lineitem, nation, supplier, partsupp, part = get_ds()

    var1 = np.datetime64(date(1994, 1, 1))
    var2 = np.datetime64(date(1995, 1, 1))
    var3 = "CANADA"
    var4 = "forest"

    q1 = (
        lineitem[(lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)][
            ["l_partkey", "l_suppkey", "l_quantity"]
        ]
        .groupby(["l_partkey", "l_suppkey"], as_index=False)
        .agg({"l_quantity": "sum"})
        .rename(columns={"l_quantity": "sum_quantity"})
    )

    q1["sum_quantity"] *= 0.5

    q2 = supplier[["s_suppkey", "s_nationkey", "s_name", "s_address"]].merge(
        nation[nation["n_name"] == var3][["n_nationkey"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    filtered_join = (
        part[part["p_name"].str.startswith(var4)][["p_partkey"]]
        .merge(
            partsupp[["ps_partkey", "ps_suppkey", "ps_availqty"]],
            left_on="p_partkey",
            right_on="ps_partkey",
        )
        .merge(
            q1,
            left_on=["ps_suppkey", "ps_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        .query("ps_availqty > sum_quantity")[["ps_suppkey"]]
        .drop_duplicates()
    )

    result = filtered_join.merge(q2, left_on="ps_suppkey", right_on="s_suppkey")[
        ["s_name", "s_address"]
    ].sort_values("s_name")

    return result


def bench_q20():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q20", bench_q20)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # utils.export_df(result, file_name, is_cudf=True)
