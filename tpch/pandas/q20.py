import pyperf
import tpch.utils as utils

from datetime import date

Q_NUM = 20


def get_ds():
    lineitem = utils.get_line_item_ds()
    nation = utils.get_nation_ds()
    supplier = utils.get_supplier_ds()
    partsupp = utils.get_part_supp_ds()
    part = utils.get_part_ds()

    return lineitem, nation, supplier, partsupp, part


def query():
    lineitem, nation, supplier, partsupp, part = get_ds()

    var1 = date(1994, 1, 1)
    var2 = date(1995, 1, 1)
    var3 = "CANADA"
    var4 = "forest"

    q1 = (
        lineitem[(lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)]
        .groupby(["l_partkey", "l_suppkey"], as_index=False)
        .agg(sum_quantity=("l_quantity", "sum"))
        .assign(sum_quantity=lambda df: df["sum_quantity"] * 0.5)
    )
    q2 = nation[nation["n_name"] == var3]
    q3 = supplier.merge(q2, left_on="s_nationkey", right_on="n_nationkey")

    ret = (
        part[part["p_name"].str.startswith(var4)]
        .drop_duplicates()
        .merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .merge(
            q1,
            left_on=["ps_suppkey", "ps_partkey"],
            right_on=["l_suppkey", "l_partkey"],
        )
        .pipe(lambda df: df[df["ps_availqty"] > df["sum_quantity"]])
        .drop_duplicates(subset=["ps_suppkey"])
        .merge(q3, left_on="ps_suppkey", right_on="s_suppkey")
        .sort_values("s_name")[["s_name", "s_address"]]
    )
    return ret


def bench_q20():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q20", bench_q20)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
