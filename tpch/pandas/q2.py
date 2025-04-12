import pyperf
import tpch.utils as utils

Q_NUM = 2


def get_ds():
    region = utils.get_region_ds()
    nation = utils.get_nation_ds()
    supplier = utils.get_supplier_ds()
    part = utils.get_part_ds()
    part_supp = utils.get_part_supp_ds()

    return region, nation, supplier, part, part_supp


def query():
    region, nation, supplier, part, part_supp = get_ds()

    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    q1 = (
        part.merge(part_supp, left_on="p_partkey", right_on="ps_partkey")
        .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
        .merge(region, left_on="n_regionkey", right_on="r_regionkey")
        .pipe(lambda df: df[df["p_size"] == var1])
        .pipe(lambda df: df[df["p_type"].str.endswith(var2)])
        .pipe(lambda df: df[df["r_name"] == var3])
    )

    q_final = (
        (
            q1.groupby("p_partkey", as_index=False)
            .agg({"ps_supplycost": "min"})
            .merge(q1, on=["p_partkey", "ps_supplycost"])
        )[
            [
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            ]
        ]
        .sort_values(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            ascending=[False, True, True, True],
        )
        .head(100)
    )

    return q_final


def bench_q2():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q2", bench_q2)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
