import pyperf

from tpch.utils import (
    get_part_supp_ds,
    get_supplier_ds,
    get_nation_ds,
    export_df,
)

Q_NUM = 11


def get_ds():
    partsupp = get_part_supp_ds()
    supplier = get_supplier_ds()
    nation = get_nation_ds()

    return partsupp, supplier, nation


def query():
    partsupp, supplier, nation = get_ds()

    var1 = "GERMANY"
    # https://www.tpc.org/tpc_documents_current_versions/pdf/tpc-h_v3.0.1.pdf
    # FRACTION is chosen as 0.0001 / SF
    # S_SUPPKEY identifier SF*10,000 are populated
    sumres_rate = 0.0001 / (supplier.shape[0] / 10000)

    # using q1 as polars impl prevents move projection. See GT #83

    q_final = (
        partsupp.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
        .pipe(lambda df: df[df["n_name"] == var1])
        .assign(value=lambda df: df["ps_supplycost"] * df["ps_availqty"])
        .groupby("ps_partkey", as_index=False, sort=False)
        .agg({"value": "sum"})
        .pipe(lambda df: df[df["value"] > df["value"].sum() * sumres_rate])
        .sort_values(by=["value", "ps_partkey"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return q_final


def bench_q11():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q11", bench_q11)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
