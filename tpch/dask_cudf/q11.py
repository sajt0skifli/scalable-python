import pyperf
import dask_cudf
import tpch.utils as utils

from dask.distributed import Client

Q_NUM = 11


def get_ds():
    partsupp = utils.get_part_supp_ds("cudask")
    supplier = utils.get_supplier_ds("cudask")
    nation = utils.get_nation_ds("cudask")

    return partsupp, supplier, nation


def query():
    partsupp, supplier, nation = get_ds()

    var1 = "GERMANY"
    sumres_rate = 0.0001 / (supplier.shape[0] / 10000)

    german_nation = nation[nation["n_name"] == var1][["n_nationkey"]]

    german_suppliers = supplier[["s_suppkey", "s_nationkey"]].merge(
        german_nation, left_on="s_nationkey", right_on="n_nationkey"
    )[["s_suppkey"]]

    partsupp = partsupp.assign(
        value=partsupp["ps_supplycost"] * partsupp["ps_availqty"]
    )

    filtered_partsupp = partsupp[["ps_partkey", "ps_suppkey", "value"]].merge(
        german_suppliers, left_on="ps_suppkey", right_on="s_suppkey"
    )

    grouped = (
        filtered_partsupp.groupby("ps_partkey").agg({"value": "sum"}).reset_index()
    )

    grouped_computed = grouped.compute()
    sum_value = grouped_computed["value"].sum()
    threshold = sum_value * sumres_rate

    q_final = (
        grouped_computed[grouped_computed["value"] > threshold]
        .sort_values(by=["value", "ps_partkey"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return q_final


def bench_q11():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    client = Client()
    print(client.scheduler_info)
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q11", bench_q11)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # utils.export_df(result, file_name, is_cudf=True)
