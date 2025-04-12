import pyperf

from dask import dataframe as dd
from tpch.utils import (
    get_part_supp_ds,
    get_supplier_ds,
    get_nation_ds,
    export_df,
)

Q_NUM = 11


def get_ds():
    partsupp = get_part_supp_ds("dask")
    supplier = get_supplier_ds("dask")
    nation = get_nation_ds("dask")

    return partsupp, supplier, nation


def query() -> dd.DataFrame:
    partsupp, supplier, nation = get_ds()

    var1 = "GERMANY"

    # Get supplier count and handle potential divide-by-zero
    supplier_count = supplier.shape[0].compute()
    # FRACTION is chosen as 0.0001 / SF
    # S_SUPPKEY identifier SF*10,000 are populated
    sumres_rate = 0.0001 / (supplier_count / 10000)

    # Step-by-step approach for better readability
    merged_data = partsupp.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
    with_nation = merged_data.merge(
        nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Filter by nation
    filtered_data = with_nation[with_nation["n_name"] == var1]

    # Calculate value
    filtered_data["value"] = (
        filtered_data["ps_supplycost"] * filtered_data["ps_availqty"]
    )

    # Group by part key and sum values
    grouped = filtered_data.groupby("ps_partkey")
    agg_result = grouped.agg({"value": "sum"}).reset_index()

    # We need to compute the total sum for filtering
    total_value_sum = agg_result["value"].sum().compute()
    threshold = total_value_sum * sumres_rate

    # Filter by threshold and sort
    filtered_result = agg_result[agg_result["value"] > threshold]
    sorted_result = filtered_result.sort_values(
        by=["value", "ps_partkey"], ascending=[False, True]
    ).reset_index(drop=True)

    return sorted_result.compute()


def bench_q11():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask-q11", bench_q11)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
