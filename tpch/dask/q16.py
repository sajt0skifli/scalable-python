import pyperf

from dask import dataframe as dd
from dask.distributed import Client
from tpch.utils import (
    get_supplier_ds,
    get_part_supp_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 16


def get_ds():
    supplier = get_supplier_ds("dask")
    partsupp = get_part_supp_ds("dask")
    part = get_part_ds("dask")

    return supplier, partsupp, part


def query() -> dd.DataFrame:
    supplier, partsupp, part = get_ds()

    var1 = "Brand#45"

    # Filter suppliers with complaints
    complaint_suppliers = supplier[
        supplier["s_comment"].str.contains(".*Customer.*Complaints.*")
    ][["s_suppkey"]]
    filtered_parts = part[
        (part["p_brand"] != var1)
        & (~part["p_type"].str.startswith("MEDIUM POLISHED"))
        & (part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9]))
    ]

    # Merge parts with part suppliers
    parts_with_suppliers = filtered_parts.merge(
        partsupp, left_on="p_partkey", right_on="ps_partkey"
    )
    merged = parts_with_suppliers.merge(
        complaint_suppliers, left_on="ps_suppkey", right_on="s_suppkey", how="left"
    )
    valid_parts = merged[merged["s_suppkey"].isna()]

    # Group by brand, type, size and count unique suppliers
    grouped = (
        valid_parts.groupby(["p_brand", "p_type", "p_size"])
        .ps_suppkey.nunique()
        .to_frame(name="supplier_cnt")
        .reset_index()
    )

    # Sort by supplier count (descending) and other fields (ascending)
    result = grouped.sort_values(
        ["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result.compute()


def bench_q16():
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
    runner.bench_func("dask-q16", bench_q16)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
