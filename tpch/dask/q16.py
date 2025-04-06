from tpch.utils import (
    get_supplier_ds,
    get_part_supp_ds,
    get_part_ds,
    export_df,
)
from dask import dataframe as dd

Q_NUM = 16


def query() -> dd.DataFrame:
    supplier = get_supplier_ds("dask")
    partsupp = get_part_supp_ds("dask")
    part = get_part_ds("dask")

    var1 = "Brand#45"

    # Filter suppliers with complaints
    complaint_suppliers = supplier[
        supplier["s_comment"].str.contains(".*Customer.*Complaints.*")
    ][["s_suppkey"]]

    # Filter parts based on multiple conditions
    filtered_parts = part[
        (part["p_brand"] != var1)
        & (~part["p_type"].str.startswith("MEDIUM POLISHED"))
        & (part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9]))
    ]

    # Merge parts with part suppliers
    parts_with_suppliers = filtered_parts.merge(
        partsupp, left_on="p_partkey", right_on="ps_partkey"
    )

    # Perform anti-join by marking rows to exclude then filtering
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

    # Convert back to Dask DataFrame for consistency
    return result.compute()


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
