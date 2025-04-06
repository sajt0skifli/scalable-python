from datetime import date
from tpch.utils import (
    get_supplier_ds,
    get_line_item_ds,
    export_df,
)
from dask import dataframe as dd

Q_NUM = 15


def query() -> dd.DataFrame:
    supplier = get_supplier_ds("dask")
    lineitem = get_line_item_ds("dask")

    var1 = date(1996, 1, 1)
    var2 = date(1996, 4, 1)

    # Filter lineitem by date range
    filtered_lineitem = lineitem[
        (lineitem["l_shipdate"] >= var1) & (lineitem["l_shipdate"] < var2)
    ]

    # Calculate revenue
    filtered_lineitem["total_revenue"] = filtered_lineitem["l_extendedprice"] * (
        1 - filtered_lineitem["l_discount"]
    )

    # Group by supplier key and sum revenue
    revenue = (
        filtered_lineitem.groupby("l_suppkey")
        .agg({"total_revenue": "sum"})
        .reset_index()
    )

    # We need to compute the revenue to find the maximum value
    computed_revenue = revenue.compute()
    max_revenue = computed_revenue["total_revenue"].max()

    # Filter suppliers with maximum revenue
    max_revenue_suppliers = computed_revenue[
        computed_revenue["total_revenue"] == max_revenue
    ]

    # Merge with supplier information
    result = (
        supplier.merge(max_revenue_suppliers, left_on="s_suppkey", right_on="l_suppkey")
        .round({"total_revenue": 2})[
            ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]
        ]
        .sort_values("s_suppkey")
        .reset_index(drop=True)
    )

    return result.compute()


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
