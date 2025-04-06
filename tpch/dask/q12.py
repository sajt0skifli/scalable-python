from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_orders_ds,
    export_df,
)
from dask import dataframe as dd

Q_NUM = 12


def query() -> dd.DataFrame:
    lineitem = get_line_item_ds("dask")
    orders = get_orders_ds("dask")

    var1 = "MAIL"
    var2 = "SHIP"
    var3 = date(1994, 1, 1)
    var4 = date(1995, 1, 1)
    high_priorities = ["1-URGENT", "2-HIGH"]

    # Step-by-step approach for better readability
    merged_data = orders.merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Apply filters
    filtered_data = merged_data[merged_data["l_shipmode"].isin([var1, var2])]
    filtered_data = filtered_data[
        filtered_data["l_commitdate"] < filtered_data["l_receiptdate"]
    ]
    filtered_data = filtered_data[
        filtered_data["l_shipdate"] < filtered_data["l_commitdate"]
    ]
    filtered_data = filtered_data[
        (filtered_data["l_receiptdate"] >= var3)
        & (filtered_data["l_receiptdate"] < var4)
    ]

    # Create indicator columns
    filtered_data["high_line_count"] = filtered_data["o_orderpriority"].isin(
        high_priorities
    )
    filtered_data["low_line_count"] = ~filtered_data["o_orderpriority"].isin(
        high_priorities
    )

    # Group and aggregate
    grouped = filtered_data.groupby("l_shipmode")
    agg_result = grouped.agg(
        {"high_line_count": "sum", "low_line_count": "sum"}
    ).reset_index()

    # Convert boolean sums to integers
    final_result = agg_result.astype({"high_line_count": int, "low_line_count": int})

    # Sort by l_shipmode (sort=True in pandas)
    sorted_result = final_result.sort_values(by="l_shipmode").reset_index(drop=True)

    return sorted_result.compute()


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
