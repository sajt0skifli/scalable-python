import pandas as pd

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 14


def query() -> pd.DataFrame:
    lineitem = get_line_item_ds("dask")
    part = get_part_ds("dask")

    var1 = date(1995, 9, 1)
    var2 = date(1995, 10, 1)

    # Merge lineitem and part
    merged_data = lineitem.merge(part, left_on="l_partkey", right_on="p_partkey")

    # Filter by date range
    filtered_data = merged_data[
        (merged_data["l_shipdate"] >= var1) & (merged_data["l_shipdate"] < var2)
    ]

    # Create column for revenue calculations
    filtered_data["revenue"] = filtered_data["l_extendedprice"] * (
        1 - filtered_data["l_discount"]
    )

    # Identify promo parts
    filtered_data["is_promo"] = filtered_data["p_type"].str.startswith("PROMO")

    # Calculate promo revenue
    # This requires computing actual values
    computed_data = filtered_data.compute()

    # Sum of revenue from promo parts
    promo_revenue = computed_data[computed_data["is_promo"]]["revenue"].sum()

    # Total revenue
    total_revenue = computed_data["revenue"].sum()

    # Calculate percentage
    promo_percentage = 100.00 * promo_revenue / total_revenue

    # Format as dataframe with rounded value
    result = pd.DataFrame({"promo_revenue": [round(promo_percentage, 2)]})

    return result


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
