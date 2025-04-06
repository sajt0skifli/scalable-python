from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)
from dask import dataframe as dd

Q_NUM = 13


def query() -> dd.DataFrame:
    customer = get_customer_ds("dask")
    orders = get_orders_ds("dask")

    var1 = "special"
    var2 = "requests"

    # Filter out orders containing the specified pattern in comments
    # Note: Dask string operations are similar to pandas but might behave differently with complex regex
    orders = orders[~orders["o_comment"].str.contains(f"{var1}.*{var2}", regex=True)]

    # Left merge to include customers without orders
    merged_data = customer.merge(
        orders, how="left", left_on="c_custkey", right_on="o_custkey"
    )

    # Group by customer key and count orders
    customer_order_counts = (
        merged_data.groupby("c_custkey").agg({"o_orderkey": "count"}).reset_index()
    )
    customer_order_counts = customer_order_counts.rename(
        columns={"o_orderkey": "c_count"}
    )

    # Group by count and count customers
    count_distribution = (
        customer_order_counts.groupby("c_count")
        .agg({"c_custkey": "count"})
        .reset_index()
    )
    count_distribution = count_distribution.rename(columns={"c_custkey": "custdist"})

    # Sort by custdist (descending) and c_count (descending)
    sorted_result = count_distribution.sort_values(
        by=["custdist", "c_count"], ascending=[False, False]
    ).reset_index(drop=True)

    # Materialize the result
    return sorted_result.compute()


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
