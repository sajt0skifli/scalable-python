import pandas as pd
from tpch.utils import (
    get_line_item_ds,
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 18


def query() -> pd.DataFrame:
    lineitem = get_line_item_ds("dask")
    customer = get_customer_ds("dask")
    orders = get_orders_ds("dask")

    var1 = 300

    # Aggregate lineitem quantities by order key
    order_quantities = (
        lineitem.groupby("l_orderkey")
        .agg(sum_quantity=("l_quantity", "sum"))
        .reset_index()
    )

    # Filter orders with quantity > 300
    large_quantity_orders = order_quantities[order_quantities["sum_quantity"] > var1]

    # Join with orders
    result = (
        orders.merge(large_quantity_orders, left_on="o_orderkey", right_on="l_orderkey")
        # Join with lineitem for quantities
        .merge(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        # Join with customer data
        .merge(customer, left_on="o_custkey", right_on="c_custkey")
    )

    # Group by and sort for final result
    final_result = (
        result.groupby(
            ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"],
        )
        .agg(col6=("l_quantity", "sum"))
        .reset_index()
        .sort_values(["o_totalprice", "o_orderdate"], ascending=[False, True])
        .rename(columns={"o_orderdate": "o_orderdat"})
        .astype({"col6": "float64"})
        .head(100)
    )

    return final_result


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
