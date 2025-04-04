from tpch.utils import (
    get_customer_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 13


def query():
    customer = get_customer_ds()
    orders = get_orders_ds()

    var1 = "special"
    var2 = "requests"

    orders = orders[~orders["o_comment"].str.contains(f"{var1}.*{var2}", regex=True)]

    q_final = (
        customer.merge(orders, how="left", left_on="c_custkey", right_on="o_custkey")
        .groupby("c_custkey", as_index=False, sort=False)
        .agg(c_count=("o_orderkey", "count"))
        .groupby("c_count", as_index=False, sort=False)
        .agg(custdist=("c_custkey", "count"))
        .sort_values(by=["custdist", "c_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return q_final


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
