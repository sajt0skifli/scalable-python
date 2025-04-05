from datetime import date
from dask import dataframe as dd
from tpch.utils import (
    get_customer_ds,
    get_line_item_ds,
    get_orders_ds,
    export_df,
)

Q_NUM = 3


def query() -> dd.DataFrame:
    customer_ds = get_customer_ds("dask")
    line_item_ds = get_line_item_ds("dask")
    orders_ds = get_orders_ds("dask")

    var1 = "BUILDING"
    var2 = date(1995, 3, 15)

    fcustomer = customer_ds[customer_ds["c_mktsegment"] == var1]

    jn1 = fcustomer.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")

    jn2 = jn2[jn2["o_orderdate"] < var2]
    jn2 = jn2[jn2["l_shipdate"] > var2]
    jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

    gb = jn2.groupby(["o_orderkey", "o_orderdate", "o_shippriority"])
    agg = gb["revenue"].sum().reset_index()

    sel = agg.loc[:, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    sel = sel.rename(columns={"o_orderkey": "l_orderkey"})

    sorted = sel.sort_values(by=["revenue", "o_orderdate"], ascending=[False, True])
    result_df = sorted.head(10)

    return result_df  # type: ignore[no-any-return]


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
