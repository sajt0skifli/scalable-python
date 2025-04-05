import pandas as pd
import tpch.utils as utils

from datetime import date
from dask import dataframe as dd

Q_NUM = 7


def query() -> dd.DataFrame:
    nation_ds = utils.get_nation_ds("dask")
    customer_ds = utils.get_customer_ds("dask")
    line_item_ds = utils.get_line_item_ds("dask")
    orders_ds = utils.get_orders_ds("dask")
    supplier_ds = utils.get_supplier_ds("dask")

    var1 = "FRANCE"
    var2 = "GERMANY"
    var3 = date(1995, 1, 1)
    var4 = date(1996, 12, 31)

    n1 = nation_ds[(nation_ds["n_name"] == var1)]
    n2 = nation_ds[(nation_ds["n_name"] == var2)]

    # Part 1
    jn1 = customer_ds.merge(n1, left_on="c_nationkey", right_on="n_nationkey")
    jn2 = jn1.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn2.rename(columns={"n_name": "cust_nation"})
    jn3 = jn2.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn4 = jn3.merge(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
    jn5 = jn4.merge(n2, left_on="s_nationkey", right_on="n_nationkey")
    df1 = jn5.rename(columns={"n_name": "supp_nation"})

    # Part 2
    jn1 = customer_ds.merge(n2, left_on="c_nationkey", right_on="n_nationkey")
    jn2 = jn1.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn2.rename(columns={"n_name": "cust_nation"})
    jn3 = jn2.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn4 = jn3.merge(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
    jn5 = jn4.merge(n1, left_on="s_nationkey", right_on="n_nationkey")
    df2 = jn5.rename(columns={"n_name": "supp_nation"})

    # Combine
    total = dd.concat([df1, df2])  # type: ignore[attr-defined,no-untyped-call]

    total = total[(total["l_shipdate"] >= var3) & (total["l_shipdate"] <= var4)]
    total["volume"] = total["l_extendedprice"] * (1.0 - total["l_discount"])
    total["l_year"] = total["l_shipdate"].dt.year

    gb = total.groupby(["supp_nation", "cust_nation", "l_year"])
    agg = gb.agg(revenue=pd.NamedAgg(column="volume", aggfunc="sum")).reset_index()

    result_df = agg.sort_values(by=["supp_nation", "cust_nation", "l_year"])

    return result_df.compute()  # type: ignore[no-any-return]


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    utils.export_df(result, file_name)
