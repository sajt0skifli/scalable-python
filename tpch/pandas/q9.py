import tpch.utils as utils

Q_NUM = 9


def query():
    part = utils.get_part_ds()
    supplier = utils.get_supplier_ds()
    lineitem = utils.get_line_item_ds()
    partsupp = utils.get_part_supp_ds()
    orders = utils.get_orders_ds()
    nation = utils.get_nation_ds()

    q_final = (
        part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(
            lineitem,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .merge(orders, left_on="l_orderkey", right_on="o_orderkey")
        .merge(nation, left_on="s_nationkey", right_on="n_nationkey")
        .pipe(lambda df: df[df["p_name"].str.contains("green", regex=False)])
        .assign(
            o_year=lambda df: df["o_orderdate"].dt.year,
            amount=lambda df: df["l_extendedprice"] * (1 - df["l_discount"])
            - (df["ps_supplycost"] * df["l_quantity"]),
        )
        .rename(columns={"n_name": "nation"})
        .groupby(["nation", "o_year"], as_index=False, sort=False)
        .agg(sum_profit=("amount", "sum"))
        .sort_values(by=["nation", "o_year"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return q_final


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    utils.export_df(result, file_name)
