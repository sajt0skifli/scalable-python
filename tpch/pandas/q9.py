import pyperf
import tpch.utils as utils

Q_NUM = 9


def get_ds():
    part = utils.get_part_ds()
    supplier = utils.get_supplier_ds()
    lineitem = utils.get_line_item_ds()
    partsupp = utils.get_part_supp_ds()
    orders = utils.get_orders_ds()
    nation = utils.get_nation_ds()

    return part, supplier, lineitem, partsupp, orders, nation


def query():
    part, supplier, lineitem, partsupp, orders, nation = get_ds()

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


def bench_q9():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q9", bench_q9)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
