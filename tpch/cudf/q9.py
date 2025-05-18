import pyperf
import cudf
import tpch.utils as utils

Q_NUM = 9


def get_ds():
    part = utils.get_part_ds("cudf")
    supplier = utils.get_supplier_ds("cudf")
    lineitem = utils.get_line_item_ds("cudf")
    partsupp = utils.get_part_supp_ds("cudf")
    orders = utils.get_orders_ds("cudf")
    nation = utils.get_nation_ds("cudf")

    return part, supplier, lineitem, partsupp, orders, nation


def query():
    part, supplier, lineitem, partsupp, orders, nation = get_ds()

    filtered_part = part[part["p_name"].str.contains("green")][["p_partkey", "p_name"]]

    slim_partsupp = partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]]
    slim_supplier = supplier[["s_suppkey", "s_nationkey"]]
    slim_lineitem = lineitem[
        [
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
        ]
    ]
    slim_orders = orders[["o_orderkey", "o_orderdate"]]
    slim_nation = nation[["n_nationkey", "n_name"]]

    q_final = (
        filtered_part.merge(slim_partsupp, left_on="p_partkey", right_on="ps_partkey")
        .merge(slim_supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(
            slim_lineitem,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .merge(slim_orders, left_on="l_orderkey", right_on="o_orderkey")
        .merge(slim_nation, left_on="s_nationkey", right_on="n_nationkey")
        .assign(
            o_year=lambda df: df["o_orderdate"].dt.year,
            amount=lambda df: df["l_extendedprice"] * (1 - df["l_discount"])
            - (df["ps_supplycost"] * df["l_quantity"]),
        )
        .rename(columns={"n_name": "nation"})
        .groupby(["nation", "o_year"], sort=False)
        .agg({"amount": "sum"})
        .rename(columns={"amount": "sum_profit"})
        .reset_index()
        .sort_values(by=["nation", "o_year"], ascending=[True, False])
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
    runner.bench_func("cudf-q9", bench_q9)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # utils.export_df(result, file_name, is_cudf=True)
