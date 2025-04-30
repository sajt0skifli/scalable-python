import pyperf
import dask_cudf
import tpch.utils as utils

from dask.distributed import Client

Q_NUM = 9


def get_ds():
    part = utils.get_part_ds("cudask")
    supplier = utils.get_supplier_ds("cudask")
    lineitem = utils.get_line_item_ds("cudask")
    partsupp = utils.get_part_supp_ds("cudask")
    orders = utils.get_orders_ds("cudask")
    nation = utils.get_nation_ds("cudask")

    return part, supplier, lineitem, partsupp, orders, nation


def query():
    part, supplier, lineitem, partsupp, orders, nation = get_ds()

    # Early filter on part name to reduce join size
    filtered_part = part[part["p_name"].str.contains("green")][["p_partkey", "p_name"]]

    # Select only needed columns to reduce memory usage
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

    # Perform the join chain with filtered/slimmed dataframes
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

    return q_final.compute()


def bench_q9():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    client = Client()
    print(client.scheduler_info)
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("dask_cudf-q9", bench_q9)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # utils.export_df(result, file_name, is_cudf=True)
