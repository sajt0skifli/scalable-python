import pyperf
import tpch.utils as utils

from datetime import date

Q_NUM = 8


def get_ds():
    customer = utils.get_customer_ds()
    orders = utils.get_orders_ds()
    lineitem = utils.get_line_item_ds()
    part = utils.get_part_ds()
    supplier = utils.get_supplier_ds()
    nation = utils.get_nation_ds()
    region = utils.get_region_ds()

    return customer, orders, lineitem, part, supplier, nation, region


def query():
    customer, orders, lineitem, part, supplier, nation, region = get_ds()

    var1 = "BRAZIL"
    var2 = "AMERICA"
    var3 = "ECONOMY ANODIZED STEEL"
    var4 = date(1995, 1, 1)
    var5 = date(1996, 12, 31)

    n1 = nation[["n_nationkey", "n_regionkey"]]
    n2 = nation[["n_nationkey", "n_name"]]

    result = (
        part.merge(lineitem, left_on="p_partkey", right_on="l_partkey")
        .merge(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .merge(orders, left_on="l_orderkey", right_on="o_orderkey")
        .merge(customer, left_on="o_custkey", right_on="c_custkey")
        .merge(n1, left_on="c_nationkey", right_on="n_nationkey")
        .merge(region, left_on="n_regionkey", right_on="r_regionkey")
        .pipe(lambda df: df[df["r_name"] == var2])
        .merge(n2, left_on="s_nationkey", right_on="n_nationkey")
        .pipe(lambda df: df[(df["o_orderdate"] <= var5) & (df["o_orderdate"] >= var4)])
        .pipe(lambda df: df[df["p_type"] == var3])
        .assign(
            volume=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]),
            o_year=lambda df: df["o_orderdate"].dt.year,
            case_volume=lambda df: df["volume"].where(df["n_name"] == var1),
        )
        .groupby("o_year", as_index=False)
        .agg({"case_volume": "sum", "volume": "sum"})
        .assign(mkt_share=lambda df: df["case_volume"] / df["volume"])
        .round(2)
    )

    return result[["o_year", "mkt_share"]]


def bench_q8():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("pandas-q8", bench_q8)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name)
