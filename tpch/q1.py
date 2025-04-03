from datetime import date
from tpch.utils import (
    get_line_item_ds,
)


Q_NUM = 1


def query():
    lineitem = get_line_item_ds()

    q_final = (
        lineitem[lineitem["l_shipdate"] <= date(1998, 9, 2)]
        .assign(disc_price=lambda df: df["l_extendedprice"] * (1 - df["l_discount"]))
        .assign(charge=lambda df: df["disc_price"] * (1 + df["l_tax"]))
        .groupby(["l_returnflag", "l_linestatus"], as_index=False)
        .agg(
            sum_qty=("l_quantity", "sum"),
            sum_base_price=("l_extendedprice", "sum"),
            sum_disc_price=("disc_price", "sum"),
            sum_charge=("charge", "sum"),
            avg_qty=("l_quantity", "mean"),
            avg_price=("l_extendedprice", "mean"),
            avg_disc=("l_discount", "mean"),
            count_order=("l_returnflag", "count"),
        )
        .sort_values(["l_returnflag", "l_linestatus"])
    )

    return q_final


if __name__ == "__main__":
    result = query()
    print(f"Query result for query {Q_NUM}:")
    print(result)
