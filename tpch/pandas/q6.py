import pandas as pd

from datetime import date
from tpch.utils import (
    get_line_item_ds,
    export_df,
)

Q_NUM = 6


def query():
    lineitem = get_line_item_ds()

    var1 = date(1994, 1, 1)
    var2 = date(1995, 1, 1)
    var3 = 0.05
    var4 = 0.07
    var5 = 24

    result = (
        lineitem[
            (lineitem["l_shipdate"] < var2)
            & (lineitem["l_shipdate"] >= var1)
            & (lineitem["l_discount"] <= var4)
            & (lineitem["l_discount"] >= var3)
            & (lineitem["l_quantity"] < var5)
        ]
        .pipe(lambda df: df["l_extendedprice"] * df["l_discount"])
        .sum()
    )
    return pd.DataFrame({"revenue": [result]})


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
