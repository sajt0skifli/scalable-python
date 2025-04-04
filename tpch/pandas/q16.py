from tpch.utils import (
    get_supplier_ds,
    get_part_supp_ds,
    get_part_ds,
    export_df,
)

Q_NUM = 16


def query():
    supplier = get_supplier_ds()
    partsupp = get_part_supp_ds()
    part = get_part_ds()

    var1 = "Brand#45"

    supplier = supplier[supplier["s_comment"].str.contains(".*Customer.*Complaints.*")][
        ["s_suppkey"]
    ]

    q_final = (
        part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .pipe(lambda df: df[df["p_brand"] != var1])
        .pipe(lambda df: df[~(df["p_type"].str.startswith("MEDIUM POLISHED"))])
        .pipe(lambda df: df[df["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])])
        .merge(supplier, left_on="ps_suppkey", right_on="s_suppkey", how="left")
        .pipe(lambda df: df[df["s_suppkey"].isnull()])
        .groupby(["p_brand", "p_type", "p_size"], as_index=False)
        .agg(supplier_cnt=("ps_suppkey", "nunique"))
        .sort_values(
            ["supplier_cnt", "p_brand", "p_type", "p_size"],
            ascending=[False, True, True, True],
        )
    )

    return q_final


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    export_df(result, file_name)
