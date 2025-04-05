import tpch.utils as utils

from dask import dataframe as dd

Q_NUM = 2


def query() -> dd.DataFrame:
    region_ds = utils.get_region_ds("dask")
    nation_ds = utils.get_nation_ds("dask")
    supplier_ds = utils.get_supplier_ds("dask")
    part_ds = utils.get_part_ds("dask")
    part_supp_ds = utils.get_part_supp_ds("dask")

    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    jn = (
        part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .merge(region_ds, left_on="n_regionkey", right_on="r_regionkey")
    )

    jn = jn[jn["p_size"] == var1]
    jn = jn[jn["p_type"].str.endswith(var2)]
    jn = jn[jn["r_name"] == var3]

    gb = jn.groupby("p_partkey")
    agg = gb["ps_supplycost"].min().reset_index()
    jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

    sel = jn2.loc[
        :,
        [
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        ],
    ]

    sort = sel.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    )
    result_df = sort.head(100)

    return result_df  # type: ignore[no-any-return]


if __name__ == "__main__":
    result = query()

    file_name = "q" + str(Q_NUM) + ".out"
    utils.export_df(result, file_name)
