import pyperf
import cudf

from tpch.utils import get_part_supp_ds, get_supplier_ds, get_nation_ds, export_df

Q_NUM = 11


def get_ds():
    partsupp = get_part_supp_ds("cudf")
    supplier = get_supplier_ds("cudf")
    nation = get_nation_ds("cudf")

    return partsupp, supplier, nation


def query():
    partsupp, supplier, nation = get_ds()

    var1 = "GERMANY"
    # https://www.tpc.org/tpc_documents_current_versions/pdf/tpc-h_v3.0.1.pdf
    # FRACTION is chosen as 0.0001 / SF
    # S_SUPPKEY identifier SF*10,000 are populated
    sumres_rate = 0.0001 / (supplier.shape[0] / 10000)

    # Filter nation first - this is typically the smallest table and will reduce joins
    german_nation = nation[nation["n_name"] == var1][["n_nationkey"]]

    # Join supplier with filtered nation - this creates a much smaller intermediate table
    german_suppliers = supplier[["s_suppkey", "s_nationkey"]].merge(
        german_nation, left_on="s_nationkey", right_on="n_nationkey"
    )[["s_suppkey"]]

    # Pre-calculate value in partsupp before the join to reduce computation after join
    partsupp["value"] = partsupp["ps_supplycost"] * partsupp["ps_availqty"]

    # Only keep the partsupp rows that match German suppliers
    # This significantly reduces the data size for aggregation
    filtered_partsupp = partsupp[["ps_partkey", "ps_suppkey", "value"]].merge(
        german_suppliers, left_on="ps_suppkey", right_on="s_suppkey"
    )

    # Group by partkey and sum values
    grouped = filtered_partsupp.groupby("ps_partkey", as_index=False).agg(
        {"value": "sum"}
    )

    # Calculate the threshold using the sum of values
    sum_value = grouped["value"].sum()
    threshold = sum_value * sumres_rate

    # Apply threshold filter and sort
    q_final = (
        grouped[grouped["value"] > threshold]
        .sort_values(by=["value", "ps_partkey"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return q_final


def bench_q11():
    t0 = pyperf.perf_counter()
    query()
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(
        quiet=False, loops=1, values=1, processes=1, warmups=0
    )
    runner.bench_func("cudf-q11", bench_q11)
    # result = query()
    #
    # file_name = "q" + str(Q_NUM) + ".out"
    # export_df(result, file_name, is_cudf=True)
