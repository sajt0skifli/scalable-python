import pandas as pd

ROOTPATH = "../TPC-H/tables"


def get_table_path(table_name: str) -> str:
    """Get the path to the table based on the table name."""
    return f"{ROOTPATH}/{table_name}.tbl"


def _read_ds(
    table_name: str, col_names: list = None, dtypes: dict = None, date_cols: list = None
) -> pd.DataFrame:
    path = get_table_path(table_name)

    # Set up read_csv parameters with an extra column for the trailing delimiter
    params = {"sep": "|", "header": None, "dtype_backend": "pyarrow"}

    # Add an extra column for the trailing delimiter
    if col_names:
        params["names"] = col_names + ["dummy"]

    if dtypes:
        # Only apply dtypes to the real columns
        params["dtype"] = dtypes

    # Parse dates during reading
    if date_cols:
        params["parse_dates"] = date_cols

    df = pd.read_csv(path, **params)

    # Remove the dummy column that's created due to the trailing delimiter
    if "dummy" in df.columns:
        df = df.drop(columns=["dummy"])
    elif df.columns.size > 0 and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]

    # Convert parsed dates to date32 type
    if date_cols:
        for col in date_cols:
            df[col] = df[col].astype("date32[day][pyarrow]")

    return df


def get_customer_ds() -> pd.DataFrame:
    cols = [
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ]
    dtypes = {
        "c_custkey": int,
        "c_name": str,
        "c_address": str,
        "c_nationkey": int,
        "c_phone": str,
        "c_acctbal": float,
        "c_mktsegment": str,
        "c_comment": str,
    }
    return _read_ds("customer", cols, dtypes)


def get_line_item_ds() -> pd.DataFrame:
    cols = [
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    ]
    dtypes = {
        "l_orderkey": int,
        "l_partkey": int,
        "l_suppkey": int,
        "l_linenumber": int,
        "l_quantity": float,
        "l_extendedprice": float,
        "l_discount": float,
        "l_tax": float,
        "l_returnflag": str,
        "l_linestatus": str,
        "l_shipinstruct": str,
        "l_shipmode": str,
        "l_comment": str,
    }
    date_cols = ["l_shipdate", "l_commitdate", "l_receiptdate"]

    return _read_ds("lineitem", cols, dtypes, date_cols)


def get_nation_ds() -> pd.DataFrame:
    cols = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]
    dtypes = {"n_nationkey": int, "n_name": str, "n_regionkey": int, "n_comment": str}
    return _read_ds("nation", cols, dtypes)


def get_orders_ds() -> pd.DataFrame:
    cols = [
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ]
    dtypes = {
        "o_orderkey": int,
        "o_custkey": int,
        "o_orderstatus": str,
        "o_totalprice": float,
        "o_orderpriority": str,
        "o_clerk": str,
        "o_shippriority": int,
        "o_comment": str,
    }
    date_cols = ["o_orderdate"]

    return _read_ds("orders", cols, dtypes, date_cols)


def get_part_ds() -> pd.DataFrame:
    cols = [
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ]
    dtypes = {
        "p_partkey": int,
        "p_name": str,
        "p_mfgr": str,
        "p_brand": str,
        "p_type": str,
        "p_size": int,
        "p_container": str,
        "p_retailprice": float,
        "p_comment": str,
    }
    return _read_ds("part", cols, dtypes)


def get_part_supp_ds() -> pd.DataFrame:
    cols = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]
    dtypes = {
        "ps_partkey": int,
        "ps_suppkey": int,
        "ps_availqty": int,
        "ps_supplycost": float,
        "ps_comment": str,
    }
    return _read_ds("partsupp", cols, dtypes)


def get_region_ds() -> pd.DataFrame:
    cols = ["r_regionkey", "r_name", "r_comment"]
    dtypes = {"r_regionkey": int, "r_name": str, "r_comment": str}
    df = _read_ds("region", cols, dtypes)
    return df


def get_supplier_ds() -> pd.DataFrame:
    cols = [
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ]
    dtypes = {
        "s_suppkey": int,
        "s_name": str,
        "s_address": str,
        "s_nationkey": int,
        "s_phone": str,
        "s_acctbal": float,
        "s_comment": str,
    }
    return _read_ds("supplier", cols, dtypes)


def export_df(df: pd.DataFrame, output_file: str) -> None:
    # Default column width (adjust as needed)
    str_col_width = 25
    num_col_width = 10

    with open(output_file, "w") as f:
        # Write header
        header = "|".join(str(col).ljust(str_col_width) for col in df.columns)
        f.write(header + "\n")

        # Write data rows
        for _, row in df.iterrows():
            # Format each value with appropriate justification
            formatted_values = []
            for val in row:
                if isinstance(val, (int, float)):
                    # Right-justify numbers
                    formatted_values.append(str(val).rjust(num_col_width))
                else:
                    # Left-justify strings
                    formatted_values.append(str(val).ljust(str_col_width))

            formatted_row = "|".join(formatted_values)
            f.write(formatted_row + "\n")
