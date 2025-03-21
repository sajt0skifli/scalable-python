import pandas as pd
import os
from datetime import timedelta

# Define the path to read data and write output
input_path = "../TPC-H/tables/lineitem.tbl"
output_path = "../TPC-H/output/q1.out"

# Define the parameter (in the SQL it's ':1')
param1 = 90  # This is the interval day parameter

# Define column names based on the schema
column_names = [
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

# Read the data with explicit data types for each column
lineitem = pd.read_csv(
    input_path,
    sep="|",
    header=None,
    names=column_names,
    usecols=range(len(column_names)),  # Only use the columns we defined
    dtype={
        "l_returnflag": str,
        "l_linestatus": str,
        "l_shipinstruct": str,
        "l_shipmode": str,
        "l_comment": str,
    },
    engine="python",  # Python engine handles trailing delimiters better
)

# Convert numeric columns properly
lineitem["l_quantity"] = pd.to_numeric(lineitem["l_quantity"])
lineitem["l_extendedprice"] = pd.to_numeric(lineitem["l_extendedprice"])
lineitem["l_discount"] = pd.to_numeric(lineitem["l_discount"])
lineitem["l_tax"] = pd.to_numeric(lineitem["l_tax"])

# Convert date columns
lineitem["l_shipdate"] = pd.to_datetime(lineitem["l_shipdate"])
lineitem["l_commitdate"] = pd.to_datetime(lineitem["l_commitdate"])
lineitem["l_receiptdate"] = pd.to_datetime(lineitem["l_receiptdate"])

# Calculate the threshold date (1998-12-01 - interval ':1' day)
threshold_date = pd.to_datetime("1998-12-01") - timedelta(days=param1)

# Filter rows where l_shipdate <= threshold_date
filtered_df = lineitem[lineitem["l_shipdate"] <= threshold_date].copy()

# Calculate the derived columns according to the SQL query
filtered_df["sum_disc_price"] = filtered_df["l_extendedprice"] * (
    1 - filtered_df["l_discount"]
)
filtered_df["sum_charge"] = (
    filtered_df["l_extendedprice"]
    * (1 - filtered_df["l_discount"])
    * (1 + filtered_df["l_tax"])
)

# Group by and calculate aggregates
result = (
    filtered_df.groupby(["l_returnflag", "l_linestatus"])
    .agg(
        sum_qty=("l_quantity", "sum"),
        sum_base_price=("l_extendedprice", "sum"),
        sum_disc_price=("sum_disc_price", "sum"),
        sum_charge=("sum_charge", "sum"),
        avg_qty=("l_quantity", "mean"),
        avg_price=("l_extendedprice", "mean"),
        avg_disc=("l_discount", "mean"),
        count_order=("l_orderkey", "count"),
    )
    .reset_index()
)

# Sort the result as specified in the SQL query
result = result.sort_values(["l_returnflag", "l_linestatus"])

# Format the output to match the expected format in q1.out
result["sum_qty"] = result["sum_qty"].apply(lambda x: f"{x:.2f}")
result["sum_base_price"] = result["sum_base_price"].apply(lambda x: f"{x:.2f}")
result["sum_disc_price"] = result["sum_disc_price"].apply(lambda x: f"{x:.2f}")
result["sum_charge"] = result["sum_charge"].apply(lambda x: f"{x:.2f}")
result["avg_qty"] = result["avg_qty"].apply(lambda x: f"{x:.2f}")
result["avg_price"] = result["avg_price"].apply(lambda x: f"{x:.2f}")
result["avg_disc"] = result["avg_disc"].apply(lambda x: f"{x:.2f}")
result["count_order"] = result["count_order"].apply(lambda x: f"{x:20d}")

# Create the output string in the expected format
header = "l|l|sum_qty                                  |sum_base_price                           |sum_disc_price                           |sum_charge                               |avg_qty                                  |avg_price                                |avg_disc                                 |count_order           "
rows = []
for _, row in result.iterrows():
    row_str = f"{row['l_returnflag']}|{row['l_linestatus']}|{row['sum_qty']}|{row['sum_base_price']}|{row['sum_disc_price']}|{row['sum_charge']}|{row['avg_qty']}|{row['avg_price']}|{row['avg_disc']}|{row['count_order']}"
    rows.append(row_str)

# Write to output file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(header + "\n")
    for row in rows:
        f.write(row + "\n")

print(f"Query results written to {output_path}")
