import pandas as pd

# Load the CSV file
file_path = "invoice_matches_report (6).csv"
df = pd.read_csv(file_path, header=None)

# Extract the column names from the first row
column_names = df.iloc[0].tolist()
df = df[1:]  # Remove the first row as it's now used for column names

# Reset the index and rename columns
df.columns = column_names
df.reset_index(drop=True, inplace=True)

# Transpose the data
df_transposed = df.T

# Save the transformed data to a new CSV file
output_file_path = "transposed_invoice_matches_report.csv"
df_transposed.to_csv(output_file_path, header=False)

output_file_path
