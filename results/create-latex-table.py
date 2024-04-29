import pandas as pd

# List of CSV files
csv_files = ['pawsx-results.csv', 'xnli-results.csv']

# Optional group by columns
# group_by_columns = ['test_name']  # Can be empty list for no grouping
group_by_columns = []
# values_with_significance = [('effect_size', 'p_value')]  # Example, adjust as needed
values_with_significance = []
skip_columns = ['date', 'is_debiased', 'p_value']  # Columns to skip in the LaTeX table
filter_column_values = {'epoch': 4}  # Filter rows based on column values

precision = 2  # Number of decimal places
significance_level = 0.05  # Significance level for marking

def format_number(x):
    try:
        return f"{x:.{precision}f}"
    except (ValueError, TypeError):
        return x

def load_and_concatenate(csv_files):
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['Source File'] = file  # Optionally track the source file
        all_data.append(df)
    
    # Concatenate all DataFrames, handling different columns
    concatenated_df = pd.concat(all_data, ignore_index=True, sort=False)
    return concatenated_df

def generate_latex_table(df, precision, filter_column_values=None, group_by_columns=[], values_with_significance=None, skip_columns=None):
    # Define columns to display
    display_columns = [col for col in df.columns if col not in skip_columns]

    # Filter rows based on column values
    if filter_column_values:
        for col, value in filter_column_values.items():
            df = df[df[col] == value]
        
    # Convert columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Start of the LaTeX table
    latex_code = "\\begin{tabular}{" + "l" * (len(display_columns)) + "}\n\\hline\n"
    
    # Column headers
    latex_code += " & ".join(display_columns) + " \\\\\n\\hline\n"
    
    # Handle grouping or direct iteration
    if group_by_columns:
        grouped = df.groupby(group_by_columns)
    else:
        grouped = [(None, df)]  # Treat entire DataFrame as a single group

    # Iterate over each group or the whole DataFrame
    for name, group in grouped:
        # Number of rows in the current group or entire DataFrame
        num_rows = len(group)
        
        for i, row in group.iterrows():
            row_items = []
            for col in display_columns:
                item = row[col]

                # Formatting and checking for significance
                if values_with_significance:
                    for value_col, significance_col in values_with_significance:
                        if col == value_col and row.get(significance_col, 1) < significance_level:
                            item = format_number(item)
                            item = f"{item}*"

                if col in group_by_columns and group_by_columns:
                    if i == group.first_valid_index():
                        row_items.append(f"\\multirow{{{num_rows}}}{{*}}{{{item}}}")
                    else:
                        row_items.append("")
                else:
                    row_items.append(str(item))

            latex_code += " & ".join(row_items) + " \\\\\n"
    
    # End of the table
    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    
    return latex_code

# Load and concatenate the data
concatenated_data = load_and_concatenate(csv_files)

# Generate LaTeX table
latex_table = generate_latex_table(concatenated_data, precision, group_by_columns, values_with_significance, skip_columns)
print(latex_table)
