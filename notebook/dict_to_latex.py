from config.config import encoding, config
import pandas as pd

def escape_latex(s):
    """Escape LaTeX special characters in a string."""
    return s.replace('_', r' ').replace('%', r'\%').replace('&', r'\&').replace('{', r'\{').replace('}', r'\}')

def dict_to_latex_table(d):
    rows = []
    for varname, content in d.items():
        definition = escape_latex(content['definition'])
        encodings = content['encoding']
        enc_list = [f"{v} = {escape_latex(k)}" for k, v in sorted(encodings.items(), key=lambda x: x[1])]
        enc_str = r' \\ '.join(enc_list)
        rows.append(f"{escape_latex(varname)} & {definition} & {enc_str} \\\\")

    body = '\n'.join(rows)
    table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{|p{3.5cm}|p{3.5cm}|p{7.5cm}|}\n"
        "\\hline\n"
        "\\textbf{Variable Name} & \\textbf{Definition} & \\textbf{Encoding (Value = Label)} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Variable definitions and their encoding values.}\n"
        "\\label{tab:variables_encoding}\n"
        "\\end{table}"
    )
    return table



def dict_to_dataframe(d):
    """
    Convert a dictionary of variable definitions and encodings to a clean pandas DataFrame.
    Columns: 'Variable Name', 'Definition', 'Encoding'
    """
    def clean_definition(s:str) -> str:
        return s.replace('_', ' ')
    records = []
    for varname, content in d.items():
        definition = clean_definition(content['definition'])
        encodings = content['encoding']
        enc_list = [f"{v} = {k}" for k, v in sorted(encodings.items(), key=lambda x: x[1])]
        enc_str = " | ".join(enc_list)
        records.append({
            'Variable Name': varname,
            'Definition': definition,
            'Encoding (Value = Label)': enc_str
        })

    df = pd.DataFrame(records)
    # Create a new 'Section' column based on the first word in the 'Definition' column
    df['Section'] = df['Variable Name'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')


    return df


def generate_latex_table(df):
    """
    Converts a DataFrame to a LaTeX table with good thesis formatting.
    Capitalizes the first letter of each cell in the 'Section' column.
    """

    def escape_latex(s):
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('{', r'\{').replace('}',
                                                                                                              r'\}')

    # Capitalize first letter of each section
    df['Section'] = df['Section'].str.capitalize()

    # Reorder columns
    df = df[['Section', 'Variable Name', 'Definition', 'Encoding (Value = Label)']]

    rows = []
    for _, row in df.iterrows():
        section = escape_latex(row['Section'])
        varname = escape_latex(row['Variable Name'])
        definition = escape_latex(row['Definition'])
        encoding = escape_latex(row['Encoding (Value = Label)']).replace('|', r'\\')
        rows.append(f"{section} & {varname} & {definition} & {encoding} \\\\")

    body = '\n'.join(rows)
    latex_table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{|p{2.2cm}|p{3.5cm}|p{3.5cm}|p{5.5cm}|}\n"
        "\\hline\n"
        "\\textbf{Section} & \\textbf{Variable Name} & \\textbf{Definition} & \\textbf{Encoding (Value = Label)} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Variables related to medical history (MH) and their encodings.}\n"
        "\\label{tab:mh_variables_encoding}\n"
        "\\end{table}"
    )
    return latex_table



if __name__ == '__main__':
    latex_code = dict_to_latex_table(d=encoding)
    df_encoding_table = dict_to_dataframe(d=encoding)

    print(latex_code)

    # now lets prase the editec latex table
    # Generate the LaTeX code from the current DataFrame
    excel_path = config.get('notebook').joinpath('df_encoding_table.xlsx')
    sheets = pd.read_excel(excel_path, sheet_name=None)  # Load all sheets as a dict of DataFrames

    # Apply the latex generation to each sheet
    latex_tables = {}
    for sheet_name, df_sheet in sheets.items():
        if sheet_name in ['df_encoding_table', 'dem']:
            continue
        df_sheet['Section'] = df_sheet['Section'].str.capitalize()  # Capitalize section
        latex_tables[sheet_name] = generate_latex_table(df_sheet)


