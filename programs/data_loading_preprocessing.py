import pandas as pd
import numpy as np


def load_data(file_path, sheet_name=None, n_rows=None, dtype=None):
    """
    This function loads data from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.
    sheet_name (str or list or None): The name of the sheet(s) to read. If None, all sheets are read. Default is None.
    n_rows (int or None): The number of rows to read. If None, all rows are read. Default is None.
    dtype (dict or None): The data type for each column. If None, the data type is inferred. Default is None.

    Returns:
    pd.DataFrame: The loaded data.
    """
    if sheet_name is None:
        data = pd.concat(pd.read_excel(file_path, sheet_name=None, nrows=n_rows), ignore_index=True)
    else:
        data = pd.read_excel(file_path, sheet_name=sheet_name, nrows=n_rows, dtype=dtype)
    data['Код'] = data['Код'].astype(int)  # Set the data type of the 'Код' column to int
    return data


def process_data(data, n_rows=None):
    """
    This function processes the loaded data.

    Parameters:
    data (pd.DataFrame): The data to be processed.
    n_rows (int or None): The number of rows to keep. If None, all rows are kept. Default is None.

    Returns:
    pd.DataFrame: The processed data.
    """
    data = data.copy()  # Make a copy of the data to avoid SettingWithCopyWarning
    data = data.iloc[:n_rows] if n_rows is not None else data
    data = data.dropna()
    data['Описание'] = data['Описание'].astype(str)
    return data


if __name__ == '__main__':
    file_path = 'C:/Users/Arina/SGDclf_KNclf/pythonProject/data/Base_model_data_500000.xlsx'
    data500000 = load_data(file_path)
    data500000 = process_data(data500000)
    data300000 = process_data(data500000, n_rows=300001)
    print(data500000)
