import os
import pandas as pd
import statsmodels.api as sm

def get_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# Get file paths from a directory for price histories
file_path_for_GARCH_results = "./ResultsFromGARCH(1,1).csv"
file_paths = get_files_in_folder("./PriceHistories")

# Get the currencies to use for the regression tests
currencies_list = []
for i in range(len(file_paths)):
    if "csv" not in file_paths[i]:
        continue
    currencies_list.append(file_paths[i].split("/")[-1].split("_")[0])

# Test regression models for each currency based on the other currencies price histories
GARCH_results = pd.read_csv(file_path_for_GARCH_results)
for i in range(len(currencies_list)):
    # Get the data (the price history, const, and conditional variance) from target currencies
    data = {}
    target_currency = currencies_list[i]
    file_path_for_price_history = f"./PriceHistories/{target_currency}_USD.csv"
    data[target_currency] = pd.read_csv(file_path_for_price_history, parse_dates=True, index_col='Date')['Adj Close']
    target_currency_row = GARCH_results[GARCH_results["Currency"] == target_currency]
    data["Const"] = target_currency_row["Const"].values[0]
    data["ConditionalVariance"] = target_currency_row["ConditionalVariance"].values[0]

    # Get the price histories for the other currencies
    for j in range(len(currencies_list)):
        if currencies_list[j] != target_currency:
            file_path_for_price_history = f"./PriceHistories/{currencies_list[j]}_USD.csv"
            price_history = pd.read_csv(file_path_for_price_history, parse_dates=True, index_col='Date')['Adj Close']
            data[currencies_list[j]] = price_history

    # Convert float values to Series
    data = {key: pd.Series(value) for key, value in data.items()}
    
    # Combine all price histories into a single DataFrame
    all_data = pd.concat(data.values(), axis=1, keys=data.keys())

    # Fill missing values for day off's prices in fiat currencies with forward-fill method
    all_data = all_data.ffill()
    # Fill Const and ConditionalVariance with backward-fill method
    all_data = all_data.bfill()

    # Specify independent and dependent variables
    independent_variables_for_X = [currency for currency in currencies_list if currency != target_currency] + ['Const', 'ConditionalVariance']
    X = all_data[independent_variables_for_X]
    y = all_data[target_currency]

    # Now try fitting the OLS model again
    model = sm.OLS(y, X).fit()

    # Get the model summary
    summary = str(model.summary())

    # Specify the file path
    file_path = f"./SummariesForOLS/{target_currency}_USD.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the string to the file
        file.write(summary)
