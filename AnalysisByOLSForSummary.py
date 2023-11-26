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

file_path_for_GARCH_results = "./ResultsFromGARCH(1,1).csv"
GARCH_results = pd.read_csv(file_path_for_GARCH_results)
target_currency_row = GARCH_results[GARCH_results["Currency"] == "JPY"]

file_paths = get_files_in_folder("./PriceHistories")
for i in range(len(file_paths)):
    if "csv" not in file_paths[i]:
        continue
    currency = file_paths[i].split("/")[-1].split("_")[0]

    # Load the CSV file
    file_path_for_price_history = f"./PriceHistories/{currency}_USD.csv"
    data = pd.read_csv(file_path_for_price_history, parse_dates=True, index_col='Date')
    
    # Load ResultsFromGARCH(1,1).csv
    target_currency_row = GARCH_results[GARCH_results["Currency"] == currency]

    # Create a constant term
    data["Const"] = target_currency_row["Const"].values[0]
    data["ConditionalVariance"] = target_currency_row["ConditionalVariance"].values[0]

    # Specify independent and dependent variables
    X = data[['Const','ConditionalVariance']]
    y = data['Adj Close']

    # Fit the OLS regression model
    model = sm.OLS(y, X).fit()
    
    # Get the model summary
    summary = str(model.summary())

    # Specify the file path
    file_path = f"./SummariesForOLS/{currency}_USD.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the string to the file
        file.write(summary)
