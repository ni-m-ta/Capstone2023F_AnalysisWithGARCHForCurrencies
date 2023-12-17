import os
import numpy as np
import pandas as pd
import arch

def get_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

file_paths = get_files_in_folder("./PriceHistories")
for i in range(len(file_paths)):
    if "csv" not in file_paths[i]:
        continue
    currency = file_paths[i].split("/")[-1].split("_")[0]

    # Load the CSV file
    file_path = f"./PriceHistories/{currency}_USD.csv"
    data = pd.read_csv(file_path, parse_dates=True, index_col='Date')

    close_prices = data['Adj Close']

    # Calculate daily returns
    daily_returns = close_prices.pct_change()

    # Add the returns to the original DataFrame
    data['Daily returns'] = daily_returns
    # Drop missing values
    returns = data['Daily returns'].dropna()

    # Fit a GARCH(1,1) model
    garch_model = arch.arch_model(returns, vol='Garch', p=1, q=1)
    result = garch_model.fit()

    # Perform the ARCH-LM test
    lm_test = str(result.arch_lm_test(lags=15))

    # Specify the file path
    file_path = f"./Summaries/ARCH_LM/{currency}_USD.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the string to the file
        file.write(lm_test)
