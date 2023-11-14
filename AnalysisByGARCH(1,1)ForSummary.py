import os

import numpy as np
import pandas as pd

from arch import arch_model

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

    # Extract the log returns
    returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()

    # Fit GARCH(1,1) model
    model = arch_model(returns*10, vol='Garch', p=1, q=1)
    results = model.fit()

    # Display the model summary
    summary = str(results.summary())

    # Specify the file path
    file_path = f"./Summaries/{currency}_USD.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the string to the file
        file.write(summary)