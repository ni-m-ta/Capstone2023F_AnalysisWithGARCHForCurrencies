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

    # Calculate mean and standard deviation
    mean_price = data['Adj Close'].mean()
    std_dev_price = data['Adj Close'].std()

    with open("./Summaries/SummaryForDescriptiveStatistics/summary.txt", 'a') as file:
        # Write the string to the file
        file.write(f"{currency}\n")
        file.write(f"Mean Price: {mean_price}\n")
        file.write(f"Standard Deviation of Price: {std_dev_price}\n")