import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt

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
currencies_list = [file.split("/")[-1].split("_")[0] for file in file_paths if "csv" in file]

# Test regression models for each currency based on the other currencies price histories
GARCH_results = pd.read_csv(file_path_for_GARCH_results)

for target_currency in currencies_list:
    # File paths for each results
    Breusch_Pagabn_test_file_path = "./OLS_Diagnostics/Breusch-Pagabn_test.txt"
    Durbin_Watson_test_file_path = "./OLS_Diagnostics/Durbin-Watson_test.txt"
    Multicollinearity_test_file_path = "./OLS_Diagnostics/Multicollinearity-test.txt"
    Cook_Distance_test_file_path = "./OLS_Diagnostics/Cook-Distance-test.txt"
    Shapiro_Wilk_test_file_path = "./OLS_Diagnostics/Shapiro-Wilk_test.txt"
    good_fits_test_file_path = f"./OLS_Diagnostics/Good_Fits/{target_currency}_USD.txt"
    folder_path  = f"./OLS_Diagnostics/OLS_Plots/Partial_Regression_Plot/{target_currency}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get the data (the price history, const, and conditional variance) from target currencies
    data = {target_currency: pd.read_csv(f"./PriceHistories/{target_currency}_USD.csv", parse_dates=True, index_col='Date')['Adj Close']}
    target_currency_row = GARCH_results[GARCH_results["Currency"] == target_currency]
    data["Const"] = target_currency_row["Const"].values[0]
    data["ConditionalVariance"] = target_currency_row["ConditionalVariance"].values[0]

    # Get the price histories for the other currencies
    for other_currency in currencies_list:
        if other_currency != target_currency:
            file_path_for_price_history = f"./PriceHistories/{other_currency}_USD.csv"
            price_history = pd.read_csv(file_path_for_price_history, parse_dates=True, index_col='Date')['Adj Close']
            data[other_currency] = price_history

    # Convert float values to Series
    data = {key: pd.Series(value) for key, value in data.items()}
    
    # Combine all price histories into a single DataFrame
    all_data = pd.concat(data.values(), axis=1, keys=data.keys())

    # Fill missing values with forward-fill method
    all_data = all_data.ffill()
    # Fill Const and ConditionalVariance with backward-fill method
    all_data = all_data.bfill()

    # Specify independent and dependent variables
    independent_variables_for_X = [currency for currency in currencies_list if currency != target_currency] + ['Const', 'ConditionalVariance']
    X = all_data[independent_variables_for_X]
    y = all_data[target_currency]

    # Now try fitting the OLS model
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # Residual Analysis
    residuals = model.resid

    # Scatter plot of residuals vs. fitted values
    plt.scatter(model.fittedvalues, residuals)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs. Fitted Values for {target_currency}_USD')
    plt.savefig(f"./OLS_Diagnostics/OLS_Plots/Residuals_vs_Fitted/{target_currency}_USD.png")
    plt.close()

    # Normal Q-Q plot
    sm.qqplot(residuals, line='s')
    plt.title(f'Normal Q-Q Plot for {target_currency}_USD')
    plt.savefig(f"./OLS_Diagnostics/OLS_Plots/Normal_QQ_Plot/{target_currency}_USD.png")
    plt.close()

    # Homoscedasticity: Breusch-Pagan test
    het_test = sms.het_breuschpagan(residuals, model.model.exog)
    print(f'p-value for Breusch-Pagan test for {target_currency}_USD: {het_test[1]}')
    with open(Breusch_Pagabn_test_file_path, "a") as result_file:
        result_file.write(f"{target_currency}_USD: {het_test[1]}\n")

    # Independence of Residuals: Durbin-Watson statistic
    durbin_watson_statistic = sm.stats.stattools.durbin_watson(residuals)
    print(f'Durbin-Watson Statistic for {target_currency}_USD: {durbin_watson_statistic}')
    with open(Durbin_Watson_test_file_path, "a") as result_file:
        result_file.write(f"{target_currency}_USD: {durbin_watson_statistic}\n")

    # Linearity: Partial regression plots
    for independent_var in independent_variables_for_X:
        exog_others = [var for var in independent_variables_for_X if var != independent_var]
        sm.graphics.plot_partregress(target_currency, independent_var, exog_others=exog_others, data=all_data, obs_labels=False)
        plt.savefig(f"./OLS_Diagnostics/OLS_Plots/Partial_Regression_Plot/{target_currency}/{independent_var}.png")
        plt.close()


    # Multicollinearity: VIF
    vif_data = pd.DataFrame({'Variable': independent_variables_for_X, 'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    print(f'\nVIF for {target_currency}_USD:\n{vif_data}')
    with open(Multicollinearity_test_file_path, "a") as result_file:
        result_file.write(f"{target_currency}_USD:\n{vif_data}\n")

    # Outliers and Influential Observations: Cook's Distance
    infl = model.get_influence()
    cook_distance = infl.cooks_distance
    print(f'\nCook\'s Distance for {target_currency}_USD:\n{cook_distance}')
    with open(Cook_Distance_test_file_path, "a") as result_file:
        result_file.write(f"{target_currency}_USD:\n{cook_distance}\n")

    # Leverage-residual square plot
    sm.graphics.influence_plot(model, criterion='cooks')
    plt.savefig(f"./OLS_Diagnostics/OLS_Plots/Leverage_Residual_Plot/{target_currency}_USD.png")
    plt.close()

    # Goodness of Fit
    print(f'\nGoodness of Fit for {target_currency}_USD:')
    print(f'R-squared: {model.rsquared}')
    print(f'Adjusted R-squared: {model.rsquared_adj}')
    print(f'F-statistic: {model.fvalue}')
    with open(good_fits_test_file_path, "a") as result_file:
        result_file.write(f'R-squared: {model.rsquared}\n')
        result_file.write(f'Adjusted R-squared: {model.rsquared_adj}\n')
        result_file.write(f'F-statistic: {model.fvalue}\n')

    # Normality of Residuals: Shapiro-Wilk test
    shapiro_test = stats.shapiro(residuals)
    print(f'\nShapiro-Wilk test for normality of residuals for {target_currency}_USD:\n{shapiro_test}')
    with open(Shapiro_Wilk_test_file_path, "a") as result_file:
        result_file.write(f"{target_currency}_USD:\n{shapiro_test}\n")

    # Histogram and kernel density plot
    plt.hist(residuals, bins='auto', density=True, alpha=0.75)
    plt.title(f'Residuals Histogram for {target_currency}_USD')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.savefig(f"./OLS_Diagnostics/OLS_Plots/Residuals_Histogram/{target_currency}_USD.png")
    plt.close()
