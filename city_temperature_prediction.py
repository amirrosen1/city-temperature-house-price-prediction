import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    # Load data and preprocess it by removing NaN values and duplicates and filtering out invalid temperatures and dates
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df = df[(df['Temp'] > -45) & (df['Temp'] < 45)]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Year'] = df['Year'].astype(str)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    df = df[df['Day'] >= 1]
    df = df[df['Day'] <= 31]
    df = df[df['Month'] >= 1]
    df = df[df['Month'] <= 12]
    return df


def plot_scatter_by_year(df, output_path):
    """
    Plots a scatter plot of daily average temperature vs day of the year,
    colored by year, and saves the plot to the specified output path.

    Parameters:
    df (pd.DataFrame): DataFrame containing temperature data.
    output_path (str): Path to save the scatter plot image.
    """
    plt.figure(figsize=(10, 6))
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        plt.scatter(year_data['DayOfYear'], year_data['Temp'], label=year, alpha=0.5)
    plt.xlabel('Day of the Year')
    plt.ylabel('Temperature (°C)')
    plt.title("Daily Average Temperature vs Day of the Year in Israel")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()


def calculate_monthly_stats(df):
    """
    Calculate average temperature and standard deviation of daily temperatures by month and country.
    Parameters
    ----------
    df (pd.DataFrame) containing temperature data.

    Returns
    -------
    monthly_stats (pd.DataFrame): DataFrame containing average temperature and standard deviation of daily temperatures.
    """

    # Group by 'Country' and 'Month' and calculate statistics
    monthly_stats = df.groupby(['Country', 'Month']).agg(
        AverageTemp=('Temp', 'mean'),
        TempStdDev=('Temp', 'std')
    ).reset_index()
    return monthly_stats


def plot_monthly_temp_std_dev(df, output_path):
    """
    Plots a bar plot of the standard deviation of daily temperatures by month,
    and saves the plot to the specified output path.

    Parameters:
    df (pd.DataFrame): DataFrame containing temperature data.
    output_path (str): Path to save the bar plot image.
    """
    monthly_stats = df.groupby('Month').agg(AverageTemp=('Temp', 'mean'),
                                            TempStdDev=('Temp', 'std')).reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(monthly_stats['Month'], monthly_stats['TempStdDev'])
    plt.xlabel('Month')
    plt.ylabel('Temperature Standard Deviation (°C)')
    plt.title("Standard Deviation of Daily Temperatures by Month in Israel")
    plt.savefig(output_path)
    plt.clf()


def plot_monthly_temperature_stats(monthly_stats):
    """
    Plots a line plot of the average temperature by month for each country, with error bars representing the standard
     deviation.
    Parameters
    ----------
    monthly_stats (pd.DataFrame): DataFrame containing average temperature and standard deviation of daily temperatures.
    """
    plt.figure(figsize=(10, 6))
    for country in monthly_stats['Country'].unique():
        country_data = monthly_stats[monthly_stats['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['AverageTemp'], yerr=country_data['TempStdDev'],
                     label=country, fmt='-o')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.title('Average Monthly Temperature by Country')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("average_monthly_temperatures.png")
    plt.clf()


def manual_train_test_split(X, y, test_size=0.25, random_state=None):
    """
    Manually split the data into training and testing sets.
    Parameters
    ----------
    X (np.ndarray): Input data to split.
    y (np.ndarray): Responses of input data to split.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns
    -------
    X_train (np.ndarray): Training data.
    X_test (np.ndarray): Testing data.
    y_train (np.ndarray): Training responses.
    y_test (np.ndarray): Testing responses.
    """
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    df_israel_q3 = df[df['Country'] == 'Israel']
    scatter_output_path = "scatter_daily_temp_israel.png"
    bar_output_path = "monthly_temp_std_dev_israel.png"
    plot_scatter_by_year(df_israel_q3, scatter_output_path)
    plot_monthly_temp_std_dev(df_israel_q3, bar_output_path)

    # Question 4 - Exploring differences between countries
    monthly_stats_q4 = calculate_monthly_stats(df)
    plot_monthly_temperature_stats(monthly_stats_q4)

    # Question 5 - Fitting model for different values of `k`
    df_israel_q5 = df[df['Country'] == 'Israel']
    X = df_israel_q5['DayOfYear'].values
    y = df_israel_q5['Temp'].values
    train_X, test_X, train_y, test_y = manual_train_test_split(X, y, test_size=0.25, random_state=0)
    loss_lst = []
    for k in range(1, 11):
        model = PolynomialFitting(k=k)
        model.fit(train_X, train_y)
        loss = round(model.loss(test_X, test_y), 2)
        loss_lst.append(loss)
        print(f'k={k}, Loss error={loss}')

    losses_series = pd.Series(loss_lst)
    k_degrees = pd.Series(range(1, 11))
    plt.bar(k_degrees, losses_series)
    plt.xlabel('Degree of Polynomial (k)')
    plt.ylabel('Loss Error')
    plt.title('Loss Error for Different Values of k')
    for i, loss in enumerate(loss_lst):
        plt.text(i + 1, loss + 0.5, str(loss), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('loss_error_vs_k.png')
    plt.clf()

    # Question 6 - Evaluating fitted model on different countries
    k_chosen = 5
    pf_israel = PolynomialFitting(k_chosen)
    pf_israel.fit(X, y)

    countries = df['Country'].unique()
    countries = countries[countries != "Israel"]

    country_losses = {}

    for country in countries:
        df_country = df[df['Country'] == country]
        X_country = df_country['DayOfYear'].values
        y_country = df_country['Temp'].values
        loss = pf_israel.loss(X_country, y_country)
        country_losses[country] = round(loss, 2)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20.colors
    bar_colors = [colors[i % len(colors)] for i in range(len(country_losses))]
    plt.bar(country_losses.keys(), country_losses.values(), color=bar_colors)
    plt.title("Model's Error on Different Countries")
    plt.xlabel("Country")
    plt.ylabel("Loss")

    for i, (country, loss) in enumerate(country_losses.items()):
        plt.text(i, loss + 0.5, str(loss), ha='center', va='bottom')

    # Add custom legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)]) for i in
                      range(len(country_losses))]
    plt.legend(legend_handles, country_losses.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("model_error_different_countries.png")
    plt.clf()
