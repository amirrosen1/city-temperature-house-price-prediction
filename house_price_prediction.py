import os
from typing import NoReturn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X_pro = X.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)
    X_pro.dropna(inplace=True)
    y_pro = y.loc[X_pro.index]

    # Preprocess yr_renovated and yr_built columns
    X_pro["renovated"] = np.where(X_pro["yr_renovated"] >= np.percentile(X_pro['yr_renovated'].dropna().unique(),
                                                                         80), 1, 0)

    # Drop yr_renovated column
    X_pro.drop(["yr_renovated"], axis=1, inplace=True)

    # Create decade_built columns and drop yr_built column
    X_pro["decade_built"] = (X_pro["yr_built"] / 10).astype(int)
    X_pro.drop(["yr_built"], axis=1, inplace=True)
    X_pro = pd.get_dummies(X_pro, columns=["decade_built"], prefix=['decade_built'])

    # Remove outliers from bedrooms and sqft_lot columns
    X_pro = X_pro[(X_pro["bedrooms"] < 15) & (X_pro["sqft_lot"] < 1000000)]
    y_pro = y_pro.loc[X_pro.index].dropna()
    X_pro = X_pro.loc[y_pro.index]
    return X_pro, y_pro


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X_pro = X.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)
    X_pro.ffill(inplace=True)

    # Preprocess yr_renovated and yr_built columns
    if "yr_renovated" in X_pro.columns and np.any(X_pro['yr_renovated'].notna()):
        recent_renovation_threshold = np.percentile(X_pro.loc[X['yr_renovated'].notna(), 'yr_renovated'].unique(), 80)
        X_pro["renovated"] = np.where(X_pro["yr_renovated"] >= recent_renovation_threshold, 1, 0)
    else:
        X_pro["renovated"] = 0
    X_pro.drop(["yr_renovated"], axis=1, inplace=True)

    # Create decade_built columns and drop yr_built column
    X_pro["decade_built"] = (X_pro["yr_built"] / 10).astype(int) if "yr_built" in X.columns else 0
    X_pro.drop(["yr_built"], axis=1, inplace=True)
    X_pro = pd.get_dummies(X_pro, columns=["decade_built"], prefix=['decade_built'])
    return X_pro


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for column in X.columns:
        feature_data = X[column]
        covariance = np.cov(feature_data, y)[0, 1]
        std_x = np.std(feature_data, ddof=1)
        std_y = np.std(y, ddof=1)
        pearson_correlation = covariance / (std_x * std_y)

        plt.figure(figsize=(10, 6))
        plt.scatter(feature_data, y, alpha=0.5)
        plt.title(f'{column} vs. Price | Pearson Correlation: {pearson_correlation:.2f}')
        plt.xlabel(column)
        plt.ylabel('Price')

        plt.savefig(os.path.join(output_path, f"{column}_vs_price.png"))
        plt.close()


def plot_loss(mean_losses, std_losses, p_values):
    """
    Plot the mean squared error loss as a function of the training set size.

    Parameters
    ----------
    mean_losses : np.ndarray
        Mean loss values for each training set size percentage.
    std_losses : np.ndarray
        Standard deviation of loss values for each training set size percentage.
    p_values : list
        List of training set size percentages.
    """
    plt.figure(figsize=(10, 5))
    plt.errorbar(p_values, mean_losses, yerr=2 * std_losses, fmt='-o', capsize=5,
                 capthick=2, ecolor='gray', errorevery=1,
                 linewidth=2, label='Mean Loss ± 2 STD')
    plt.title('Mean Loss vs. Training Set Size')
    plt.xlabel('Percentage of Training Data Used (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def manual_train_test_split(X, y, test_size=0.25, random_state=None):
    """
    Manually split the data into training and testing sets.

    Parameters
    ----------
    X (pd.DataFrame): Input data to split.
    y (pd.Series): Responses of input data to split.
    test_size (float): Fraction of the data to reserve for testing.
    random_state (int): Random seed for reproducibility.

    Returns
    -------
    X_train (pd.DataFrame): Training data.
    X_test (pd.DataFrame): Testing data.
    y_train (pd.Series): Training responses.
    y_test (pd.Series): Testing responses.
    """
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    train_X, test_X, train_y, test_y = manual_train_test_split(X, y, test_size=0.25, random_state=42)

    # Question 3 - preprocessing of housing prices train dataset
    X_train_processed, y_train_processed = preprocess_train(train_X, train_y)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train_processed, y_train_processed, output_path="feature_plots")

    # Question 5 - preprocess the test data
    X_test_processed = preprocess_test(test_X)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p_values = range(10, 101)
    results = np.zeros((len(p_values), 10))

    for index, p in enumerate(p_values):
        for trial in range(10):
            sample_X = X_train_processed.sample(frac=p / 100, random_state=42 + trial)
            sample_y = y_train_processed.loc[sample_X.index]
            model = LinearRegression(include_intercept=True)
            model.fit(sample_X, sample_y)
            results[index, trial] = model.loss(X_test_processed, test_y)

    # Calculate mean and std of losses for each percentage
    mean_losses = np.mean(results, axis=1)
    std_losses = np.std(results, axis=1)

    plot_loss(mean_losses, std_losses, list(p_values))
