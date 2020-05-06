import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_histogram(df: pd.DataFrame, col: str, xlabel: str, ylabel: str, title: str, **kwargs):
    """
    take a dataframe and graph the histogram of the variable of interest
    :param df: dataframe
    :param col: variable of interest
    :param xlabel: title on x axis
    :param ylabel: title on y axis
    :param title: graph title
    :param kwargs:
    :return:
    """
    sns.distplot(df[col], **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return plt.show()


def plot_boxplot(df: pd.DataFrame, col: str, label: str = None):
    """
    take a dataframe and graph the boxplot of the variable of interest
    :param df: dataframe
    :param col: variable of interest
    :param label: target variable of predict
    :return:
    """
    if label is None:
        figure = sns.boxplot(data=df, x=col, linewidth=2.5)
    else:
        figure = sns.boxplot(data=df, x=label, y=col, linewidth=2.5)
    return figure
