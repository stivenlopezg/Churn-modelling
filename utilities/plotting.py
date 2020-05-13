import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_histogram(df: pd.DataFrame, col: str,
                   xlabel: str = None, ylabel: str = None, title: str = None, **kwargs):
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
    plt.figure(figsize=(16, 10))
    if label is None:
        return sns.boxplot(data=df, y=col, linewidth=2.5, palette='Blues')
    else:
        return sns.boxplot(data=df, x=label, y=col, linewidth=2.5, palette='Blues')


def univariate_graph(df: pd.DataFrame, col: str):
    """
    Take a dataframe and graph the histogram and boxplot of the variable of interest
    :param df: dataframe
    :param col: variable of interest
    :return:
    """
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    sns.distplot(df[col])
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y=col)


def barplot(df: pd.DataFrame, col: str):
    """
    Take a dataframe and graph the barchart of the variable of interest
    :param df:
    :param col:
    :return:
    """
    aux = df[col].value_counts().sort_values(ascending=True)
    bars = tuple(aux.index.tolist())
    values = aux.values.tolist()
    y_pos = np.arange(len(bars))
    colors = ['lightblue'] * len(bars)
    colors[-1] = 'blue'
    plt.figure(figsize=(16, 10))
    plt.barh(y_pos, values, color=colors)
    plt.title(f'{col} bar chart')
    plt.yticks(y_pos, bars)
    return plt.show()

