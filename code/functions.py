# -*- coding: utf-8 -*-
"""
Created on 2020.07.20

@author: jct
"""
import sys
import os
import glob
import re 
import pandas as pd
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt
import requests
import time
from collections import Counter
import seaborn as sns
from scipy import stats

sys.path.append(os.path.abspath("./Dropbox/MTB/Göttingen/research/"))

def calculate_regression_line(df, drop_nan = True):
    
    results = []

    for column in df.columns.tolist():

        if drop_nan == True:
            series = df[column].dropna()

        slope, intercept, rvalue, pvalue, stderr = stats.linregress(series.index, series.values, )

        results.append([column, slope, intercept, rvalue, pvalue, stderr])
    
    results_df = pd.DataFrame(results, columns = ["column", "slope", "intercept", "rvalue", "pvalue", "stderr"]).sort_values(by="slope", ascending=False)

    results_df.index = results_df.column

    return results_df


import scipy.stats as stats

def add_significance(df, class_pvalue = "test_result_pvalue"):
    df["significance"] = ""
    df.loc[df[class_pvalue] < 0.05,"significance"] = "*"
    df.loc[df[class_pvalue] < 0.01,"significance"] = "**"
    df.loc[df[class_pvalue] < 0.001,"significance"] = "***"
    return df
    
def test_differences_columns(df, column_class, column_value, equal_var=False):
    results_lt = []
    seen_values = []
    for value1 in sorted(list(set(df[column_class]))):
        for value2 in sorted(list(set(df[column_class]))):
            if value2 not in seen_values and value1 != value2:

                statistic, pvalue = stats.ttest_ind(
                            df.loc[df[column_class]==value1][column_value],
                            df.loc[df[column_class]==value2][column_value],
                    equal_var = equal_var
                            )
                seen_values.append(value1)
                results_lt.append([value1, value2, pvalue, statistic, df.loc[df[column_class]==value1][column_value].mean(), df.loc[df[column_class]==value2][column_value].mean(), df.loc[df[column_class]==value1][column_value].median(), df.loc[df[column_class]==value2][column_value].median()])
    results_df = pd.DataFrame(results_lt, columns=["value1","value2","pvalue","statistic", "mean_value_1", "mean_value_2", "median_value_1", "median_value_2"])
    results_df = add_significance(results_df, class_pvalue = "pvalue")
    
    return results_df


color = "#669999"
cmap_20 = "tab20_r"
cmap_8 = "Dark2"


def plot_boxplots_by(df, column_to_plot, column_by, figsize = (20,5), color = color, xlabel = "", ylabel = "", title = "", outdir = "./../visualizations/", figure_name = "boxplot_xlabel_by_ylabel", rot = 0):
    """
        plot_boxplots_by(df.sample(10000, random_state=2021), column_to_plot = "entry_first_date_year", column_by = "year_publication", xlabel= "year of publication", ylabel= "year of first record in K10plus",
        title = "Boxplot of year of first entry by year of publication",
        figure_name = "boxplot_year_publication_year_entry_catalog")
    """

    boxprops = dict(linestyle='-', linewidth=2, )
    medianprops = dict(linestyle='-', linewidth=2)
    whiskerprops = dict(linestyle='-',linewidth=2)    

    ax = df.boxplot(column_to_plot, by = column_by, figsize = figsize, color=color, boxprops=boxprops,
                medianprops=medianprops, whiskerprops = whiskerprops, rot = rot)
    ax.set_title("")
    ax.set_axisbelow(True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)

    fig = ax.get_figure()

    plt.tight_layout()

    fig.suptitle("")


    fig.savefig(outdir + figure_name + ".png")


"""
def plot_boxplots_by(df, column_to_plot, column_by, figsize = (20,5), color = color, xlabel = "", ylabel = "", title = "", outdir = "./../visualizations/", figure_name = "boxplot_xlabel_by_ylabel", rot = 0):

    boxprops = dict(linestyle='-', linewidth=2, )
    medianprops = dict(linestyle='-', linewidth=2)
    whiskerprops = dict(linestyle='-',linewidth=2)    

    ax = df.boxplot(column_to_plot, by = column_by, figsize = figsize, color=color, boxprops=boxprops,
                medianprops=medianprops, whiskerprops = whiskerprops, rot = rot)
    ax.set_title("")
    ax.set_axisbelow(True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


    fig = ax.get_figure()

    plt.tight_layout()

    fig.suptitle(title)


    fig.savefig(outdir + figure_name + ".png")
"""

from pylab import *

cmap = cm.get_cmap('tab20', 20)    # PiYG

colors = []
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    colors.append(matplotlib.colors.rgb2hex(rgba))



def plot_scatter_w_regression_line(df, column_x, column_y, figsize = (20,5), color = "purple", xlabel = "", ylabel = "", title = "Title", outdir = "./../visualizations/", figure_name = "plot_scatter_w_regression_line", rot = 0):
    
    ax = sns.regplot(x = column_x, y = column_y, data = df, color= color)


    slope, intercept, r_value, p_value, std_err = stats.linregress(df[column_x], df[column_y])
    print(p_value, slope)


    ax.set_title("")
    ax.set_axisbelow(True)
    #ax.set_title(title)

    if xlabel != "":
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(column_x)

    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(column_y)


    fig = ax.get_figure()
    fig.suptitle(title + "\np-value = " + str(round(p_value, 4)) + ", slope = " + str(round(slope, 4)))
    fig.savefig(outdir + figure_name + ".png")
    return p_value, slope




def get_colors_lt(cmap_name = "tab20"):

    cmap = cm.get_cmap(cmap_name, 20)    # PiYG

    colors = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        colors.append(matplotlib.colors.rgb2hex(rgba))

    return colors
        
def add_color_column_based_on_str_column(df, colors = colors, column_name = "main writing system"):

    colors = colors[0:len(df[column_name].unique().tolist())]
    color_dict = dict(zip(df[column_name].unique().tolist(), colors))
    
    df[column_name + "_colors"]= [ color_dict[i] for i in df[column_name] ]

    return df, color_dict


def plot_scatter_w_colors(df, column_x, column_y, column_color, figsize = (10,6), palette = "tab20", xlabel = "", ylabel = "", title = "Title", outdir = "./../visualizations/", figure_name = "plot_scatter_w_colors"):
    """
    functions.plot_scatter_w_colors(metadata.loc[(metadata["characters_count_mean"] < 400) ],
        column_x = "characters_count_mean", column_y = "text_count_tokens_mean",
        column_color = "macro family of languages", title = "Number of characters and number of typographic tokens\n in each book by macro family of languages ",
        outdir = "./../visualizations/", figure_name = "characters_tokens_macro_familiy", figsize = (10,6)
        )
    """


    plt.figure(figsize = figsize)

    ax = sns.scatterplot(x = column_x, y = column_y,
            hue = column_color, data = df,
            palette = palette, alpha = 1, 
            )
    

    ax.set_title(title)
    ax.set_axisbelow(True)

    if xlabel != "":
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(column_x)

    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(column_y)

    plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left')

    fig = ax.get_figure()

    plt.tight_layout()

    #fig.suptitle(title)

    fig.savefig(outdir + figure_name + ".png")

    fig.show()

