# -*- coding: utf-8 -*-

#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

#Used from class
import cluster_tools as ct
import errors as err
import importlib
 

def reading_data(filepath):
    '''
    reading_data will create dataframe from file in the given filepath

    Parameters
    ----------
    filepath : STR
        File path or location.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame created from given filepath.

    '''
    df = pd.read_csv(filepath, skiprows=4)
    df = df.set_index('Country Name', drop=True)
    df = df.loc[:, '1960':'2021']

    return df


def transpose(df):
    '''
    transpose create transpose of given dataframe

    Parameters
    ----------
    df  : pandas.DataFrame
        DataFrame for which transpose to be found.

    Returns
    -------
    data_tr : pandas.DataFrame
        Transposed DataFrame of given DataFrame.

    '''
    df_tr = df.transpose()

    return df_tr


def correlation_and_scattermatrix(df):
    '''
    correlation_and_scattermatrix plots correlation matrix and scatter plots
    of data among columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame for which analysis will be done.

    Returns
    -------
    None.

    '''
    corr = df.corr()
    print(corr)
    plt.figure(figsize=(10, 10))
    plt.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation between Years and Countries over Energy Use')
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(df, figsize=(12, 12), s=5, alpha=0.8)
    plt.show()

    return


def cluster_number(df, df_normalised):
    '''
    cluster_number calculates the best number of clusters based on silhouette
    score

    Parameters
    ----------
    df : pandas.DataFrame
        Actual data.
    df_normalised : pandas.DataFrame
        Normalised data.

    Returns
    -------
    INT
        Best cluster number.

    '''

    clusters = []
    scores = []
    # loop over number of clusters
    for ncluster in range(2, 10):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Cluster fitting
        kmeans.fit(df_normalised)
        lab = kmeans.labels_

        # Silhoutte score over number of clusters
        print(ncluster, skmet.silhouette_score(df, lab))

        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(df, lab))

    clusters = np.array(clusters)
    scores = np.array(scores)

    best_ncluster = clusters[scores == np.max(scores)]
    print()
    print('best n clusters', best_ncluster[0])

    return best_ncluster[0]
