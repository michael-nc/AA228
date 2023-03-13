import copy

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import loggamma

def statistics(G: nx.DiGraph, D: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Calculates count and pesudocount for the network given the data"""

    r = D.max().to_numpy()
    q = [np.prod([r[j] for j in preds[1]], dtype=int) for preds in G.pred.items()]
    M = [np.zeros((q[node], r[node])) for node in G]
    
    prior = [np.ones((q[node], r[node])) for node in G]
    
    for index, row in D.iterrows():
        for i in G:
            k = row[i] - 1
            parents = list(G.predecessors(i))
            j = 0
            if len(parents):
                parents_size = np.array(r)[parents]
                coordinate = row[parents] - 1
                j = np.ravel_multi_index(tuple(coordinate), tuple(parents_size))
            M[i][j, k] += 1
            
    return M, prior

def statistics_vectorized(G: nx.DiGraph, D: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Calculates count and pesudocount for the network given the data. Same functionality as statistics shown above but vectorized so much faster."""
    r = D.max().to_numpy()
    q = [np.prod([r[j] for j in preds[1]], dtype=int) for preds in G.pred.items()]
    M = [np.zeros((q[node], r[node])) for node in G]
    
    prior = [np.ones((q[node], r[node])) for node in G]
    
    df2 = D.copy()
    df2.columns = list(range(len(D.columns)))

    for i in G:
        parents = list(G.predecessors(i))
        parents.append(i)
        group = df2.groupby(parents)

        series = group[i].count()

        if isinstance(series.index, pd.MultiIndex):
            multi_index = []
            for node in (parents):
                multi_index.append([value for value in range(1, r[node]+1)])
                
            new_index = pd.MultiIndex.from_product(multi_index)

            new_series = series.reindex(new_index, fill_value=0)

            M[i] += new_series.array.reshape((M[i].shape))
        else:

            new_series = series.reindex(list(range(1, r[i]+1)), fill_value=0)
            M[i][0, :] += new_series.array
            
    return M, prior


def bayesian_score_component(M: list[np.ndarray], alpha: list[np.ndarray]) -> float:
    """Calculates the Bayesian Score for a specific node"""
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p

def bayesian_score(G:nx.DiGraph, D:pd.DataFrame) -> float:
    """Calculates the overall Bayesian Score of the entire network given the data"""
    M, alpha = statistics_vectorized(G, D)
    return np.sum(bayesian_score_component(M[node], alpha[node]) for node in G)