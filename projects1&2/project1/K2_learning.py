import sys
import time
import random

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bayesian_score_calculator import bayesian_score

def write_gph(dag, idx2names, filename):
    """Saves resulting network edge information in a text file with .gph extension and saves the network as a PNG image"""
    # saving the edge information of the resulting network
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

    # drawing and network then saving it using matplotlib
    nx.draw_networkx(dag, labels=idx2names, node_size=400, arrowsize=10, width=1.5, font_size=6)
    plt.tight_layout()
    plt.savefig(f"{filename[:-4]}.png", format="PNG")
    plt.close()

def compute(input_file: str) -> tuple[nx.DiGraph, dict[int: str], float, float]:
    """Reads in the csv file and performs the K2 algorithm on data."""

    # read in csv file and turn into a data frame using Pandas
    df = pd.read_csv(input_file)

    # create mapping between node number and node names
    node_names = {}
    for i, name in enumerate(df.columns):
        node_names[i] = name

    # start timer
    start_time = time.perf_counter()

    # create a DAG network based on the number of variables present
    G = nx.DiGraph()
    G.add_nodes_from(range(len(df.columns)))

    # shuffle the list of nodes
    nodes = list(G.nodes)
    random.shuffle(nodes)

    l = len(nodes)

    # begin K2 algorithm
    for k, i in enumerate(nodes[1:]):
        print(f"{l-k-1} more nodes to go")

        # calcuate a base score
        base_score = bayesian_score(G, df)

        #parents attempt
        parents_attempt = 0
        
        while True:
            best_score, best_j = -np.inf, 0

            # loop through all other nodes
            for j in nodes[:k]:
                
                # if the two nodes are not linked (no edge between)
                if not G.has_edge(nodes[j], nodes[i]):
                    # add an edge between the two nodes
                    G.add_edge(nodes[j], nodes[i])
                    # calculate the new score when the edge is added
                    new_score = bayesian_score(G, df)
                    # if new score is better than the best score so far, update best score and record the parent index 
                    if new_score > best_score:
                        best_score, best_j = new_score, j
                    # remove the added edge and retry with another node
                    G.remove_edge(nodes[j], nodes[i])

                    parents_attempt += 1
                    
            # if the best score is better than the base score, update the base score and add the edge into the network
            if best_score > base_score:
                base_score = best_score
                G.add_edge(nodes[best_j], nodes[i])
            # if best score isn't better than the base score, break and move on to a new node
            else:
                break 

    # calculate elapsed time between the start and end of the K2 algorithm loop
    elapsed_time = time.perf_counter() - start_time

    return G, node_names, base_score, elapsed_time


def main():
    small_data_file = "./data/small.csv"
    medium_data_file = "./data/medium.csv"
    large_data_file = "./data/large.csv"

    small_data_output = "./output/small.gph"
    medium_data_output = "./output/medium.gph"
    large_data_output = "./output/large.gph"
    
    G, node_names, score, time = compute(small_data_file)
    print(f"Running K2 on the small dataset took {time:0.4f} achieving a best score of {score}")
    print("Saving Results")
    write_gph(G, node_names, small_data_output)

    G, node_names, score, time = compute(medium_data_file)
    print(f"Running K2 on the medium dataset took {time:0.4f} achieving a best score of {score}")
    print("Saving Results")
    write_gph(G, node_names, medium_data_output)

    G, node_names, score, time = compute(large_data_file)
    print(f"Running K2 on the large dataset took {time:0.4f} achieving a best score of {score}")
    print("Saving Results")
    write_gph(G, node_names, large_data_output)


if __name__ == '__main__':
    main()
