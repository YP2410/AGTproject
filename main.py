import math
import random
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib
import pylab as plt
import numpy as np

dictionary = {}

def calc_naive_agent(G: nx.Graph, src, target, bias = 3, curr_sum = 0):
    if src == target:
        return curr_sum
    min_dist_and_bias = math.inf
    min_neighbor = None
    for neighbor in G.neighbors(src):
        try:
            curr_dist = nx.dijkstra_path_length(G,neighbor, target, 'weight') + \
                        bias * nx.dijkstra_path_length(G,src, neighbor, 'weight')
        except:
            curr_dist = math.inf

        if curr_dist < min_dist_and_bias:
            min_neighbor = neighbor
            min_dist_and_bias = curr_dist


    if min_dist_and_bias == math.inf:
        return None
    return calc_naive_agent(G, min_neighbor, target, bias,
                            curr_sum + G.get_edge_data(src, min_neighbor, math.inf)['weight'])


def calc_sophisticated_agent(G: nx.Graph, src, target, bias=3):
    global dictionary

    if src in dictionary:
        return dictionary[src]
    if src == target:
        dictionary[src] = 0
        return 0
    if not G.neighbors(src):
        dictionary[src] = math.inf
        return math.inf
    min_dist_and_bias = math.inf
    min_neighbor = None
    for neighbor in G.neighbors(src):
        try:
            curr_dist = calc_sophisticated_agent(G,neighbor, target, bias) + \
                        bias * nx.dijkstra_path_length(G, src, neighbor, 'weight')

        except:
            curr_dist = math.inf

        if curr_dist < min_dist_and_bias:
            min_neighbor = neighbor
            min_dist_and_bias = curr_dist

    if min_dist_and_bias == math.inf:
        dictionary[src] = math.inf
        return math.inf

    dictionary[src] = calc_sophisticated_agent(G,min_neighbor, target, bias) + nx.dijkstra_path_length(G, src, min_neighbor, 'weight')
    return calc_sophisticated_agent(G,min_neighbor, target, bias) + \
           nx.dijkstra_path_length(G, src, min_neighbor, 'weight')

def calc_partially_naive_agent(G: nx.Graph, src, target, bias_ratio, bias=3):
    if src == target:
        return 0
    if not G.neighbors(src):
        return math.inf
    min_dist_and_bias = math.inf
    min_neighbor = None
    for neighbor in G.neighbors(src):
        try:
            curr_dist = calc_sophisticated_agent(G, neighbor, target, bias*bias_ratio) + \
                        bias * nx.dijkstra_path_length(G, src, neighbor, 'weight')

        except:
            curr_dist = math.inf

        if curr_dist < min_dist_and_bias:
            min_neighbor = neighbor
            min_dist_and_bias = curr_dist

    if min_dist_and_bias == math.inf:
        return math.inf

    return calc_partially_naive_agent(G, min_neighbor, target, bias_ratio, bias) + \
           nx.dijkstra_path_length(G, src, min_neighbor, 'weight')



def run_test():
    global dictionary
    worst_case_pessimistic = 0
    worst_case_optimistic = 0

    j = 0
    large_sum = [[],[],[],[],[]]
    for bias_ratio in [1, 1.1, 1.2, 1.3, 1.5, 1.8, 2.25, 2.75, 3.3333, 4]:
        counter = 0
        i = 0
        sum = np.array([0,0,0,0,0,])
        for n in range(30, 171, 10):
            for k in range(50):
                p = 0.2
                g = erdos_renyi_graph(n, p, seed=None, directed= True)
                DAG = nx.DiGraph([(u,v,{'weight':random.randint(1,200)}) for (u,v) in g.edges() if u<v])
                result_1 = calc_sophisticated_agent(DAG, 0, n-1, 4)
                if result_1 != math.inf:
                    sum[0] = sum[0] + result_1

                    dictionary = {}
                    result_2 = calc_partially_naive_agent(DAG, 0, n-1, 1/bias_ratio, 4)
                    sum[1] = sum[1] + result_2
                    dictionary = {}
                    result_3 = calc_partially_naive_agent(DAG, 0, n - 1, bias_ratio, 4)
                    sum[2] = sum[2] + result_3
                    result_4 = nx.dijkstra_path_length(DAG, 0, n-1, 'weight')
                    sum[3] = sum[3] + result_4
                    result_5 = calc_naive_agent(DAG, 0, n-1, 4)
                    sum[4] = sum[4] + result_5
                dictionary = {}
                i+=1
                if result_2 / result_4 > worst_case_optimistic:
                    worst_case_optimistic = result_2 / result_4

                if result_3 / result_4 > worst_case_pessimistic:
                    worst_case_pessimistic = result_3/ result_4
                counter += 1

        sum = sum / counter
        large_sum[0].append(sum[0])
        large_sum[1].append(sum[1])
        large_sum[2].append(sum[2])
        large_sum[3].append(sum[3])
        large_sum[4].append(sum[4])
        print(large_sum)
        j +=1
    percentage_array = np.array([np.array(large_sum[0])/ np.array(large_sum[3])*100,
                        np.array(large_sum[1])/ np.array(large_sum[3])*100,
                        np.array(large_sum[2])/ np.array(large_sum[3])*100,
                        np.array(large_sum[4]) / np.array(large_sum[3])*100])
    print(percentage_array)
    bias_x = np.array([1, 1.1, 1.2, 1.3, 1.5, 1.8, 2.25, 2.75, 3.3333, 4])
    plt.plot(bias_x, large_sum[0], "-b", label="sophisticated")
    plt.plot(bias_x, large_sum[1], "-r", label="optimistic")
    plt.plot(bias_x, large_sum[2], "-g", label="pessimistic")
    plt.plot(bias_x, large_sum[3], "-y", label="optimally")
    plt.plot(bias_x, large_sum[4], "-k", label="naive")
    plt.legend(loc="upper left")
    plt.savefig('total.pdf')
    print(worst_case_optimistic, worst_case_pessimistic)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_test()