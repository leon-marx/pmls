# The goal of this file is to reproduce the results obtained by Safari et al. in "Persistence of hierarchical network..."
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from animation import DynamicsAnimation
import networkx as nx
import itertools


def plot_HMN(links):
    plt.figure(figsize=(8, 8))
    plt.imshow(links)
    # plt.xticks([])
    # plt.yticks([])
    plt.title(f"HMN connectivity matrix", size=18)
    plt.xlabel("Node", size=14)
    plt.ylabel("Node", size=14)
    plt.show()


def plot_FC(mat, theta = None):
    plt.figure(figsize=(8, 8))
    plt.imshow(mat, vmin=0.0, vmax=1.0)
    # plt.xticks([])
    # plt.yticks([])
    if theta == None:
        plt.title(f"Functional connectivity matrix", size=18)
    else:
        plt.title(f"Binary functional connectivity matrix for theta = {theta}", size=18)
    plt.xlabel("Node", size=14)
    plt.ylabel("Node", size=14)
    plt.colorbar(fraction=0.0455, pad=0.04)
    plt.show()


def plot_dynamics(data):
    # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 16), sharex=True)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(np.mean(data, axis=1))
    ax1.set_title(f"SIS-dynamics", size=18)
    ax1.set_ylabel("Density", size=14)

    ax2.imshow(data.T)
    ax2.set_aspect("auto")
    ax2.set_ylabel("Node", size=14)
    ax2.set_xlabel("Time", size=14)

    plt.show()


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def get_HMN(M0, levels, alpha, verbose=False):
    if (alpha < 4 / (M0 ** 2)) | (alpha > 4):
        print("WARNING! alpha should be in [4/(M0**2), 4]")

    N = int(2 ** levels * M0)
    links = np.zeros((N, N), dtype=np.uint8)

    # connect base modules
    if verbose:
        print("Connecting level 0 ...")
    for base_start in range(0, N, M0):
        for i in range(base_start, base_start + M0, 1):
            for j in range(i+1, base_start + M0, 1):
                links[i, j] = 1
                links[j, i] = 1

    # connect bigger modules
    for i in range(1, levels+1, 1):  # loop over all hierarchies
        if verbose:
            print("Connecting level " + str(i) + " ...")
        for module_start in range(0, N, (2 ** i) * M0):  # loop over every second module at current hierarchy
            for ind1 in range(module_start, module_start + (2 ** (i-1)) * M0, 1):  # loop over elements of first module
                for ind2 in range(module_start + (2 ** (i-1)) * M0, module_start + (2 ** i) * M0, 1):  # loop over elements of second module
                    p = np.random.random()
                    if p < alpha / (4.0 ** i):
                        links[ind1, ind2] = 1  # + i to see color-coding in plot_HMN()
                        links[ind2, ind1] = 1  # + i to see color-coding in plot_HMN()

    return links


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def run_dynamics(links, initial, T, kappa):
    N = len(links)
    data = np.zeros((int(T+1), N), dtype=np.uint8)
    data[0] = initial
    for t in range(T):  # iterate over time
        for i, s in enumerate(data[t]):
            if s == 0:  # site is inactive
                for j, is_neighbour in enumerate(links[i]):  # loop over all other nodes
                    if is_neighbour == 1:  # other node is a neighbour
                        if data[t, j] == 1:  # neighbour is active
                            p = np.random.random()
                            if p < kappa:  # infection with rate kappa
                                data[t+1, i] = 1  # activate node
                                break
            else:  # site is active
                data[t+1, i] = 0  # deactivate
    return data


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def get_FC(slice, delta):
    """
    slice is a binary array of shape (I, N) with
        I: number of timesteps in the slice of the whole dynamics
        N: number of nodes in the HMN
    delta: int, binsize
    """
    # We assume, that the authors used a rolling window instead of actual binning (THIS MIGHT NOT BE CORRECT -> CHECK AGAIN)
    C = np.zeros((slice.shape[1], slice.shape[1]), dtype=np.float64)
    for start in range(0, slice.shape[0]-delta, 1):  # rolling bin over whole array
        to_add = np.zeros((slice.shape[1], slice.shape[1]), dtype=np.uint8)
        for col in slice[start:start+delta]:
            for i, s1 in enumerate(col):
                if s1 == 1:
                    for j, s2 in enumerate(col):
                        if s2 == 1:
                            to_add[i, j] = 1
                            to_add[j, i] = 1
        C = C + to_add
    for i in range(C.shape[0]):
        C[i, i] = 0.0
    return C / (slice.shape[0]-delta)


# @jit(nopython=True, fastmath=True, parallel=True, cache=True)
def get_mean_C(links, initials, num_rep, intervals, delta, kappa):
    C = np.zeros((links.shape[0], links.shape[0]), dtype=np.float64)
    # ratios = (num_rep * np.arange(0.1, 1.1, 0.1)).astype(np.uint64)
    for i in tqdm(range(num_rep)):
        # for r in ratios:
        #     if i == r:
        #         print(int(100 * i / num_rep), "%")
        #         break
        slice = run_dynamics(links, initials[i], intervals[i][1], kappa)[intervals[i][0]:]
        C = C + get_FC(slice, delta)
    return C / num_rep


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def get_A(C, theta):
    return np.where(C > theta, 1, 0).astype(np.uint8)


def get_s1_g(A):
    edgelist = []
    for i, j in itertools.combinations(np.arange(len(A)), 2):
        if A[i, j] == 1:
            edgelist.append((i, j))
    G = nx.Graph()
    G.add_edges_from(edgelist)
    try:
        s1 = len(max(nx.connected_components(G), key=len))
    except ValueError:
        s1 = 0

    w, v = np.linalg.eig(A)
    w = sorted(w)[::-1]
    g = np.real(w[0] - w[1])


    # return s1, g
    return s1, g

if __name__ == "__main__":
    M0 = 2
    levels = 4  # current max: 16
    alpha = 3.0  # must be in [4/M0^2, 4]

    links = get_HMN(M0, levels, alpha)
    # plot_HMN(links)

    # initial = np.random.randint(0, 2, len(links), dtype=np.uint8)
    # T = 2 * len(links)

    # data = run_dynamics(links, initial, T, kappa)
    # plot_dynamics(data)

    # a = DynamicsAnimation(run_dynamics, links, initial, kappa, 3 * len(links))
    # plt.show()

    num_rep = 100
    initials = np.random.randint(0, 2, size=(num_rep, len(links)), dtype=np.uint8)
    a_vals = np.ones(num_rep, dtype=np.uint64) * 100
    b_vals = np.ones(num_rep, dtype=np.uint64) * 500
    intervals = np.stack([a_vals, b_vals], axis=1)
    delta = 4
    kappa = 0.4

    C = get_mean_C(links, initials, num_rep, intervals, delta, kappa)
