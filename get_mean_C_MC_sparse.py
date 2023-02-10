from paper import *
import os


# @jit(nopython=True, fastmath=True, cache=True)
def get_mean_C_MC(links, num_rep, I_start, kappa):
    C = np.zeros((links.shape[0], links.shape[0]), dtype=np.float64)
    for i in tqdm(range(num_rep)):
    # for i in range(num_rep):
    #     print(".")
        C = C + run_dynamics_to_FC(links, I_start, kappa)
    for i in range(len(C)):
        C[i, i] = 0
    return C / num_rep


@jit(nopython=True, fastmath=True, cache=True)
def run_dynamics_to_FC(links, I_start, kappa):
    C = np.zeros((links.shape[0], links.shape[0]), dtype=np.uint64)
    to_add = np.zeros((links.shape[0], links.shape[0]), dtype=np.uint8)
    state = np.ones(len(links), dtype=np.uint8)
    for t in range(I_start):  # iterate over time until I_start to let the system settle
        if np.all(state == 0):
            return C.astype(np.float64) / (9 * I_start)
        ind = np.random.choice(np.array([j for j, site in enumerate(state) if site == 1]))  # choose random active node
        p = np.random.random()
        if p < kappa / (kappa+1):
            for j, is_neighbour in enumerate(links[ind]):  # loop over all other nodes
                if is_neighbour == 1:  # other node is a neighbour
                    if state[j] == 0:  # neighbour is inactive
                        p = np.random.random()
                        if p < kappa:  # infection with rate kappa
                            state[j] = 1  # activate node
        state[ind] = 0  # deactivate chosen node
    for i, site1 in enumerate(state):
        if site1 == 1:
            for j, site2 in enumerate(state):
                if site2 == 1:
                    to_add[i, j] = 1
                    to_add[j, i] = 1
    for t in range(I_start, 10 * I_start, 1):  # iterate over time from I_start on to calculate C
        if np.all(state == 0):
            return C.astype(np.float64) / (9 * I_start)
        ind = np.random.choice(np.array([j for j, site in enumerate(state) if site == 1]))  # choose random active node
        state[ind] = 0  # deactivate chosen node
        to_add[:, ind] = 0
        to_add[ind, :] = 0
        p = np.random.random()
        if p < kappa / (kappa+1):
            for j, is_neighbour in enumerate(links[ind]):  # loop over all other nodes
                if is_neighbour == 1:  # other node is a neighbour
                    if state[j] == 0:  # neighbour is inactive
                        p = np.random.random()
                        if p < kappa:  # infection with rate kappa
                            state[j] = 1  # activate node
                            for i, site in enumerate(state):
                                if site == 1:
                                    to_add[i, j] = 1
                                    to_add[j, i] = 1
        C += to_add
    return C.astype(np.float64) / (9 * I_start)


if __name__ == "__main__":
    exp_name = "MC_C_low_kappa"
    hmn_path = "data/links.npy"
    num_rep = 1000
    I_start = 10000
    # kappa = 0.6
    kappa = 0.522705078125

    os.makedirs(f"results/{exp_name}", exist_ok=True)
    with open(f"results/{exp_name}/metadata.csv", "w") as f:
        f.write("hmn_path;num_rep;I_start;kappa\n")
        f.write(f"{hmn_path};{num_rep};{I_start};{kappa}")

    links = np.load(hmn_path)

    C = get_mean_C_MC(links, num_rep, I_start, kappa)

    np.savetxt(f"results/{exp_name}/C.txt", C)
